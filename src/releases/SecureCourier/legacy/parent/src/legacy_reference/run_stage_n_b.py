#!/usr/bin/env python3
"""Run one frozen Phase-N closed-loop S/C/D/P continuation."""
from __future__ import annotations
import argparse,gzip,hashlib,json,math,os,pickle,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parents[1]; ROOT=HERE.parents[2]; DIY=ROOT/'diy_task_set'; BRIDGE=HERE.parent
sys.path[:0]=[str(DIY),str(BRIDGE/'scripts')]
from agents.actor import Actor
from agents.critic import Critic
from scripts.run_phase1 import ReplayBuffer,collate_batch,collect_online_episode
from secure_courier.generator import generate_pilot_split
from secure_courier.oracle import dp_values
from secure_courier.state import EnvState
from secure_courier.task_graph import GRAPH
import run_actor_causal_bridge as bridge

ELIGIBLE=(1,2,3,4,6,7,9,10); CONDITIONS=('S','C','D','P'); TIMES=(0,50,100,200,300,400,500)
GAMMA=.97; LAMBDA=.3; CLIP=20.; TAU=.005; MAX_STEPS=40; OBS_MAX=128; ACT_MAX=32
MANIFEST=HERE/'STAGE_N_A_ACCEPTED_CHECKPOINT_MANIFEST.json'; RUNS=HERE/'stage_n_b_runs'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def module_hash(m):
 h=hashlib.sha256()
 for n,v in sorted(m.state_dict().items()):
  x=v.detach().cpu().contiguous();h.update(n.encode());h.update(str(x.dtype).encode());h.update(np.asarray(x.shape,dtype=np.int64).tobytes());h.update(x.numpy().tobytes())
 return h.hexdigest()
def object_hash(x):
 h=hashlib.sha256()
 def visit(v):
  if torch.is_tensor(v):
   z=v.detach().cpu().contiguous();h.update(b't');h.update(str(z.dtype).encode());h.update(np.asarray(z.shape,dtype=np.int64).tobytes());h.update(z.numpy().tobytes())
  elif isinstance(v,dict):
   h.update(b'd');
   for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
  elif isinstance(v,(list,tuple)):
   h.update(b'l');
   for q in v:visit(q)
  else:h.update(repr(v).encode())
 visit(x);return h.hexdigest()
def source(seed):
 x=json.loads(MANIFEST.read_text()); rows={int(r['seed']):r for r in x['checkpoints']}
 if seed not in rows:raise RuntimeError('seed not in frozen accepted manifest')
 return rows[seed],Path(rows[seed]['run_dir'])
def load_models(ck,device):
 with (ck/'vocab.pkl').open('rb') as f:vocab=pickle.load(f)
 a=Actor(vocab.size,32,64).to(device);c=Critic(vocab.size,32,64).to(device);t=Critic(vocab.size,32,64).to(device)
 a.load_state_dict(torch.load(ck/'actor.pt',map_location=device,weights_only=True));c.load_state_dict(torch.load(ck/'critic.pt',map_location=device,weights_only=True));t.load_state_dict(torch.load(ck/'target_critic.pt',map_location=device,weights_only=True))
 ao=torch.optim.Adam(a.parameters(),lr=3e-4);co=torch.optim.Adam(c.parameters(),lr=3e-4)
 ao.load_state_dict(torch.load(ck/'actor_optimizer.pt',map_location=device,weights_only=True));co.load_state_dict(torch.load(ck/'critic_optimizer.pt',map_location=device,weights_only=True))
 return a,c,t,ao,co,vocab
def load_offline(path):
 out=[]
 with path.open() as f:
  for i,line in enumerate(f):r=json.loads(line);r['replay_index']=i;r['replay_source']='offline';out.append(r)
 if len(out)!=20000:raise RuntimeError('offline replay is not 20000 rows')
 return out
def node_by_action():
 d={}
 for n,edges in GRAPH.items():
  for a,_ in edges:
   if a in d and d[a]!=n:raise RuntimeError('nonunique structural action')
   d[a]=n
 return d
NODE=node_by_action(); DEPTH={'depot':0,'sc2':10,'sc3':11,'sc4':12,'sc5':13}
def augment_episode(rows,wall,stream_start):
 first=NODE[rows[0]['action_structural']];base=DEPTH[first];out=[]
 for j,r0 in enumerate(rows):
  r=dict(r0);r['transition_id']=f'n_b_wall{wall}_t{j}';r['episode_id']=f'n_b_wall{wall}';r['replay_source']='online';r['stream_transition_index']=stream_start+j;r['full_state']={'node':NODE[r['action_structural']],'step_count':base+j,'done':False,'terminal_success':False,'instance_id':r['instance_id']};out.append(r)
 return out,first
def greedy(actor,vocab,inst,node,step,device,force=None):
 return bridge.run_greedy_from_state(actor,vocab,inst,EnvState(node=node,step_count=step,done=False,terminal_success=False,instance_id=inst.instance_id),device,force)
def stochastic_schedule_eval(probs_by_inst,heldout,eval_seed):
 rng=np.random.Generator(np.random.PCG64(eval_seed));choices=rng.integers(0,len(heldout),size=100);uniforms=rng.random((100,MAX_STEPS));rets=[];succ=[];shortcut=[]
 for e,ii in enumerate(choices):
  node='depot';ret=0.;ok=False;sc=False
  for t in range(MAX_STEPS):
   tab=probs_by_inst[heldout[int(ii)].instance_id][node];acts=list(tab);u=uniforms[e,t];cum=0.;a=acts[-1]
   for q in acts:
    cum+=tab[q]
    if u<cum:a=q;break
   sc=sc or a=='take_shortcut'
   for aa,nxt in GRAPH[node]:
    if aa==a:
     reward=1. if node=='drop_box_open' and a=='put_parcel' else 0.;node=nxt;break
   if reward:ret+=(GAMMA**t)*reward;ok=True;break
  rets.append(ret);succ.append(ok);shortcut.append(sc)
 return {'n':100,'discounted_return_mean':float(np.mean(rets)),'success_rate':float(np.mean(succ)),'shortcut_utilization':float(np.mean(shortcut)),'schedule_seed':eval_seed,'schedule_sha256':canonical({'instances':choices.tolist(),'uniforms':uniforms.tolist()})}
def evaluate(actor,vocab,heldout,device,eval_seed):
 actor.eval();rows=[];tabs={}
 for inst in heldout:
  probs=bridge.policy_probs_by_node(actor,vocab,inst,device);tabs[inst.instance_id]=probs;policy=lambda node,actions,p=probs:{a:p[node][a] for a in actions};V,_=dp_values(policy,gamma=GAMMA,max_steps=MAX_STEPS)
  full,sc=greedy(actor,vocab,inst,'depot',0,device);gate,_=greedy(actor,vocab,inst,'gate_lobby',14,device);safe,_=greedy(actor,vocab,inst,'junction',8,device,'take_safe_route');j=probs['junction'];g=probs['gate_lobby'];b=probs['gate_badged'];c=probs['gate_coded']
  rows.append({'instance_id':inst.instance_id,'exact_dp_start_value':float(V[('depot',MAX_STEPS)]),'full_task_success':float(full),'shortcut_utilization':float(sc),'gate_sequence_success':float(gate),'safe_route_success':float(safe),'junction_p_shortcut':j['take_shortcut'],'junction_p_safe':j['take_safe_route'],'junction_shortcut_minus_safe_probability':j['take_shortcut']-j['take_safe_route'],'junction_shortcut_minus_safe_logit':math.log(max(j['take_shortcut'],1e-30))-math.log(max(j['take_safe_route'],1e-30)),'gate_p_scan_badge':g['scan_badge'],'gate_p_wait_at_gate':g['wait_at_gate'],'gate_scan_minus_wait_probability':g['scan_badge']-g['wait_at_gate'],'gate_scan_minus_wait_logit':math.log(max(g['scan_badge'],1e-30))-math.log(max(g['wait_at_gate'],1e-30)),'enter_code_probability':b['enter_code'],'enter_code_greedy_correct':float(max(b,key=b.get)=='enter_code'),'open_gate_probability':c['open_gate'],'open_gate_greedy_correct':float(max(c,key=c.get)=='open_gate')})
 actor.train();return {'instances':rows,'stochastic_rollouts':stochastic_schedule_eval(tabs,heldout,eval_seed)}
def six_oracle(actor,vocab,rows,instances,device):
 vals=[]
 for r in rows:
  inst=instances[r['instance_id']];p=bridge.policy_probs_by_node(actor,vocab,inst,device);policy=lambda node,actions,p=p:{a:p[node][a] for a in actions};V,Q=dp_values(policy,gamma=GAMMA,max_steps=MAX_STEPS);n=r['full_state']['node'];h=MAX_STEPS-int(r['full_state']['step_count']);vals.append(Q[(n,r['action_structural'],h)]-V[(n,h)])
 return torch.tensor(vals,dtype=torch.float32,device=device)
def update(condition,a,c,t,ao,co,batch,samples,candidates,placebos,vocab,instances,device):
 obs,acts,ai,rew,nobs,nacts,done=(batch[k] for k in ('obs_tokens','act_tokens','action_idx','reward','next_obs_tokens','next_act_tokens','done'))
 with torch.no_grad():qn=t(nobs,nacts);pn=a(nobs,nacts).exp();y=rew+GAMMA*(pn*qn).sum(-1)*(1-done)
 q=c(obs,acts);qsa=q.gather(1,ai[:,None]).squeeze(1);cl=F.mse_loss(qsa,y);co.zero_grad(set_to_none=True);cl.backward();torch.nn.utils.clip_grad_norm_(c.parameters(),5.);co.step()
 oracle=six_oracle(a,vocab,samples[:6],instances,device)
 a.train()
 with torch.no_grad():qa=c(obs,acts);pi=a(obs,acts).exp();adv=qa.gather(1,ai[:,None]).squeeze(1)-(pi*qa).sum(-1);std=torch.clamp(torch.exp(adv/LAMBDA),0,CLIP);applied=std.clone();ow=torch.clamp(torch.exp(oracle/LAMBDA),0,CLIP);idx=[int(r['replay_index']) for r in samples[:6]];cp=[i for i,x in enumerate(idx) if x in candidates];pp=[i for i,x in enumerate(idx) if x in placebos]
 if condition=='C' and cp:applied[torch.tensor(cp,device=device)]=ow[cp]
 if condition=='P' and pp:applied[torch.tensor(pp,device=device)]=ow[pp]
 if condition=='D' and cp:applied[torch.tensor(cp,device=device)]=0.
 logp=a(obs,acts).gather(1,ai[:,None]).squeeze(1);al=-(applied.detach()*logp).mean();ao.zero_grad(set_to_none=True);al.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),5.);ao.step();t.soft_update(c,tau=TAU)
 if condition=='S' and not torch.equal(applied,std):raise RuntimeError('S intervention')
 return {'critic_loss':float(cl),'actor_loss':float(al),'offline_replay_indices':idx,'online_stream_indices':[int(r['stream_transition_index']) for r in samples[6:]],'candidate_positions':cp,'placebo_positions':pp,'standard_weights_offline':[float(x) for x in std[:6]],'oracle_advantages_offline':[float(x) for x in oracle],'oracle_weights_offline':[float(x) for x in ow],'applied_weights_offline':[float(x) for x in applied[:6]],'oracle_dp_sweeps':6}
def save_state(path,a,c,t,ao,co,rng):
 torch.save({'actor':a.state_dict(),'critic':c.state_dict(),'target_critic':t.state_dict(),'actor_optimizer':ao.state_dict(),'critic_optimizer':co.state_dict(),'python_rng':rng.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},path)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--acquisition-seed',type=int,required=True);ap.add_argument('--replicate',type=int,choices=(1,2),required=True);ap.add_argument('--condition',choices=CONDITIONS,required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
 if args.acquisition_seed not in ELIGIBLE:raise RuntimeError('unregistered acquisition seed')
 cont=700000+100*args.acquisition_seed+args.replicate;out=RUNS/f'seed{args.acquisition_seed}'/f'rep{args.replicate}'/args.condition;out.mkdir(parents=True,exist_ok=True);rp=out/'result.json'
 if rp.exists() and json.loads(rp.read_text()).get('status')=='COMPLETE':print('already complete');return
 if any(out.iterdir()):raise RuntimeError('refusing to overwrite incomplete run')
 rec,src=source(args.acquisition_seed);ck=src/'checkpoints/online_final';offline_path=src/'offline_dataset.jsonl';stale_path=src/'stage_n_a_stale_audit/stale_manifest.json';stale=json.loads(stale_path.read_text());candidates=frozenset(map(int,stale['candidate_replay_indices']));placebos=frozenset(map(int,stale['placebo_replay_indices']))
 if candidates&placebos or len(candidates)!=len(placebos) or not stale['coverage_complete']:raise RuntimeError('invalid per-seed manifest')
 for n,h in rec['checkpoint_hashes'].items():
  if sha(ck/n)!=h:raise RuntimeError('checkpoint hash mismatch')
 if sha(offline_path)!=rec['offline_dataset_sha256']:raise RuntimeError('dataset hash mismatch')
 device=torch.device(args.device);random.seed(cont);np.random.seed(cont);torch.manual_seed(cont)
 if device.type=='cuda':torch.cuda.manual_seed_all(cont)
 rng=random.Random(cont);start=time.time();a,c,t,ao,co,vocab=load_models(ck,device);initial={'actor':module_hash(a),'critic':module_hash(c),'target_critic':module_hash(t),'actor_optimizer':object_hash(ao.state_dict()),'critic_optimizer':object_hash(co.state_dict())};offline=load_offline(offline_path);online=ReplayBuffer(20000);stream=[];updates=[];episodes=[]
 split=generate_pilot_split(seed_train=1000,seed_val=2000,seed_test=3000,n_train=16,n_val=8,n_test=8);train=split['train'];held=split['val']+split['test'];instances={x.instance_id:x for x in train+held};snap={};ckhash={}
 for wall in range(0,501):
  if wall in TIMES:
   snap[str(wall)]=evaluate(a,vocab,held,device,cont+900000+wall);sp=out/f'state_step{wall}.pt';save_state(sp,a,c,t,ao,co,rng);ckhash[sp.name]=sha(sp);print(f'seed={args.acquisition_seed} rep={args.replicate} condition={args.condition} wall={wall} updates={len(updates)}',flush=True)
  if wall==500:break
  raw=collect_online_episode(train,a,vocab,device,rng,GAMMA,MAX_STEPS,OBS_MAX,ACT_MAX,gate_curriculum_prob=.8,gate_epsilon=1.);rows,first=augment_episode(raw,wall+1,len(stream));stream.extend(rows)
  for r in rows:online.add_transition(r)
  episodes.append({'wall_step':wall+1,'first_node':first,'n_transitions':len(rows),'success':bool(any(r['reward']>0 for r in rows)),'stream_start':len(stream)-len(rows),'stream_end':len(stream)})
  if len(online)>=64:
   samples=rng.choices(offline,k=6)+online.sample(58,rng);batch=collate_batch(samples,vocab,OBS_MAX,ACT_MAX,device);m=update(args.condition,a,c,t,ao,co,batch,samples,candidates,placebos,vocab,instances,device);m['wall_step']=wall+1;m['update_index']=len(updates)+1
   if not updates:m['first_update_critic_hash_after']=module_hash(c)
   updates.append(m)
 with gzip.open(out/'trajectories.jsonl.gz','wt') as f:
  for r in stream:f.write(json.dumps(r,sort_keys=True)+'\n')
 with gzip.open(out/'updates.jsonl.gz','wt') as f:
  for r in updates:f.write(json.dumps(r,sort_keys=True)+'\n')
 (out/'episodes.json').write_text(json.dumps(episodes,indent=2,sort_keys=True)+'\n')
 artifacts={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}
 result={'status':'COMPLETE','acquisition_seed':args.acquisition_seed,'replicate':args.replicate,'continuation_seed':cont,'condition':args.condition,'physical_gpu':int(os.environ.get('PHYSICAL_GPU','-1')),'started_at_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.time()-start,'source_run':str(src),'source_checkpoint_hashes':rec['checkpoint_hashes'],'offline_dataset_sha256':sha(offline_path),'stale_manifest_sha256':sha(stale_path),'candidate_index_sha256':stale['candidate_index_int64_sha256'],'placebo_index_sha256':stale['placebo_index_int64_sha256'],'candidate_count':len(candidates),'placebo_count':len(placebos),'initial_state_hashes':initial,'snapshots':snap,'checkpoint_hashes':ckhash,'artifact_hashes_pre_result':artifacts,'n_wall_steps':500,'environment_episodes':500,'environment_transitions':len(stream),'optimizer_updates':len(updates),'offline_samples':6*len(updates),'online_samples':58*len(updates),'oracle_dp_sweeps':6*len(updates),'critic_intervention':False,'replay_rule':'6 offline + 58 own closed-loop online with replacement','update_order':'critic->six detached exact DP sweeps->actor->target','runner_sha256':sha(Path(__file__)),'authorization_sha256':sha(HERE/'STAGE_N_B_AUTHORIZATION.md')}
 rp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print('COMPLETE',rp,flush=True)
if __name__=='__main__':main()
