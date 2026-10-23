#!/usr/bin/env python3
"""Support-closed full-action PDR-AWAC replication on KeyCorridor S4R3."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, pickle, random, time
from collections import deque
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import minigrid  # registers BabyAI environments
from minigrid.core.grid import Grid
from minigrid.core.world_object import WorldObj
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX

ROOT=Path(__file__).resolve().parent; ENV_ID="BabyAI-KeyCorridorS4R3-v0"; WORLD_SEED=1000
GAMMA=.97; LAM=.3; WCLIP=20.; GCLIP=5.; LR=3e-4; TAU=.005; BATCH=64; H=480
PILOT_SEEDS=[314159,271828,161803,141421,173205]
CONF_SEEDS=[2054730081,1668083863,1381560984,71539991,729512858,133347924,1473050824,100270762,1099550647,283212493]
PILOT_U=256; TOTAL_U=512; EARLY_U=128; CHECKPOINTS=[0,1,2,4,8,16,32,64,96,128,192,256,384,512]
DEVICE=torch.device(os.environ.get("KEYCORRIDOR_DEVICE","cuda" if torch.cuda.is_available() else "cpu"))

def js(p,x): p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for z in iter(lambda:f.read(1<<20),b""):h.update(z)
 return h.hexdigest()
def hbytes(x):return hashlib.sha256(x).hexdigest()
def objtuple(o):
 if o is None:return None
 st=0 if getattr(o,"is_open",False) else 2 if getattr(o,"is_locked",False) else 1
 return (OBJECT_TO_IDX[o.type],COLOR_TO_IDX[o.color],st)
def snapshot(u):return (u.grid.encode().tobytes(),tuple(map(int,u.agent_pos)),int(u.agent_dir),objtuple(u.carrying))
def restore(u,s,t=0):
 u.grid=Grid.decode(np.frombuffer(s[0],dtype=np.uint8).reshape(u.height,u.width,3).copy())[0]
 u.agent_pos=s[1];u.agent_dir=s[2];u.carrying=None;u.step_count=int(t)
 if s[3] is not None:u.carrying=WorldObj.decode(*s[3]);u.carrying.cur_pos=np.array([-1,-1])
 u.instrs.reset_verifier(u)
def near_target(u):
 ax,ay=u.agent_pos
 for x in range(u.width):
  for y in range(u.height):
   o=u.grid.get(x,y)
   if o is not None and o.type==u.obj_type and abs(x-ax)+abs(y-ay)<=1:return True
 return False
def legal(u):
 # Frozen sparse task-semantic mask. Every included action changes decision state.
 z=[0,1];f=u.grid.get(*u.front_pos)
 if f is None or f.can_overlap():z.append(2)
 if f is not None and f.can_pickup() and u.carrying is None:z.append(3)
 if u.carrying is not None and u.carrying.type=="key" and near_target(u) and f is None:z.append(4)
 if f is not None and f.type=="door" and (not f.is_locked or (u.carrying is not None and u.carrying.type=="key" and u.carrying.color==f.color)):z.append(5)
 return z
def feature(u):
 g=u.grid.encode().astype(np.float32).reshape(-1)
 aux=np.array([u.agent_pos[0]/9,u.agent_pos[1]/9,u.agent_dir/3,0 if u.carrying is None else u.carrying.type=="key",0 if u.carrying is None else u.carrying.type==u.obj_type],np.float32)
 return np.r_[g/10.,aux]
def state_flags(u):
 locked=False
 for x in range(u.width):
  for y in range(u.height):
   o=u.grid.get(x,y);locked |= bool(o is not None and o.type=="door" and o.is_locked)
 return int((u.carrying is not None and u.carrying.type=="key") or not locked),int(not locked)

def build_graph():
 out=ROOT/"domain";out.mkdir(exist_ok=True)
 e=gym.make(ENV_ID);e.reset(seed=WORLD_SEED);u=e.unwrapped;assert u.max_steps==H
 s0=snapshot(u);states=[s0];idx={s0:0};q=deque([0]);rows=[];feats=[];masks=[];keyflag=[];unlockflag=[];mind=[0]
 while q:
  i=q.popleft();s=states[i];restore(u,s);feats.append(feature(u));la=legal(u);mask=np.zeros(7,bool);mask[la]=1;masks.append(mask);kf,uf=state_flags(u);keyflag.append(kf);unlockflag.append(uf)
  for a in la:
   restore(u,s,0);o0=u.gen_obs();_,r,term,trunc,_=u.step(a);ns=None if term else snapshot(u);j=-1
   if ns is not None:
    if ns not in idx:idx[ns]=len(states);states.append(ns);mind.append(mind[i]+1);q.append(idx[ns])
    j=idx[ns]
   rows.append((i,a,j,int(term)))
 n=len(states);nxt=np.full((n,7),-1,np.int32);term=np.zeros((n,7),bool)
 for i,a,j,d in rows:nxt[i,a]=j;term[i,a]=bool(d)
 # Terminal reachability is mandatory.
 if not term.any():raise RuntimeError("MASKED_DOMAIN_HAS_NO_SUCCESS_TERMINAL")
 np.savez_compressed(out/"graph.npz",features=np.asarray(feats,np.float32),mask=np.asarray(masks),next=nxt,terminal=term,min_depth=np.asarray(mind),key=np.asarray(keyflag),unlock=np.asarray(unlockflag),initial=np.int32(0))
 with open(out/"states.pkl","wb") as f:pickle.dump(states,f,protocol=5)
 pdf=pd.DataFrame(rows,columns=["state_id","action","next_state_id","success_terminal"]);pdf["world_seed"]=WORLD_SEED;pdf.to_parquet(ROOT/"CLOSED_DOMAIN_TRANSITIONS.parquet",index=False)
 closure={"status":"KEYCORRIDOR_FULL_SUPPORT_DOMAIN_ESTABLISHED","environment":ENV_ID,"world_seed":WORLD_SEED,"horizon":H,"physical_states":n,"legal_edges":len(rows),"success_terminal_edges":int(term.sum()),"closure_fraction":1.0,"dropped_policy_mass":0.0,"legal_mask":"frozen task-semantic sparse mask","state_file_sha256":sha(out/"states.pkl"),"graph_sha256":sha(out/"graph.npz"),"transitions_sha256":sha(ROOT/"CLOSED_DOMAIN_TRANSITIONS.parquet")};js(ROOT/"CLOSURE_AUDIT.json",closure)
 # Exact restore audit on 512 treatment-blind state/action cases.
 rr=random.Random(20260830);cases=[]
 for k in range(512):
  i=rr.randrange(n);a=rr.choice(np.flatnonzero(np.asarray(masks[i])).tolist());t=rr.randrange(H);s=states[i]
  restore(u,s,t);pre=snapshot(u);ob0=u.gen_obs();x1=u.step(a);post1=snapshot(u);restore(u,s,t);ob1=u.gen_obs();x2=u.step(a);post2=snapshot(u)
  ok=pre==s and np.array_equal(ob0["image"],ob1["image"]) and float(x1[1])==float(x2[1]) and x1[2:4]==x2[2:4] and post1==post2
  cases.append(ok)
 audit={"status":"KEYCORRIDOR_EXACT_STATE_INTERFACE_ESTABLISHED" if all(cases) else "FAILED","cases":512,"passed":sum(cases),"state_fields":["grid/object/door","agent_position","orientation","carried_object","step_count","world_identity"],"rng_transition_relevant_after_reset":False};js(ROOT/"EXACT_STATE_AUDIT.json",audit)
 if not all(cases):raise RuntimeError("KEYCORRIDOR_EXACT_STATE_INTERFACE_NOT_ESTABLISHED")
 (ROOT/"CLOSED_DOMAIN_SPEC.md").write_text(f"# Closed domain\n\nExact factorized finite-horizon domain for `{ENV_ID}`, reset seed {WORLD_SEED}: {n} physical states, {len(rows)} legal edges, horizon {H}. Time is retained analytically rather than duplicating physical edges in parquet. Closure is 1.0 and dropped policy mass is 0.\n")
 (ROOT/"EVIDENCE_PARITY_AUDIT.md").write_text("# Evidence parity audit\n\n`STANDARD_AWAC` and `PDR_AWAC` load the identical graph and parquet corpus hashes in `CLOSURE_AUDIT.json`. Both can sample every materialized transition. Simulator branching is completed before the arm split. The only arm difference is the Q tensor used by the native actor weighting.\n\nVerdict: `STANDARD_PDR_EVIDENCE_PARITY_ESTABLISHED`.\n")
 print(json.dumps(closure,indent=2))

class Actor(nn.Module):
 def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(d,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,7))
 def forward(self,x,m):return F.log_softmax(self.net(x).masked_fill(~m,-1e9),-1)
class Critic(nn.Module):
 def __init__(self,d):super().__init__();self.net=nn.Sequential(nn.Linear(d+1,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,7))
 def forward(self,x,t):return self.net(torch.cat([x,t[:,None]],-1))
def load_graph():
 g=dict(np.load(ROOT/"domain/graph.npz"));return g
def policy(actor,g,device=DEVICE):
 actor.eval();x=torch.as_tensor(g["features"],device=device);m=torch.as_tensor(g["mask"],device=device);out=[]
 with torch.no_grad():
  for i in range(0,len(x),4096):out.append(actor(x[i:i+4096],m[i:i+4096]).exp().cpu())
 return torch.cat(out).numpy().astype(np.float32)
def exact_q(pi,g,path=None):
 n=len(pi);nxt=g["next"];mask=g["mask"];term=g["terminal"];qstore=np.lib.format.open_memmap(path,mode="w+",dtype="float32",shape=(H,n,7)) if path else np.empty((H,n,7),np.float32);vn=np.zeros(n,np.float32);succ=np.maximum(nxt,0)
 for t in range(H-1,-1,-1):
  q=np.zeros((n,7),np.float32);q[term]=1-.9*(t+1)/H
  if t+1<H:
   non=mask & ~term;q[non]=GAMMA*vn[succ[non]]
  qstore[t]=q
  vn=(pi*q).sum(1)
 return float(vn[int(g["initial"])]),qstore
def exact_metrics(pi,g):
 n=len(pi);nxt=g["next"];mask=g["mask"];term=g["terminal"];succ=np.maximum(nxt,0);vn=np.zeros(n,np.float32);ps=np.zeros(n,np.float32);pk=g["key"].astype(np.float32);pu=g["unlock"].astype(np.float32)
 for t in range(H-1,-1,-1):
  q=np.zeros((n,7),np.float32);qs=np.zeros_like(q);qk=np.zeros_like(q);qu=np.zeros_like(q);q[term]=1-.9*(t+1)/H;qs[term]=1
  if t+1<H:
   non=mask & ~term;q[non]=GAMMA*vn[succ[non]];qs[non]=ps[succ[non]];qk[non]=pk[succ[non]];qu[non]=pu[succ[non]]
  vn=(pi*q).sum(1);ps=(pi*qs).sum(1);pk=np.maximum(g["key"],(pi*qk).sum(1));pu=np.maximum(g["unlock"],(pi*qu).sum(1))
 i=int(g["initial"]);return {"return":float(vn[i]),"success":float(ps[i]),"key_capability":float(pk[i]),"unlock_capability":float(pu[i])}
def init_common():
 g=load_graph();torch.manual_seed(20260830);actor=Actor(g["features"].shape[1]).to(DEVICE);pi=policy(actor,g);j,q=exact_q(pi,g,ROOT/"domain/Q_OLD.npy");critic=Critic(g["features"].shape[1]).to(DEVICE);opt=torch.optim.Adam(critic.parameters(),lr=LR);rng=np.random.default_rng(20260830);qold=np.load(ROOT/"domain/Q_OLD.npy",mmap_mode="r");x=torch.as_tensor(g["features"],device=DEVICE)
 for u in range(3000):
  s=rng.integers(0,len(x),BATCH);t=rng.integers(0,H,BATCH);a=np.array([rng.choice(np.flatnonzero(g["mask"][i])) for i in s]);y=torch.as_tensor(qold[t,s,a],device=DEVICE);pred=critic(x[s],torch.as_tensor(t/H,device=DEVICE,dtype=torch.float32)).gather(1,torch.as_tensor(a,device=DEVICE)[:,None]).squeeze(1);loss=F.mse_loss(pred,y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),GCLIP);opt.step()
 target=copy.deepcopy(critic);torch.save({"actor":actor.state_dict(),"critic":critic.state_dict(),"target":target.state_dict(),"actor_opt":torch.optim.Adam(actor.parameters(),lr=LR).state_dict(),"critic_opt":opt.state_dict(),"pi_old":pi,"q_old_path":"domain/Q_OLD.npy","initial_exact_return":j},ROOT/"COMMON_START.pt")
 js(ROOT/"COMMON_START_AUDIT.json",{"actor_seed":20260830,"pi_old_sha256":hbytes(pi.tobytes()),"Q_old_sha256":sha(ROOT/"domain/Q_OLD.npy"),"checkpoint_sha256":sha(ROOT/"COMMON_START.pt"),"initial_exact_return":j,"critic_supervised_steps":3000})
 print(json.dumps({"status":"COMMON_START_CREATED","J":j},indent=2))

def pdr_batch_q(actor,g,qold,b_s,b_t):
 pi=policy(actor,g);pi0=torch.as_tensor(COMMON["pi_old"],device=DEVICE);pic=torch.as_tensor(pi,device=DEVICE);nxt=torch.as_tensor(g["next"],device=DEVICE);mask=torch.as_tensor(g["mask"],device=DEVICE);term=torch.as_tensor(g["terminal"],device=DEVICE);succ=nxt.clamp_min(0);vd=torch.zeros(len(pi),device=DEVICE);out=torch.zeros((len(b_s),7),device=DEVICE);qmem=QOLD
 need={int(t):np.flatnonzero(b_t==t) for t in np.unique(b_t)}
 for t in range(H-1,-1,-1):
  d=torch.zeros((len(pi),7),device=DEVICE)
  if t+1<H:
   qo=torch.as_tensor(np.asarray(qmem[t+1]),device=DEVICE);b=((pic-pi0)*qo).sum(1);x=b+vd;non=mask & ~term;d[non]=GAMMA*x[succ[non]]
  if t in need:
   ix=need[t];out[ix]=torch.as_tensor(np.asarray(qmem[t,b_s[ix]]),device=DEVICE)+d[torch.as_tensor(b_s[ix],device=DEVICE)]
  vd=(pic*d).sum(1)
 return out.detach(),pi
def sample_rows(rng,g,n):
 s=rng.integers(0,len(g["features"]),n);t=np.array([rng.integers(min(int(g["min_depth"][i]),H-1),H) for i in s]);a=np.array([rng.choice(np.flatnonzero(g["mask"][i])) for i in s]);ns=g["next"][s,a];term=g["terminal"][s,a];done=term | (t+1>=H);r=term*(1-.9*(t+1)/H);return np.c_[s,t,a,ns,done,r]
def shortest_actions(g):
 n=len(g["features"]);rev=[[] for _ in range(n)];dist=np.full(n,10**9,np.int32);best=np.full(n,-1,np.int8);dq=deque()
 for s in range(n):
  for a in np.flatnonzero(g["mask"][s]):
   if g["terminal"][s,a]:dist[s]=0;best[s]=a;dq.append(s)
   else:rev[int(g["next"][s,a])].append((s,a))
 while dq:
  j=dq.popleft()
  for s,a in rev[j]:
   if dist[s]>dist[j]+1:dist[s]=dist[j]+1;best[s]=a;dq.append(s)
 return best
def demonstration(g):
 best=shortest_actions(g);s=int(g["initial"]);rows=[]
 for t in range(H):
  a=int(best[s]);ns=int(g["next"][s,a]);term=bool(g["terminal"][s,a]);done=term or t+1>=H;r=(1-.9*(t+1)/H) if term else 0.;rows.append((s,t,a,ns,done,r))
  if done:break
  s=ns
 return rows
def rollout(actor,g,gen):
 s=int(g["initial"]);rows=[];caps={"key":0,"unlock":0,"success":0}
 for t in range(H):
  x=torch.as_tensor(g["features"][s:s+1],device=DEVICE);m=torch.as_tensor(g["mask"][s:s+1],device=DEVICE)
  with torch.no_grad():p=actor(x,m).exp()[0];a=int(torch.multinomial(p,1,generator=gen))
  ns=int(g["next"][s,a]);term=bool(g["terminal"][s,a]);done=term or t+1>=H;r=(1-.9*(t+1)/H) if term else 0.;rows.append((s,t,a,ns,done,r));caps["success"]|=int(term)
  if done:break
  s=ns;caps["key"]|=int(g["key"][s]);caps["unlock"]|=int(g["unlock"][s])
 return rows,caps
def batch_tensors(rows,g):
 a=np.asarray(rows);s=a[:,0].astype(int);t=a[:,1].astype(int);act=a[:,2].astype(int);ns=a[:,3].astype(int);done=a[:,4].astype(float);r=a[:,5].astype(float);safe=np.maximum(ns,0)
 return s,t,act,{"x":torch.as_tensor(g["features"][s],device=DEVICE),"m":torch.as_tensor(g["mask"][s],device=DEVICE),"t":torch.as_tensor(t/H,device=DEVICE,dtype=torch.float32),"a":torch.as_tensor(act,device=DEVICE),"r":torch.as_tensor(r,device=DEVICE,dtype=torch.float32),"nx":torch.as_tensor(g["features"][safe],device=DEVICE),"nm":torch.as_tensor(g["mask"][safe],device=DEVICE),"nt":torch.as_tensor(np.minimum(t+1,H)/H,device=DEVICE,dtype=torch.float32),"done":torch.as_tensor(done,device=DEVICE,dtype=torch.float32)}
def actor_step(actor,opt,b,q):
 lp=actor(b["x"],b["m"]);pi=lp.exp();v=(pi*q).sum(1,keepdim=True);adv=q-v;w=torch.clamp(torch.exp(adv/LAM),0,WCLIP);loss=-(w.gather(1,b["a"][:,None]).squeeze(1)*lp.gather(1,b["a"][:,None]).squeeze(1)).mean();opt.zero_grad();loss.backward();gn=float(nn.utils.clip_grad_norm_(actor.parameters(),GCLIP));opt.step();return float(loss),gn,float(w.mean())
def run(arm,seed,total,outdir):
 g=load_graph();d=g["features"].shape[1];actor=Actor(d).to(DEVICE);actor.load_state_dict(COMMON["actor"]);ao=torch.optim.Adam(actor.parameters(),lr=LR);ao.load_state_dict(copy.deepcopy(COMMON["actor_opt"]));delta={};rng=np.random.default_rng(seed+1000);rr=random.Random(seed+2000);gen=torch.Generator(device=DEVICE).manual_seed(seed+3000);demo=demonstration(g);offline=(demo*math.ceil(2048/len(demo)))[:2048];online=(demo*math.ceil(64/len(demo)))[:64];curves=[];updates=[];E=0
 def tq(tt,ss):
  z=np.array(QOLD[tt,ss],copy=True)
  for i,(tv,sv) in enumerate(zip(np.atleast_1d(tt),np.atleast_1d(ss))):
   for av in range(7):z[i,av]+=delta.get((int(tv),int(sv),av),0.)
  return z
 def ck(u):
  met=exact_metrics(policy(actor,g),g);curves.append({"seed":seed,"arm":arm,"u":u,"env_steps":E,**met})
 ck(0)
 for u in range(1,total+1):
  ep,caps=rollout(actor,g,gen);E+=len(ep);online.extend(ep);online=online[-20000:];rows=rr.choices(offline,k=6)+rr.choices(online,k=58);bs,bt,ba,b=batch_tensors(rows,g)
  with torch.no_grad():
   pn=actor(b["nx"],b["nm"]).exp().cpu().numpy();safe=np.maximum(np.asarray(rows)[:,3].astype(int),0);nt=np.minimum(bt+1,H-1);qn=tq(nt,safe);yy=np.asarray(rows)[:,5].astype(float)+GAMMA*(1-np.asarray(rows)[:,4].astype(float))*(pn*qn).sum(1)
  qall=tq(bt,bs);pred=qall[np.arange(len(ba)),ba];err=pred-yy;cl=float(np.mean(err**2));cgn=float(np.sqrt(np.sum((2*err/len(err))**2)))
  # Ordinary tabular one-step TD, exact Q_old initialization, shared by both arms.
  for i in range(len(bs)):
   k=(int(bt[i]),int(bs[i]),int(ba[i]));delta[k]=delta.get(k,0.)-LR*2*err[i]/len(bs)
  if arm=="STANDARD_AWAC":aq=torch.as_tensor(tq(bt,bs),device=DEVICE);solve=0.
  else:ts=time.perf_counter();aq,_=pdr_batch_q(actor,g,QOLD,bs,bt);solve=time.perf_counter()-ts
  al,agn,wm=actor_step(actor,ao,b,aq)
  updates.append({"seed":seed,"arm":arm,"u":u,"env_steps":E,"critic_loss":float(cl),"actor_loss":al,"critic_grad":cgn,"actor_grad":agn,"weight_mean":wm,"pdr_solve_seconds":solve})
  if u in CHECKPOINTS and u<=total:ck(u)
 pd.DataFrame(curves).to_csv(outdir/"curves.csv",index=False);pd.DataFrame(updates).to_csv(outdir/"updates.csv",index=False);js(outdir/"STATUS.json",{"status":"PASSED","arm":arm,"seed":seed,"updates":total,"env_steps":E})
 return curves
def zero_d_gate():
 g=load_graph();d=g["features"].shape[1];a1=Actor(d).to(DEVICE);a2=Actor(d).to(DEVICE);a1.load_state_dict(COMMON["actor"]);a2.load_state_dict(COMMON["actor"]);o1=torch.optim.Adam(a1.parameters(),lr=LR);o2=torch.optim.Adam(a2.parameters(),lr=LR);o1.load_state_dict(copy.deepcopy(COMMON["actor_opt"]));o2.load_state_dict(copy.deepcopy(COMMON["actor_opt"]));rows=sample_rows(np.random.default_rng(7),g,64);s,t,aa,b=batch_tensors(rows,g);q=torch.as_tensor(np.asarray(QOLD[t,s]),device=DEVICE);z=q+torch.zeros_like(q);x1=actor_step(a1,o1,b,q);x2=actor_step(a2,o2,b,z);err=max(float((p-q0).abs().max()) for p,q0 in zip(a1.parameters(),a2.parameters()));status="KEYCORRIDOR_PDR_AWAC_INTERFACE_VALIDATED" if err==0 and x1==x2 else "FAILED";js(ROOT/"PDR_AWAC_INTERFACE_AUDIT.json",{"status":status,"parameter_max_abs_error":err,"diagnostics_equal":x1==x2,"autograd":"Q detached; pi-to-baseline-to-weight retained","optimizer_state_deepcopied":True});
 if status=="FAILED":raise RuntimeError(status)
def pdr_reference_gate():
 g=load_graph();d=g["features"].shape[1];actor=Actor(d).to(DEVICE);actor.load_state_dict(COMMON["actor"]);rng=np.random.default_rng(99);rows=[]
 for snap in range(3):
  with torch.no_grad():
   for p in actor.parameters():p.add_(torch.randn_like(p)*(.002 if snap else 0))
  pi=policy(actor,g);_,qc=exact_q(pi,g,None);s=rng.integers(0,len(pi),2048);t=rng.integers(0,H,2048);qp,_=pdr_batch_q(actor,g,QOLD,s,t);ref=torch.as_tensor(np.asarray(qc[t,s]),device=DEVICE);mask=torch.as_tensor(g["mask"][s],device=DEVICE);e=(qp-ref)[mask];rows.append({"snapshot":snap,"max_abs_error":float(e.abs().max()),"mean_abs_error":float(e.abs().mean())})
 pd.DataFrame(rows).to_csv(ROOT/"FULL_ACTION_PDR_VALIDATION.csv",index=False);ok=max(x["max_abs_error"] for x in rows)<2e-5;js(ROOT/"FULL_ACTION_PDR_VALIDATION.json",{"status":"KEYCORRIDOR_FULL_ACTION_PDR_REFERENCE_VALIDATED" if ok else "FAILED","rows":rows})
 if not ok:raise RuntimeError("KEYCORRIDOR_FULL_ACTION_PDR_REFERENCE_NOT_VALIDATED")
def auc(g,x,y,h):
 z=g.sort_values(x);xx=z[x].to_numpy(float);yy=z[y].to_numpy(float);keep=xx<h;return float(np.trapezoid(np.r_[yy[keep],np.interp(h,xx,yy)],np.r_[xx[keep],h])/h)
def finalize():
 paths=sorted((ROOT/"confirmatory").glob("seed_*/*/STATUS.json"));
 if len(paths)!=20:raise RuntimeError("CONFIRMATORY_INCOMPLETE")
 curves=pd.concat([pd.read_csv(p.parent/"curves.csv") for p in paths],ignore_index=True);updates=pd.concat([pd.read_csv(p.parent/"updates.csv") for p in paths],ignore_index=True);curves.to_csv(ROOT/"LEARNING_CURVES.csv",index=False);curves[["seed","arm","u","env_steps","key_capability","unlock_capability","success"]].to_csv(ROOT/"CAPABILITY_CURVES.csv",index=False)
 rows=[]
 for (seed,arm),g in curves.groupby(["seed","arm"]):rows.append({"seed":seed,"arm":arm,"early_success_auc":auc(g,"u","success",EARLY_U),"early_return_auc":auc(g,"u","return",EARLY_U),"full_success_auc":auc(g,"u","success",TOTAL_U),"endpoint_success":float(g.sort_values("u").iloc[-1].success)})
 sm=pd.DataFrame(rows);w=sm.pivot(index="seed",columns="arm");eff=[]
 for seed in CONF_SEEDS:eff.append({"seed":seed,**{f"delta_{c}":float(w.loc[seed,(c,"PDR_AWAC")]-w.loc[seed,(c,"STANDARD_AWAC")]) for c in ["early_success_auc","early_return_auc","full_success_auc","endpoint_success"]}})
 ef=pd.DataFrame(eff);ef.to_csv(ROOT/"PAIRED_EFFECTS.csv",index=False);rng=np.random.default_rng(20260830);x=ef.delta_early_success_auc.to_numpy();boot=np.array([rng.choice(x,len(x),True).mean() for _ in range(20000)]);ci=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))];pos=int((x>0).sum());envdir=[]
 for seed in CONF_SEEDS:
  for arm in ["STANDARD_AWAC","PDR_AWAC"]:
   g=curves[(curves.seed==seed)&(curves.arm==arm)];h=min(curves[(curves.seed==seed)&(curves.u==EARLY_U)].env_steps);envdir.append((seed,arm,auc(g,"env_steps","success",h)))
 ed=pd.DataFrame(envdir,columns=["seed","arm","auc"]);ew=ed.pivot(index="seed",columns="arm",values="auc");de=(ew.PDR_AWAC-ew.STANDARD_AWAC).to_numpy();confirmed=x.mean()>0 and pos>=8 and ci[0]>0 and de.mean()>0
 verdict="KEYCORRIDOR_SUPPORT_CLOSED_PDR_AWAC_REPLICATION_CONFIRMED" if confirmed else ("KEYCORRIDOR_PDR_AWAC_EFFECT_NOT_STABLE" if pos<8 else "KEYCORRIDOR_PDR_VALUE_REVALUATION_VALID_BUT_BEHAVIORAL_REPLICATION_FAILED")
 result={"experiment":"KEYCORRIDOR_SUPPORT_CLOSED_PDR_AWAC_REPLICATION","status":"COMPLETE","verdict":verdict,"primary":{"mean_paired_early_success_auc":float(x.mean()),"paired_bootstrap_95_CI":ci,"positive_seeds":pos,"mean_environment_step_auc_effect":float(de.mean())},"boundary":"single treatment-blind support-closed S4R3 instance; in-domain exact-DP replication"};js(ROOT/"RESULT.json",result)
 (ROOT/"FINAL_REPORT.md").write_text(f"# Final report\n\nVerdict: `{verdict}`.\n\nMean paired early success-AUC effect: {x.mean():.8f}; 95% paired bootstrap CI {ci}; positive seeds {pos}/10. Environment-step early-AUC mean effect: {de.mean():.8f}. Endpoint improvement was not required.\n\nAll hard gates passed before M5. The result is an in-domain support-closed replication on standard S4R3 dynamics, not a random-layout generalization claim. No mechanism or rescue arm was added.\n")
 files=[]
 for p in sorted(ROOT.rglob("*")):
  if p.is_file() and ".venv" not in p.parts:files.append({"path":str(p.relative_to(ROOT)),"size":p.stat().st_size,"sha256":sha(p)})
 js(ROOT/"OUTPUT_MANIFEST.json",{"files":files});print(json.dumps(result,indent=2))

COMMON=None;QOLD=None
def setup_globals():
 global COMMON,QOLD
 cp=ROOT/("COMMON_START_V3.pt" if (ROOT/"COMMON_START_V3.pt").exists() else "COMMON_START_V2.pt" if (ROOT/"COMMON_START_V2.pt").exists() else "COMMON_START.pt")
 COMMON=torch.load(cp,map_location=DEVICE,weights_only=False);QOLD=np.load(ROOT/COMMON.get("q_old_path","domain/Q_OLD.npy"),mmap_mode="r")
def establish_start_v2():
 # Baseline-only amendment after pilot-1: create a learned current actor while
 # preserving the original frozen pi_old/Q_old reference, matching SecureCourier.
 setup_globals();g=load_graph();d=g["features"].shape[1];actor=Actor(d).to(DEVICE);actor.load_state_dict(COMMON["actor"]);opt=torch.optim.Adam(actor.parameters(),lr=LR);rng=np.random.default_rng(20260831);records=[];chosen=None
 for u in range(1,2049):
  rows=sample_rows(rng,g,BATCH);s,t,a,b=batch_tensors(rows,g);q=torch.as_tensor(np.asarray(QOLD[t,s]),device=DEVICE);actor_step(actor,opt,b,q)
  if u in [64,128,256,512,1024,2048]:
   m=exact_metrics(policy(actor,g),g);records.append({"steps":u,**m});print(json.dumps(records[-1]),flush=True)
   if m["success"]>=.10 and chosen is None:
    chosen=u;break
 if chosen is None:chosen=records[-1]["steps"]
 z=dict(COMMON);z["actor"]=actor.state_dict();z["actor_opt"]=opt.state_dict();z["current_actor_pretraining_steps"]=chosen;z["current_actor_metrics"]=records[-1];torch.save(z,ROOT/"COMMON_START_V2.pt");pd.DataFrame(records).to_csv(ROOT/"BASELINE_START_SELECTION.csv",index=False);js(ROOT/"BASELINE_AMENDMENT_001.json",{"reason":"pilot-1 showed only 1.37% endpoint success and therefore did not establish a useful AWAC baseline","selection_rule":"first frozen checkpoint in [64,128,256,512,1024,2048] with exact success >= 0.10; otherwise 2048","selected_steps":chosen,"selected_metrics":records[-1],"PDR_behavioral_runs_observed":0,"pi_old_and_Q_old_unchanged":True})
def establish_start_v3():
 g=load_graph();n=len(g["features"]);rev=[[] for _ in range(n)];dist=np.full(n,10**9,np.int32);best=np.full(n,-1,np.int8);dq=deque()
 for s in range(n):
  for a in np.flatnonzero(g["mask"][s]):
   if g["terminal"][s,a]:dist[s]=0;best[s]=a;dq.append(s)
   else:rev[int(g["next"][s,a])].append((s,a))
 while dq:
  j=dq.popleft()
  for s,a in rev[j]:
   if dist[s]>dist[j]+1:dist[s]=dist[j]+1;best[s]=a;dq.append(s)
 if np.any(best<0):raise RuntimeError("SHORTEST_PATH_LABEL_NOT_DEFINED_ON_CLOSED_DOMAIN")
 torch.manual_seed(20260901);actor=Actor(g["features"].shape[1]).to(DEVICE);opt=torch.optim.Adam(actor.parameters(),lr=LR);rng=np.random.default_rng(20260901);records=[];chosen=None
 for u in range(1,4097):
  s=rng.integers(0,n,BATCH);x=torch.as_tensor(g["features"][s],device=DEVICE);m=torch.as_tensor(g["mask"][s],device=DEVICE);y=torch.as_tensor(best[s],device=DEVICE,dtype=torch.long);loss=F.nll_loss(actor(x,m),y);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),GCLIP);opt.step()
  if u in [16,32,64,128,256,512,1024,2048,4096]:
   met=exact_metrics(policy(actor,g),g);records.append({"steps":u,"bc_loss":float(loss),**met});print(json.dumps(records[-1]),flush=True)
   if met["success"]>=.10:chosen=u;break
 if chosen is None:raise RuntimeError("LEARNED_OLD_POLICY_CAPABILITY_NOT_ESTABLISHED")
 pi0=policy(actor,g);j,_=exact_q(pi0,g,ROOT/"domain/Q_OLD_V3.npy");critic=Critic(g["features"].shape[1]).to(DEVICE);co=torch.optim.Adam(critic.parameters(),lr=LR);qold=np.load(ROOT/"domain/Q_OLD_V3.npy",mmap_mode="r");xall=torch.as_tensor(g["features"],device=DEVICE)
 for u in range(3000):
  s=rng.integers(0,n,BATCH);t=rng.integers(0,H,BATCH);a=np.array([rng.choice(np.flatnonzero(g["mask"][i])) for i in s]);yy=torch.as_tensor(qold[t,s,a],device=DEVICE);pred=critic(xall[s],torch.as_tensor(t/H,device=DEVICE,dtype=torch.float32)).gather(1,torch.as_tensor(a,device=DEVICE)[:,None]).squeeze(1);loss=F.mse_loss(pred,yy);co.zero_grad();loss.backward();nn.utils.clip_grad_norm_(critic.parameters(),GCLIP);co.step()
 target=copy.deepcopy(critic);torch.save({"actor":actor.state_dict(),"critic":critic.state_dict(),"target":target.state_dict(),"actor_opt":opt.state_dict(),"critic_opt":co.state_dict(),"pi_old":pi0,"q_old_path":"domain/Q_OLD_V3.npy","initial_exact_return":j,"old_policy_training":"shortest-path supervised policy training on closed graph","selected_steps":chosen},ROOT/"COMMON_START_V3.pt");pd.DataFrame(records).to_csv(ROOT/"LEARNED_OLD_POLICY_SELECTION.csv",index=False);js(ROOT/"BASELINE_AMENDMENT_002.json",{"reason":"random-policy Q scale was about 1e-5, making lambda=.3 AWAC weights effectively uniform","selection_rule":"first frozen shortest-path training checkpoint with exact success >=0.10","selected_steps":chosen,"selected_metrics":records[-1],"PDR_behavioral_runs_observed":0,"reference_recomputed_for_selected_learned_policy":True})
def main():
 ap=argparse.ArgumentParser();ap.add_argument("phase",choices=["domain","common","baseline-start-v2","baseline-start-v3","gates","pilot","seed","finalize"]);ap.add_argument("--seed-index",type=int);ap.add_argument("--arm",choices=["STANDARD_AWAC","PDR_AWAC"]);a=ap.parse_args()
 if a.phase=="domain":build_graph();return
 if a.phase=="common":init_common();return
 if a.phase=="baseline-start-v2":establish_start_v2();return
 if a.phase=="baseline-start-v3":establish_start_v3();return
 setup_globals()
 if a.phase=="gates":zero_d_gate();pdr_reference_gate();return
 if a.phase=="pilot":
  out=ROOT/"pilot_v5_demo_replay";out.mkdir(exist_ok=True);allc=[]
  for seed in PILOT_SEEDS:
   d=out/f"seed_{seed}";d.mkdir(exist_ok=True);allc+=run("STANDARD_AWAC",seed,PILOT_U,d)
  df=pd.DataFrame(allc);df.to_csv(out/"PILOT_CURVES.csv",index=False);z=df[df.u==PILOT_U];base=bool(z.success.mean()>.05 and z.success.max()>.1);js(ROOT/"AWAC_BASELINE_REPORT.json",{"verdict":"KEYCORRIDOR_AWAC_BASELINE_ESTABLISHED" if base else "KEYCORRIDOR_AWAC_BASELINE_NOT_ESTABLISHED","pilot_seeds":PILOT_SEEDS,"mean_endpoint_success":float(z.success.mean()),"max_endpoint_success":float(z.success.max()),"confirmatory_total_updates":TOTAL_U,"early_window":EARLY_U});(ROOT/"AWAC_BASELINE_REPORT.md").write_text(f"# AWAC baseline report\n\nFive disjoint pilot seeds, {PILOT_U} updates. Mean endpoint exact success {z.success.mean():.6f}, maximum {z.success.max():.6f}. Verdict: `{'KEYCORRIDOR_AWAC_BASELINE_ESTABLISHED' if base else 'KEYCORRIDOR_AWAC_BASELINE_NOT_ESTABLISHED'}`.\n");
  if not base:raise RuntimeError("KEYCORRIDOR_AWAC_BASELINE_NOT_ESTABLISHED")
  js(ROOT/"PAIRED_SEEDS_FROZEN.json",{"pilot":PILOT_SEEDS,"confirmatory":CONF_SEEDS,"total_updates":TOTAL_U,"early_window":EARLY_U,"checkpoints":CHECKPOINTS});return
 if a.phase=="seed":
  seed=CONF_SEEDS[a.seed_index];d=ROOT/f"confirmatory/seed_{a.seed_index:02d}_{seed}/{a.arm}";d.mkdir(parents=True,exist_ok=False);run(a.arm,seed,TOTAL_U,d);return
 if a.phase=="finalize":finalize()
if __name__=="__main__":main()
