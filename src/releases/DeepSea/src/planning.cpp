// Prioritized optimal Bellman backups on the supplied graph. KeyCorridor uses
// exact (time,state,action) keys; untouched values reside in the shared Qold.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
using I = int64_t;
struct Entry { double priority; I key; };
struct Less { bool operator()(const Entry&a,const Entry&b)const {
  return a.priority==b.priority ? a.key>b.key : a.priority<b.priority;
}};
struct Planner {
  int S,A,H,kind,base_type; double gamma;
  const void* base; const int64_t* next; const uint8_t* terminal;
  const uint8_t* valid; const double* rewards; const int64_t* depth;
  std::vector<std::vector<I>> predecessors;
  std::unordered_map<I,double> values, priorities;
  std::vector<Entry> heap;
  I reads=0, backups=0;
  Planner(int s,int a,int h,int k,double g,const void*b,int bt,
          const int64_t*n,const uint8_t*t,const uint8_t*v,const double*r,const int64_t*d)
   :S(s),A(a),H(h),kind(k),base_type(bt),gamma(g),base(b),next(n),terminal(t),valid(v),rewards(r),depth(d),predecessors(s) {
    for(I i=0;i<I(S)*A;i++) if(valid[i]&&!terminal[i]&&next[i]>=0) predecessors[next[i]].push_back(i);
  }
  double q(I key)const {auto it=values.find(key);if(it!=values.end())return it->second;
    return base_type==4?static_cast<const float*>(base)[key]:static_cast<const double*>(base)[key];}
  double target(I key) {
    I edge=key%(I(S)*A),time=key/(I(S)*A);++reads;
    if(kind==1){
      if(terminal[edge])return 1.-.9*(time+1)/H;
      if(time+1>=H)return 0.;
    } else if(terminal[edge])return rewards[edge];
    I successor=next[edge];double best=-INFINITY;
    I offset=(kind==1?(time+1)*S:0)+successor;
    for(int a=0;a<A;a++)if(valid[successor*A+a])best=std::max(best,q(offset*A+a));
    return (kind==1?0.:rewards[edge])+gamma*best;
  }
  void offer(I key) {
    double p=std::abs(target(key)-q(key));
    if(p<=1e-10){priorities.erase(key);return;}
    priorities[key]=p;heap.push_back({p,key});std::push_heap(heap.begin(),heap.end(),Less{});
  }
  void run(const I* seeds,int n,int budget) {
    for(int j=0;j<n;j++)offer(seeds[j]);
    int count=0;
    while(!heap.empty()&&count<budget){
      std::pop_heap(heap.begin(),heap.end(),Less{});Entry e=heap.back();heap.pop_back();
      auto it=priorities.find(e.key);if(it==priorities.end()||it->second!=e.priority)continue;
      priorities.erase(it);double y=target(e.key),old=q(e.key);
      if(std::abs(y-old)<=1e-10)continue;
      values[e.key]=y;++count;++backups;
      I time=e.key/(I(S)*A),state=(e.key/A)%S;
      if(kind==1&&time==0)continue;
      for(I edge:predecessors[state]){
        if(kind==1&&time-1<depth[edge/A])continue;
        offer((kind==1?(time-1)*S*A:0)+edge);
      }
    }
    // Remove stale queue entries without changing valid priority ordering.
    if(heap.size()>4*priorities.size()+10000){
      heap.clear();for(auto& kv:priorities)heap.push_back({kv.second,kv.first});
      std::make_heap(heap.begin(),heap.end(),Less{});
    }
  }
};
extern "C" {
void* ps_new(int s,int a,int h,int k,double gamma,const void*b,int bt,
             const int64_t*n,const uint8_t*t,const uint8_t*v,const double*r,const int64_t*d){
 return new Planner(s,a,h,k,gamma,b,bt,n,t,v,r,d);
}
void ps_free(void*p){delete static_cast<Planner*>(p);}
void ps_run(void*p,const I*keys,int n,int budget){static_cast<Planner*>(p)->run(keys,n,budget);}
void ps_get(void*p,const I*keys,int n,double*out){auto*x=static_cast<Planner*>(p);for(int i=0;i<n;i++)out[i]=x->q(keys[i]);}
void ps_sizes(void*p,I*out){auto*x=static_cast<Planner*>(p);out[0]=x->values.size();out[1]=x->priorities.size();out[2]=x->reads;out[3]=x->backups;}
void ps_export(void*p,I*vk,double*vv,I*pk,double*pv){auto*x=static_cast<Planner*>(p);I i=0;for(auto&kv:x->values){vk[i]=kv.first;vv[i++]=kv.second;}i=0;for(auto&kv:x->priorities){pk[i]=kv.first;pv[i++]=kv.second;}}
void ps_restore(void*p,const I*vk,const double*vv,I nv,const I*pk,const double*pv,I np,I reads,I backups){
 auto*x=static_cast<Planner*>(p);x->values.clear();x->priorities.clear();x->heap.clear();
 for(I i=0;i<nv;i++)x->values[vk[i]]=vv[i];
 for(I i=0;i<np;i++){x->priorities[pk[i]]=pv[i];x->heap.push_back({pv[i],pk[i]});}
 std::make_heap(x->heap.begin(),x->heap.end(),Less{});x->reads=reads;x->backups=backups;
}
}
