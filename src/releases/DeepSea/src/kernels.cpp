#include <vector>
#include <cstdint>
#include <algorithm>

// Full finite-horizon differential. No depth truncation, policy caching,
// new reference, or actor change. float32 arithmetic matches the native lane.
extern "C" void key_solve(int n, int h, int b, const int64_t* next,
 const uint8_t* live, const float* pi, const float* oldpi, const float* oldq,
 const int64_t* bs, const int64_t* bt, float gamma, float* out) {
 std::vector<float> v(n,0), nv(n), source(n), d(n*7);
 std::vector<std::vector<int>> groups(h);
 for(int i=0;i<b;++i) groups[bt[i]].push_back(i);
 for(int t=h-1;t>=0;--t) {
  std::fill(d.begin(),d.end(),0);
  if(t+1<h) {
   for(int s=0;s<n;++s) {
    float z=0;
    for(int a=0;a<7;++a) z+=(pi[s*7+a]-oldpi[s*7+a])*oldq[((int64_t)(t+1)*n+s)*7+a];
    source[s]=z;
   }
   for(int j=0;j<n*7;++j) if(live[j]) d[j]=gamma*(source[next[j]]+v[next[j]]);
  }
  for(int i:groups[t]) for(int a=0;a<7;++a)
   out[i*7+a]=oldq[((int64_t)t*n+bs[i])*7+a]+d[bs[i]*7+a];
  for(int s=0;s<n;++s) {
   float z=0; for(int a=0;a<7;++a) z+=pi[s*7+a]*d[s*7+a]; nv[s]=z;
  }
  v.swap(nv);
 }
}

// Full reverse topological evaluation of the DeepSea differential.
extern "C" void deep_solve(int n, const int64_t* next, const uint8_t* live,
 const double* pi, const double* source, const double* oldq, double* out) {
 std::vector<double> v(n,0);
 for(int s=n-1;s>=0;--s) {
  double d[2]={0,0};
  for(int a=0;a<2;++a) {
   int j=s*2+a;
   if(live[j]) d[a]=source[next[j]]+v[next[j]];
   out[j]=oldq[j]+d[a];
  }
  v[s]=pi[s*2]*d[0]+pi[s*2+1]*d[1];
 }
}
