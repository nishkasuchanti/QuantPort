import csv,io,math,time
from dataclasses import dataclass
import numpy as np
from flask import Flask,jsonify,render_template,request
app=Flask(__name__); app.config['MAX_CONTENT_LENGTH']=2*1024*1024
NAMES=['AAPL','MSFT','GOOGL','AMZN','TSLA','JPM','BRK.B','NVDA','META','V','XOM','JNJ','WMT','PG','KO','PEP','COST','ADBE','CRM','NFLX']
@dataclass
class Portfolio:
 weights:list; return_value:float; risk:float; variance:float; sharpe:float
 def json(self,names): return {'weights':self.weights,'return_value':self.return_value,'risk':self.risk,'variance':self.variance,'sharpe':self.sharpe,'allocations':[{'asset':a,'weight':w} for a,w in zip(names,self.weights)]}
def defaults(n):
 r=np.random.default_rng(2527+n); market=r.normal(.00035,.01,420); cols=[]
 for i in range(n): cols.append((80+18*i)*np.exp(np.cumsum(.00005+(i%5)*.000035+(0.55+(i%6)*.1)*market+r.normal(0,.006+(i%4)*.0015,420))))
 return NAMES[:n],np.column_stack(cols)
def csvdata(f):
 rows=list(csv.reader(io.StringIO(f.read().decode('utf-8-sig'))))
 if len(rows)<4 or len(rows[0])<3: raise ValueError('CSV needs Date, at least two asset columns, and three price rows.')
 names=[x.strip() for x in rows[0][1:] if x.strip()]
 if len(names)>20 or len(names)<2 or len(set(names))!=len(names): raise ValueError('Use 2-20 unique asset names.')
 vals=[]
 for line,row in enumerate(rows[1:],2):
  if not row or not any(x.strip() for x in row): continue
  try: v=[float(x) for x in row[1:len(names)+1]]
  except: raise ValueError(f'Invalid price on row {line}.')
  if len(v)!=len(names) or any(x<=0 or not math.isfinite(x) for x in v): raise ValueError(f'Row {line} needs one positive price per asset.')
  vals.append(v)
 if len(vals)<3: raise ValueError('At least three complete price rows are required.')
 return names,np.array(vals)
def repair(raw,cap,long):
 n=len(raw)
 if cap*n<1-1e-9:return None
 w=np.maximum(raw,0) if long else np.clip(raw,-cap,cap); w=w/w.sum() if abs(w.sum())>1e-12 else np.ones(n)/n
 if long:
  for _ in range(n+2):
   over=w>cap
   if not over.any():break
   excess=(w[over]-cap).sum();w[over]=cap;free=~over;room=cap-w[free]
   if room.sum()<1e-12:return None
   w[free]+=excess*room/room.sum()
 return w if abs(w.sum()-1)<1e-7 and w.max()<=cap+1e-7 else None
def evaluate(w,means,cov,rf):
 ret=float(w@means);var=max(0,float(w@cov@w));risk=math.sqrt(var)
 return Portfolio(w.tolist(),ret,risk,var,(ret-rf)/risk if risk>1e-12 else 0)
def quick(a,key):
 if len(a)<2:return a[:]
 p=getattr(a[len(a)//2],key);return quick([x for x in a if getattr(x,key)<p],key)+[x for x in a if getattr(x,key)==p]+quick([x for x in a if getattr(x,key)>p],key)
def merge(a,key):
 if len(a)<2:return a[:]
 m=len(a)//2;l=merge(a[:m],key);r=merge(a[m:],key);o=[]
 while l and r:o.append(l.pop(0) if getattr(l[0],key)<=getattr(r[0],key) else r.pop(0))
 return o+l+r
def bubble(a,key):
 a=a[:]
 for end in range(len(a)-1,0,-1):
  swap=False
  for i in range(end):
   if getattr(a[i],key)>getattr(a[i+1],key):a[i],a[i+1]=a[i+1],a[i];swap=True
  if not swap:break
 return a
def timed(fn,*args):
 t=time.perf_counter_ns();v=fn(*args);return v,(time.perf_counter_ns()-t)/1e6
def linear(a,target,cap=None):return min((p for p in a if cap is None or p.risk<=cap),key=lambda p:abs(p.return_value-target),default=None)
def binary(a,target,cap=None):
 lo=0;hi=len(a)-1
 while lo<=hi:
  m=(lo+hi)//2
  if a[m].return_value<target:lo=m+1
  else:hi=m-1
 for d in range(len(a)):
  valid=[a[i] for i in (lo-d-1,lo+d) if 0<=i<len(a) and (cap is None or a[i].risk<=cap)]
  if valid:return min(valid,key=lambda p:abs(p.return_value-target))
def frontier(a):
 out=[];best=-1e99
 for p in sorted(a,key=lambda x:(x.risk,-x.return_value)):
  if p.return_value>best:out.append(p);best=p.return_value
 return out
@app.get('/')
def home():return render_template('index.html')
@app.get('/api/health')
def health():return jsonify(status='ok')
@app.post('/api/optimise')
def optimise():
 try:
  d=request.form;n=max(2,min(20,int(d.get('assets',6))));count=max(100,min(20000,int(d.get('portfolios',5000))));cap=float(d.get('weight_cap',.4));rf=float(d.get('risk_free',.042));riskcap=float(d['risk_cap']) if d.get('risk_cap') else None;target=float(d.get('target_return',.1));long=d.get('long_only','true')=='true'
  names,prices=csvdata(request.files['file']) if request.files.get('file') and request.files['file'].filename else defaults(n);n=len(names)
  if not 1/n-1e-9<=cap<=1:raise ValueError(f'Weight cap must be at least {100/n:.1f}% for {n} assets.')
  daily=prices[1:]/prices[:-1]-1;means=daily.mean(0)*252;cov=np.cov(daily,rowvar=False)*252;rng=np.random.default_rng();ports=[]
  attempts=0
  max_attempts=count*50
  while len(ports)<count and attempts<max_attempts:
   attempts+=1
   w=repair(rng.exponential(1,n) if long else rng.normal(1/n,.25,n),cap,long)
   if w is not None:
    p=evaluate(w,means,cov,rf)
    if riskcap is None or p.risk<=riskcap:ports.append(p)
  if not ports:raise ValueError('No feasible portfolios matched. Increase the risk or weight cap.')
  if len(ports)<count:raise ValueError(f'Only {len(ports)} valid portfolios could be generated. Relax the constraints.')
  obj=d.get('objective','sharpe');key={'return':'return_value','risk':'risk','target':'return_value'}.get(obj,'sharpe');sorter={'quick':quick,'merge':merge,'bubble':bubble}.get(d.get('sort_method'),quick);rank,sortms=timed(sorter,ports,key);rank=rank if obj=='risk' else rank[::-1];builtin,builtinms=timed(lambda:sorted(ports,key=lambda p:getattr(p,key)),);byret=merge(ports,'return_value');lin,linms=timed(linear,ports,target,riskcap);bis,bins=timed(binary,byret,target,riskcap);optimal=(bis if d.get('search_method')=='binary' else lin) if obj=='target' else rank[0];ef=frontier(ports);sample=ports[::max(1,len(ports)//500)][:500]
  return jsonify(assets=names,optimal=optimal.json(names),rankings=[p.json(names) for p in rank[:6]],points=[[p.risk,p.return_value,p.sharpe] for p in sample],frontier=[[p.risk,p.return_value] for p in ef],generated=len(ports),means=means.tolist(),covariance=cov.tolist(),benchmark={'selected_sort':sortms,'builtin_sort':builtinms,'linear_search':linms,'binary_search':bins})
 except (ValueError,UnicodeDecodeError) as e:return jsonify(error=str(e)),400
if __name__=='__main__':app.run(debug=True)


