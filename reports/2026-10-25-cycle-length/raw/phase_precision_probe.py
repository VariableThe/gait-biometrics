import os, resource, time
from pathlib import Path
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_CPU,(40,45))
import numpy as np
import mpmath as mp
from scipy.signal import butter, sosfiltfilt, sosfilt_zi
start=time.process_time()
x=np.load(Path(__file__).with_name('single-fixture.npy'),allow_pickle=False)
m=np.linalg.norm(x[:3],axis=0)
sos=butter(2,3,btype='lowpass',fs=100,output='sos')
b0,b1,b2,_,a1,a2=map(float,sos[0]); zi=list(map(float,sosfilt_zi(sos)[0]))
def filt(v,coef,init):
    p0,p1,p2,q1,q2=coef
    s0,s1=init[0]*v[0],init[1]*v[0]
    out=[]
    for z in v:
        y=p0*z+s0
        s0=p1*z-q1*y+s1
        s1=p2*z-q2*y
        out.append(y)
    return out
def fb(v,coef,init):
    ext=[2*v[0]-v[i] for i in range(9,0,-1)]+v+[2*v[-1]-v[-i-1] for i in range(1,10)]
    f=filt(ext,coef,init)
    return filt(f[::-1],coef,init)[::-1][9:-9]
coef=[b0,b1,b2,a1,a2]
y=np.array(fb(list(map(float,m)),coef,zi)); ys=sosfiltfilt(sos,m,padtype='odd',padlen=9)
print('hand_float64_bitwise_match',y.tobytes()==ys.tobytes(),'maxabs',np.max(np.abs(y-ys)),'zi',zi)
mp.mp.dps=80
coefh=list(map(mp.mpf,coef)); zih=list(map(mp.mpf,zi))
yh=fb(list(map(mp.mpf,map(float,m))),coefh,zih)
# Separate ideal analytic samples, exact prescribed decimal constants and pi.
def amag(t):
    th=2*mp.pi*(mp.mpf(t)/180+mp.mpf(20)/32)
    return mp.sqrt((mp.mpf('.3')*mp.sin(th))**2+(mp.mpf('.2')*mp.cos(th))**2+(mp.mpf('9.81')-mp.mpf('1.1')*mp.cos(2*th)-mp.mpf('.25')*mp.cos(th))**2)
ma=[amag(t) for t in range(768)]
ya=fb(ma,coefh,zih)
# Exact high-precision steady-state initialization, same binary64 coefficients.
gain=sum(coefh[:3])/(1+coefh[3]+coefh[4]); zie=[gain-coefh[0],coefh[2]-coefh[4]*gain]
yae=fb(ma,coefh,zie)
print('mpmath',mp.__version__,'dps',mp.mp.dps,'dcgain',mp.nstr(gain,25))
print('max_float_vs_hp_savedm',mp.nstr(max(abs(mp.mpf(float(a))-b) for a,b in zip(y,yh)),16))
print('hp_savedm_minima',[i for i in range(10,758) if yh[i]<yh[i-1] and yh[i]<yh[i+1]])
print('hp_analytic_minima',[i for i in range(10,758) if ya[i]<ya[i-1] and ya[i]<ya[i+1]])
for q in (67,157,247,337,427,517,607,697):
    ulp=mp.mpf(float(np.spacing(y[q])))
    print('q',q,'hp_savedm_right-minus-left',mp.nstr(yh[q+1]-yh[q],22),'ULPs',mp.nstr((yh[q+1]-yh[q])/ulp,16),'hp_analytic_gap',mp.nstr(ya[q+1]-ya[q],22),'hp_analytic_exactzi_gap',mp.nstr(yae[q+1]-yae[q],22))
print('raw_ideal_minima_value_d2_per_sample')
for t in (mp.mpf('67.5'),mp.mpf('157.5')):
    print(str(t),mp.nstr(amag(t),25),'d1',mp.nstr(mp.diff(amag,t),8),'d2',mp.nstr(mp.diff(amag,t,2),20),'sample_minus_continuous_min',mp.nstr(amag(t-mp.mpf('.5'))-amag(t),18))
print('cpu_seconds',time.process_time()-start)

