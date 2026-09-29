"""Null hierarchy: flat O-gauge surrogates built in kb dims, analysed at k <= kb (layer from argv)."""
import sys, time, json, numpy as np, holo as H
t0=time.time()
layer=int(sys.argv[1]); kbs=[int(x) for x in sys.argv[2].split(',')]; ks=[64,128,256]
Xf, Xt = H.load('dev', layer), H.load('devtest', layer)
A0,T0,P0,g0 = H.project(Xf, Xt, max(kbs))
def stats(A,T,k):
    A=A[:,:,:k].copy(); T=T[:,:,:k].copy(); A-=A.mean(1,keepdims=True)
    R,W = H.fit_maps(A)
    S = np.matmul(T.transpose(0,2,1),T); trS=np.trace(S,axis1=1,axis2=2)
    cs=[];prof=np.zeros(k); rho=[]
    for b,p,q in H.OTRI:
        D=R[b,p]@R[p,q]-R[b,q]; SD=S[b]@D; cs.append((D*SD).sum()/trS[b])
        prof += np.diag(D.T@SD)/trS[b]
    for i in range(12):
        for j in range(12):
            if i!=j: rho.append(((T[i]@R[i,j]-T[j])**2).sum()/trS[j])
    prof/=len(H.OTRI)
    q4 = [float(prof[a:b].sum()) for a,b in ((0,k//4),(k//4,k//2),(k//2,3*k//4),(3*k//4,k))]
    return dict(c=float(np.mean(cs)), rho=float(np.mean(rho)), quart=q4)
res={}
for k in ks:
    res[f'real_k{k}']=stats(A0,T0,k); print('real',k,res[f'real_k{k}'],flush=True)
for kb in kbs:
    A=A0[:,:,:kb]; T=T0[:,:,:kb]
    R,_=H.fit_maps(A)
    mdl = H.model_O(A,T,R)
    An,Tn,Pn,gn = H.surrogate(Xf,Xt,g0,np.ascontiguousarray(P0[:,:,:kb]),mdl,seed=11)
    for k in ks:
        if k<=kb:
            res[f'O{kb}_k{k}']=stats(An,Tn,k); print('O',kb,k,res[f'O{kb}_k{k}'],'infl',round(mdl['infl'],3),f'[{time.time()-t0:.0f}s]',flush=True)
    del An,Tn,Pn,mdl
json.dump(res, open(f'out/nullhier_L{layer}.json','w'), indent=1)
