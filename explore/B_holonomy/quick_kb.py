import time, numpy as np, holo as H
t0=time.time()
layer=14
Xf, Xt = H.load('dev', layer), H.load('devtest', layer)
A0,T0,P0,g0 = H.project(Xf, Xt, 256)
print('proj', time.time()-t0, flush=True)
def cp(A,T,k):
    A=A[:,:,:k].copy(); T=T[:,:,:k].copy(); A-=A.mean(1,keepdims=True)
    R,W = H.fit_maps(A)
    S = np.matmul(T.transpose(0,2,1),T); trS=np.trace(S,axis1=1,axis2=2)
    cs=[];cr=[]
    for b,p,q in H.OTRI:
        D=R[b,p]@R[p,q]-R[b,q]; cs.append((D*(S[b]@D)).sum()/trS[b])
        D=W[b,p]@W[p,q]-W[b,q]; cr.append((D*(S[b]@D)).sum()/trS[b])
    return np.mean(cs), np.mean(cr)
for k in (64,128,256):
    print('real k',k, cp(A0,T0,k), flush=True)
for kb in (256,):
    A=A0[:,:,:kb]; T=T0[:,:,:kb]
    R,_=H.fit_maps(A)
    for kind in ('O','GL'):
        mdl = H.model_O(A,T,R) if kind=='O' else H.model_GL(A,T)
        An,Tn,Pn,gn = H.surrogate(Xf,Xt,g0,np.ascontiguousarray(P0[:,:,:kb]),mdl,seed=5)
        for k in (64,128,256):
            print(kind,'kb',kb,'-> k',k, cp(An,Tn,k), 'infl',mdl['infl'], flush=True)
print(time.time()-t0)
