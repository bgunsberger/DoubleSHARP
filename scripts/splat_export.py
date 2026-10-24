"""Export SHARP tensors to the existing WebGL viewer's 32-byte format."""
import numpy as np
import torch

def export_splat(state, path):
    p=state['mean_vectors'].reshape(-1,3).numpy().copy()
    s=state['singular_values'].reshape(-1,3).numpy()
    q=state['quaternions'].reshape(-1,4).numpy()
    c=state['colors'].reshape(-1,3).numpy()
    a=state['opacities'].reshape(-1).numpy()
    c=np.where(c<=.0031308,12.92*c,1.055*np.maximum(c,0)**(1/2.4)-.055)
    dtype=np.dtype([('p','<f4',(3,)),('s','<f4',(3,)),('c','u1',(4,)),('q','u1',(4,))])
    records=np.zeros(len(p),dtype=dtype)
    records['p']=p*np.array([1,-1,-1]);records['s']=s
    records['c'][:,:3]=np.clip(c*255,0,255).astype('u1')
    records['c'][:,3]=np.clip(a*255,0,255).astype('u1')
    q=q[:,[1,0,3,2]]*np.array([-1,1,-1,1])
    q/=np.linalg.norm(q,axis=1,keepdims=True).clip(1e-12)
    records['q']=np.clip(q*128+128,0,255).astype('u1')
    assert records.dtype.itemsize==32
    assert np.isfinite(p).all() and np.isfinite(s).all() and (s>0).all()
    records.tofile(path)
    print(f'Exported {len(p):,} Gaussians: {path}',flush=True)

if __name__=='__main__':
    import sys
    export_splat(torch.load(sys.argv[1],weights_only=True),sys.argv[2])
