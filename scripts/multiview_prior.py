"""Build a scaled depth input directly from joint two-view predictions."""
import cv2
import numpy as np

def prepare_prior(baseline, data):
    depth=data['depth'];conf=data['confidence'];K=data['intrinsics'];E=data['extrinsics']
    h,w=depth.shape[1:];yy,xx=np.mgrid[:h,:w].astype('float32')
    matrices=np.tile(np.eye(4),(2,1,1));matrices[:,:3,:]=E
    relative=matrices[1]@np.linalg.inv(matrices[0])
    rays=np.stack([(xx-K[0,0,2])/K[0,0,0],(yy-K[0,1,2])/K[0,1,1],np.ones_like(xx)],axis=-1)
    points=(rays*depth[0,:,:,None])@relative[:3,:3].T+relative[:3,3]
    pixels=points@K[1].T
    u=(pixels[:,:,0]/pixels[:,:,2]).astype('float32');v=(pixels[:,:,1]/pixels[:,:,2]).astype('float32')
    depth1=cv2.remap(depth[1],u,v,cv2.INTER_LINEAR)
    conf1=cv2.remap(conf[1],u,v,cv2.INTER_LINEAR)
    consistent=np.abs(depth1-points[:,:,2])/np.maximum(points[:,:,2],1e-5)<.08
    valid=(u>1)&(u<w-2)&(v>1)&(v<h-2)&(points[:,:,2]>0)&consistent&(conf[0]>1.4)&(conf1>1.4)
    # The baseline supplies scale only. No completed Gaussian is altered.
    z=cv2.resize(baseline['monodepth'][0,0].numpy(),(w,h))
    anchor=valid&(yy>h*.7)&(xx>w*.15)&(xx<w*.85)&(z>.2)&(z<15)
    if anchor.sum()<100:
        raise ValueError(f'Insufficient consistent pedestal depth support: {anchor.sum()} pixels')
    scale=float(np.median(z[anchor]/depth[0][anchor]))
    return depth[0]*scale,valid,scale,int(anchor.sum())
