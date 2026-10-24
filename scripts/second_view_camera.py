"""Export the jointly estimated second-photo camera for the WebGL viewer."""
import json
import numpy as np

def write_second_view_camera(root):
    data=np.load(root/'output/sharp-multiview/multiview-depth.npz')
    scale=json.loads((root/'output/sharp-multiview/conditioned-native-report.json').read_text())['two_view_to_sharp_scale']
    E=np.tile(np.eye(4),(2,1,1));E[:,:3,:]=data['extrinsics']
    relative=E[1]@np.linalg.inv(E[0]);scaled=relative.copy();scaled[:3,3]*=scale
    flip=np.diag([1,-1,-1,1]);view=flip@scaled@flip
    h,w=data['depth'].shape[1:]
    camera=dict(name='Second photograph',source='2026_10_L1101219.jpg',estimator='DA3 Giant joint two-view prediction; estimated camera, not calibrated ground truth',view_matrix_column_major=view.flatten(order='F').tolist(),intrinsics=data['intrinsics'][1].tolist(),image_size=[w,h],relative_world_to_camera_cv=relative.tolist(),translation_scale=scale)
    (root/'examples/tree/second-view-camera.json').write_text(json.dumps(camera,indent=2)+'\n')

def add_second_view_support(s):
    s=s.replace('const keys = {};', '''let fixedCamera=null;
let secondCamera=null;
const secondCameraPromise=(_p.get("camera")==="second" ? loadSecondCamera() : Promise.resolve());
async function loadSecondCamera(){
  if(!secondCamera){const response=await fetch('/examples/tree/second-view-camera.json');if(!response.ok)throw Error('Second-view camera unavailable');secondCamera=await response.json();}
  fixedCamera=secondCamera;sortDirty=true;
}
const keys = {};''')
    s=s.replace('yaw=e.data.yaw; pitch=e.data.pitch;', 'if(e.data.camera==="second"){loadSecondCamera();return;}\n  fixedCamera=null;\n  yaw=e.data.yaw; pitch=e.data.pitch;')
    s=s.replace('canvas.addEventListener("pointerdown", e => {','canvas.addEventListener("pointerdown", e => {fixedCamera=null;')
    s=s.replace('canvas.addEventListener("wheel", e => {','canvas.addEventListener("wheel", e => {fixedCamera=null;')
    s=s.replace('const view=lookAt(eye,target,[0,1,0]);','const view=fixedCamera ? new Float32Array(fixedCamera.view_matrix_column_major) : lookAt(eye,target,[0,1,0]);')
    s=s.replace('const proj=perspective(fovy, aspect, 0.05, 100000);','let proj=perspective(fovy, aspect, 0.05, 100000);')
    s=s.replace('const fy=canvas.height/(2*Math.tan(fovy/2)), fx=fy;', '''let fy=canvas.height/(2*Math.tan(fovy/2)), fx=fy;
  if(fixedCamera){
    const [w,h]=fixedCamera.image_size,K=fixedCamera.intrinsics;
    const fit=Math.min(canvas.width/w,canvas.height/h);
    fx=K[0][0]*fit;fy=K[1][1]*fit;
    const cx=(canvas.width-w*fit)/2+K[0][2]*fit,cy=(canvas.height-h*fit)/2+K[1][2]*fit;
    const n=.05,f=100000;
    proj=new Float32Array([2*fx/canvas.width,0,0,0, 0,2*fy/canvas.height,0,0, 1-2*cx/canvas.width,2*cy/canvas.height-1,(f+n)/(n-f),-1, 0,0,2*f*n/(n-f),0]);
  }''')
    return s
