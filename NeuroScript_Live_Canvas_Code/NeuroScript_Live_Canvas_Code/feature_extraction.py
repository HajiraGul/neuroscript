import cv2
import numpy as np

def preprocess_image(image_path, image_size=64):
    img=cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Could not read image: {image_path}')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),0)
    _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    binary=cv2.resize(binary,(image_size,image_size))
    return binary/255.0

def extract_features_from_array(img):
    binary=(img*255).astype(np.uint8) if img.max()<=1 else img.astype(np.uint8)
    _,binary=cv2.threshold(binary,127,255,cv2.THRESH_BINARY)
    ink=np.sum(binary>0)
    total=binary.shape[0]*binary.shape[1]
    ink_density=ink/total
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas=[cv2.contourArea(c) for c in contours if cv2.contourArea(c)>0]
    xproj=np.sum(binary>0,axis=0)
    yproj=np.sum(binary>0,axis=1)
    coords=np.column_stack(np.where(binary>0))
    if len(coords)>0:
        y0,x0=coords.min(axis=0)
        y1,x1=coords.max(axis=0)
        width=x1-x0+1
        height=y1-y0+1
        aspect=width/height if height else 0
    else:
        width=height=aspect=0
    dist=cv2.distanceTransform(binary,cv2.DIST_L2,5)
    stroke=float(np.mean(dist[dist>0])) if np.any(dist>0) else 0
    return np.array([ink_density,len(contours),np.mean(areas) if areas else 0,np.std(areas) if areas else 0,np.std(xproj),np.std(yproj),width,height,aspect,stroke],dtype=np.float32)

def extract_features_from_path(path, image_size=64):
    return extract_features_from_array(preprocess_image(path,image_size))