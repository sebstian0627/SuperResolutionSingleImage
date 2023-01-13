import cv2
import numpy as np
import os
from glob import glob
from scipy.interpolate import interp2d
import faiss

#variables

#variables
ALPHA = 2**(1/3) #scaling in each level of pyramid
EPS = 1e-15 #for safety
interMethod = cv2.INTER_AREA #interpolation method
DEFAULT_BG_GREYVAL = 0.0
N_Cell= round(2*np.log(2.0)/np.log(ALPHA) +1) # number of levels in pyramid
MID = int(np.ceil(float(N_Cell)/2)) # position where the input image is to be filled
PATCH_SIZE = 25 # 5x5 patch size
STEP = 2 # steps from patch center to patch boundary

K=9 # 9 nearest neighbors to be calculated.

def image_pyramid(img, image_level):
    img_pyramid = []
    
    for l in range(MID-1):
        img_pyramid.append(cv2.resize(img, None, fx = (ALPHA**(float(l)-float(image_level))), fy = (ALPHA**(float(l)-float(image_level))), interpolation=interMethod))
   
    img_pyramid.append(img)
   
    for l in range(MID, N_Cell):
        img_pyramid.append(np.zeros_like(cv2.resize(img, None, fx = (ALPHA**(float(l)-float(image_level))), fy = (ALPHA**(float(l)-float(image_level))))))
    
    return img_pyramid

def K_nearest_cos_product(X,Y):
 #X is the train Data and Y is the query data for which we need to find K similar patches
    # as mentioned in the report, this function transforms X and Y into X' and Y' respectively such that cosine similarity of p and q is same as inner product of p' and q'

  X_n= np.linalg.norm(X, axis=1)
  X = X/(X_n[:,np.newaxis])
  X1 = X.astype(np.float32)


  Y_n= np.linalg.norm(Y, axis=1)
  Y = Y/(Y_n[:,np.newaxis])
  Y1 = Y.astype(np.float32)
  index = faiss.IndexFlatIP(PATCH_SIZE)
  index.add(X1)

  D, I = index.search(Y1, K)

  return D,I



#translates image, by 1/2 pixel in both direction

def move_by_half(img):
    h,w = img.shape[:2]

    img_padded_y = np.zeros((h+1,w))
    img_padded_y[1:,:]= img
    img_padded_y[0,:]= img[1,:]
    X = np.arange(0, h+1)
    Y = np.arange(0, w)
    # X,Y  = np.meshgrid(range(0,w), range(0,h+1))
    Y_prime=Y+0.5
    out_y = interp2d(Y,X,img_padded_y, kind='cubic')
    x = np.arange(w)
    y = np.arange(h)+0.5
    out_y_ev = out_y(x,y)

    img_padded_x = np.zeros((h,w+1))
    img_padded_x[:,1:]= img
    img_padded_x[:,0]= img[:,1]
    X = np.arange(0, h)
    Y = np.arange(0, w+1)
    # X,Y = np.meshgrid(range(0,w+1), range(0,h))
    X_prime=X+0.5

    out_x = interp2d(Y,X,img_padded_x, kind='cubic')
    x = np.arange(w)+0.5
    y = np.arange(h)
    out_x_ev = out_x(x,y)


    return out_x_ev, out_y_ev
    
# find all the valid patches from the images, simply iterates over all possible patch centers.

def image_to_patches(img):
    h,w=img.shape[:2]
    num_patches = h*w
    patches= np.zeros((num_patches,PATCH_SIZE))
    pi = np.zeros((num_patches,1))
    pj = np.zeros((num_patches,1))

    pindx = 0

    for i in range(STEP, h-STEP):
        for j in range(STEP, w-STEP):
            
            p = img[i-STEP:i+STEP+1, j-STEP:j+STEP+1]
            patches[pindx]= np.reshape(p.T, (1,PATCH_SIZE))
            pi[pindx]= i
            pj[pindx]= j  
            pindx +=1
    
    patches = patches[0:pindx,:]
    pi = pi[0:pindx]
    pj = pj[0:pindx]

    return patches, pi,pj
# check if a coordinate is a valiud patch center
def check(h,w,patch_center):

    if( patch_center[0] -STEP >=0 and patch_center[0]+STEP<h and patch_center[1]-STEP>=0 and patch_center[1]+STEP<w):

        return True
    return False
#given a coordinate, returns a patch from the image if its a valid patch center

def coordinates2Patch(img, patch_center):
    h,w = img.shape
    p=[]
    if not check(h,w, patch_center):
        return None, False
    # print(patch_center[0]-STEP)
    p = img[int(patch_center[0]-STEP):int(patch_center[0]+STEP+1), int(patch_center[1]-STEP):int(patch_center[1]+STEP+1)]
    p = np.reshape(p.T, (1,PATCH_SIZE))
    
    return p, True
# used to threshold,

def thresholding(img,half_translated, patch_center, type="SSD"):
    p1,b1 = coordinates2Patch(img, patch_center)
    p2,b2 = coordinates2Patch(half_translated, patch_center)
    if( not (b1 and b2)):
        print("some of them is none")
        return 0
    t = distance(p1,p2)
    return t
# returns as 1- cosing simi of p1 and p2
def distance(p1,p2):
    p1 = p1.reshape(PATCH_SIZE,1)
    p1 = p1/np.linalg.norm(p1)
    p2 = p2.reshape(PATCH_SIZE,1)
    p2 = p2/np.linalg.norm(p2)
    dist = np.sum(p1*p2)
    return 1-dist
# uplift the coordinates from src level to dst level, adds a small correction factor

def move_level(src_lvl, src_x, src_y, dst_lvl):
    h_s,w_s = src_lvl.shape
    h_d,w_d = dst_lvl.shape
    scale_x = w_d/w_s
    scale_y = h_d/h_s

    dstx = scale_x*src_x- 0.5*scale_x*(1-1/scale_x)
    dsty = scale_y*src_y- 0.5*scale_y*(1-1/scale_y)
    return dstx, dsty
#returns the image of the parent patch and the patch centers

def get_Parent(p_img, q_img, qi, qj, scale_factor):
    pj,pi = move_level(q_img, qj,qi,p_img )

    pi2 = qi*scale_factor
    pj2 = qj*scale_factor

    h,w = p_img.shape
    # print(h,w,pi,pj,check(h,w,(pi,pj)),'this is get parent')
    if(check(h,w,(pi,pj))):
        parent_patch = {'image':p_img, 'pi': pi, 'pj':pj}
        b = True
    else:
        parent_patch = None
        b = False
    return parent_patch,b

# this is where the interpolation model and all the patches are used to construct the image at the higher level, if the given coordinate is a valid patch center, it finds the window in the higher level image and updates it using the high resolution image.

def set_parent(interpolation_model, current_img, current_pi, current_pj, new_image, high_res, high_level, scale_factor, weighted_dists, sum_weights, low_res_patch, low_res):

    pj,pi = move_level(current_img, current_pj, current_pi, new_image)

    pi2 = current_pi*scale_factor    
    pj2 = current_pj*scale_factor 
    h,w = new_image.shape
    
    if(check(h,w,(pi,pj))):
        # print(pi,pj,'this is pi')
        left = int(np.ceil(pj-STEP+EPS))
        right= int(np.floor(pj+STEP+EPS))
        top =  int(np.ceil(pi-STEP+EPS))
        bottom =  int(np.floor(pi+STEP+EPS))
        X_q_range =np.arange(left, right+1)
        Y_q_range= np.arange(top, bottom+1)
        dist_get_x = X_q_range - pj
        coord_set_x = high_res['pj'] +dist_get_x
        dist_get_y = Y_q_range - pi
        coord_set_y = high_res['pi'] +dist_get_y
        # print(dist_get_x, coord_set_x,dist_get_y, coord_set_y)
        # print(high_res['image'].shape, coord_set_x.shape)
        # X,Y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        Vq =  interpolation_model[high_level]
        # interp2d(np.arange(shape[1]), np.arange(shape[0]),high_res['image'],kind='cubic')
        Vq = Vq(coord_set_x, coord_set_y)
        # Vq = np.nan_to_num(Vq)
        # print(Vq.shape)
        Vq[Vq<0]=0
        Vq[Vq>1]=1
        # print(Vq.shape)
        weights = distance(low_res_patch, low_res)
        # print(weights)
        # print(Vq.shape, weights.shape, weights)
        sum_weights[top:bottom+1, left:right+1] = sum_weights[top:bottom+1, left:right+1]+weights
        weighted_dists[top:bottom+1, left:right+1]=weighted_dists[top:bottom+1, left:right+1]+(Vq*weights)
        
    return weighted_dists, sum_weights, new_image
