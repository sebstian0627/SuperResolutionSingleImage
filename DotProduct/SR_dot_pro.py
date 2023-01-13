import cv2
import numpy as np
import faiss
from utils_dot_pro import *
import matplotlib.pyplot as plt

def Super_resolution_gray(img):

    #Creates the Image Pyramids
    img_pyr = image_pyramid(img, MID-1)

    #Highest level of pyramid which is known to us is MID(indexed MID-1)
    highest_level_filled = MID-1

    # To threshold the patches that can cause troubles due to presence of textures in the image, we calculate the half_shifted image.
    out_x,out_y = move_by_half(img)


    patches_db = [] # contains all the patches of lower resolution images
    qi = [] #contains patch centers' i coordinate
    qj = [] ##contains patch centers' i coordinate
    qlvl = [] #contains the level the patch belongs to
    interpolation_model = {} # this stores the interpolation model of the images, to be used


    for i in reversed(range(0,MID-1)):
        #extracts all the patches fro, the images
        patches,ii,jj = image_to_patches(img_pyr[i])

        patches_db.extend((patches))
        hh,ww = img_pyr[i].shape
        interpolation_model[i] = interp2d(np.arange(ww), np.arange(hh), img_pyr[i])
        qi.extend(ii)
        qj.extend(jj)
        qlvl.extend(np.ones(len(ii))*i)
    
    #extract patches from the input image
    input_patches_p,input_pi,input_pj =  image_to_patches(img_pyr[MID-1])
    interpolation_model[MID-1] = interp2d(np.arange(img.shape[1]), np.arange(img.shape[0]), img)
    
    #Next target to start with is MID+1 
    next_target_start = MID 

    input_patches_p = np.array(input_patches_p)
    patches_db = np.array(patches_db)

    #Find the K=9 Nearest patches for each input patch
    D, knn = K_nearest_Dot_product(patches_db, input_patches_p)
    

        
    for next_target in range(next_target_start,N_Cell):
        print('\t',next_target)

        # which of the patches are skipped
        skipped = 0 
        skipped_no_info = 0

        # finds the change in levels, used for scaling and moving
        delta = float(next_target-MID)
        ht,wt = img_pyr[next_target].shape

        #initialising variables for the reconstruction of image
        new_img = np.ones((ht,wt))*DEFAULT_BG_GREYVAL 
        weighted_dists = np.zeros((ht,wt))
        sum_weights = np.zeros((ht,wt))
        factor_src = float(ht)/float(img.shape[0])
        

        for p_idx in range(0,knn.shape[0]):
            # calculating thresholds to remove out the patches which are very far and are matching because of the presence of texture im the image
            tx = thresholding(img,out_x,(input_pi[p_idx][0],input_pj[p_idx][0]) )
            ty = thresholding(img,out_y,(input_pi[p_idx][0],input_pj[p_idx][0]) )
            t = (tx+ty)/2
            
            # contains all the nearest neighbors of p_idx patch
            pknns = knn[p_idx,:]


            taken = 0

            for k in range(K):

                nn = pknns[k]

                nn_parent_lvl = qlvl[nn]+delta
                if nn_parent_lvl > highest_level_filled:
                    skipped_no_info += 1
                    print('skipped, ok?')
                    continue
                #finds the image containing the parent patch
                imp = img_pyr[int(qlvl[nn]+delta)]
                #finds the image containing the child patch
                imq = img_pyr[int(qlvl[nn])]

                child_h,_ = imq.shape
                parent_h,_ = imp.shape

                if taken > 0 and D[p_idx,k] <= t:
                    skipped += 1
                    continue
                
                lr_patch = input_patches_p[p_idx,:]
                lr_example = patches_db[nn,:]
                
                factor_example = parent_h/child_h
                # finds the coordinates corresponding to qi and qi in child oatch image imq in imp(parent image)
                hr_example,b = get_Parent(imp,imq,qi[nn],qj[nn],factor_example)
                if b:
                    taken += 1
                    #copying the patches in high res image(reconstruction) from the high-res/low-res patch pair.
                    weighted_dists,sum_weights, new_img =  set_parent(interpolation_model, img,input_pi[p_idx],input_pj[p_idx],new_img,hr_example,int(qlvl[nn]+delta),factor_src,weighted_dists,sum_weights,lr_patch,lr_example)

                    
        #normalise image for numerical stability
        new_img = weighted_dists/(sum_weights)
        #converts nan to 0, in case present
        new_img = np.nan_to_num(new_img)  
        shape =new_img.shape
        #stores a interpolation model to be used 
        interpolation_model[next_target] = interp2d(np.arange(shape[1]), np.arange(shape[0]), new_img)
        img_pyr[next_target] = new_img
        highest_level_filled = next_target
    return new_img

def Super_resolution_color(bgr_img):
    YCrCb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    g_img = YCrCb_img[:,:,0]/255
    YCrCb_img_1 = cv2.resize(YCrCb_img,None,fx=2,fy=2)
    g_img_1 = Super_resolution_gray(g_img)
    plt.imshow(g_img_1,cmap='gray')
    YCrCb_img_1[:,:,0]=(g_img_1*255).astype(np.uint8)
    rgb=cv2.cvtColor(YCrCb_img_1, cv2.COLOR_YCrCb2BGR)
    # cv2.imwrite('SR_inp3_gauss_weakYCrCB.jpg', rgb)
    return rgb

img_path = "output/inp3_building.png"# path to the input file
output_path = "inp3.png" # path to the output file
bgr_img = cv2.imread(img_path)
bgr_img_new = Super_resolution_color(bgr_img)
cv2.imwrite(output_path, bgr_img_new)