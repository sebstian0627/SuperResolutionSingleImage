import cv2
from PIL import Image
import numpy as np
import os
from glob import glob
from PIL import Image
import math


def nearest_neighbor_interpolation(img, h_t,w_t):
    w,h= img.size
    scale_x = w_t/w
    scale_y = h_t/h

    output_image = Image.new(img.mode, (w_t, h_t), 'white')

    for y in range(h_t):
        for x in range(w_t):
            x_n = int(np.round(-1+ x/scale_x))
            y_n = int(np.round(-1+ y/scale_y))

            pixel = img.getpixel((x_n, y_n))
            output_image.putpixel((x, y),  pixel)

    return output_image

def cubic_interpolation_kernel(s,a=-1/2):
    s = abs(s)
    
    if s>=0 and s<=1:
        return (a+2)*(s**3) - (a+3)*(s*s)+1
    elif s>1 and s<=2:
        return a*(s*s*s) - 5*a*(s*s) +8*a*s - 4*a
    return 0


def bicubic_interpolation(img, h_t,w_t):
    pad =2
    h,w,C = img.shape

    scale_x = w_t/(w)
    scale_y = h_t/(h)
    output_image = np.zeros((h_t,w_t,C))
    img = cv2.copyMakeBorder(img, pad,pad,pad,pad,cv2.BORDER_REFLECT)
    for c in range(C):
        for y in range(h_t):
            for x in range(w_t):
                x_n,y_n = x/scale_x + pad, y/scale_y + pad
                # print(x_n,y_n)
                temp_x = x_n - math.floor(x_n)
                temp_y = y_n - math.floor(y_n)
                # print(temp_x,temp_y)

                x1 = 1+temp_x
                x2 = temp_x
                x3 = temp_x -1
                x4 = temp_x -2
                
                y1 = 1+temp_y
                y2 = temp_y
                y3 = temp_y -1
                y4 = temp_y -2

                left_matrix = [[cubic_interpolation_kernel(x1), cubic_interpolation_kernel(x2),cubic_interpolation_kernel(x3),cubic_interpolation_kernel(x4)]]
                left_matrix = np.matrix(left_matrix)
                right_matrix = [[cubic_interpolation_kernel(y1)],[cubic_interpolation_kernel(y2)],[cubic_interpolation_kernel(y3)],[cubic_interpolation_kernel(y4)]]
                right_matrix = np.matrix(right_matrix)

                
                middle_matrix = np.matrix([[img[int(y_n-y1), int(x_n-x1),c],
                                            img[int(y_n-y2), int(x_n-x1),c],
                                            img[int(y_n-y3), int(x_n-x1),c],
                                            img[int(y_n-y4), int(x_n-x1),c]],
                                        [img[int(y_n-y1), int(x_n-x2),c],
                                            img[int(y_n-y2), int(x_n-x2),c],
                                            img[int(y_n-y3), int(x_n-x2),c],
                                            img[int(y_n-y4), int(x_n-x2),c]],
                                        [img[int(y_n-y1), int(x_n-x3),c],
                                            img[int(y_n-y2), int(x_n-x3),c],
                                            img[int(y_n-y3), int(x_n-x3),c],
                                            img[int(y_n-y4), int(x_n-x3),c]],
                                        [img[int(y_n-y1), int(x_n-x4),c],
                                            img[int(y_n-y2), int(x_n-x4),c],
                                            img[int(y_n-y3), int(x_n-x4),c],
                                            img[int(y_n-y4), int(x_n-x4),c]]])
                output_image[y,x,c]=np.dot(np.dot(left_matrix, middle_matrix), right_matrix)
                
    return output_image

img_path = ""# path to the input file
output_path = "" # path to the output file
img = cv2.imread(img_path)
# for Nearest neigbor
h,w = img.shape[:2]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
out_img = nearest_neighbor_interpolation(img, 2*h,2*w)
cv2.imwrite(output_path, out_img)
# for bicubic
img = cv2.imread(img_path)

h,w = img.shape[:2]
out_img = bicubic_interpolation(img, 2*h,2*w)
cv2.imwrite(output_path, out_img)
