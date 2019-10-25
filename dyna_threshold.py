import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

import os
import argparse
import logging

import cv2
import numpy

from blur_detection import estimate_blur
from blur_detection import fix_image_size
from blur_detection import pretty_blur_map
from noise_calc import estimate_noise

global overlapping_array
overlapping_array = [1, 0.51, 0.52, 0.42, 0.37]
#overlapping_array = [1,0.]

class cameras(object):

    def __init__(self, cid, overlapping):
        self.cid = cid
        self.threshold = [0, 0, 0, 0, 0]  # array for threshold values
        self.overlapping = overlapping
        self.crowd_array = [0, 0, 0, 0, 0]
        self.noises = [0, 0, 0, 0, 0]
        self.blurriness = [0,0,0,0,0]

    def crowd_density(self, cr_de):
        self.crowd_array[cr_de[0]] = cr_de[1]

    def noise(self, noise_value):
        self.noises[noise_value[0]] = noise_value[1]

    def blur(self,blur_value):
        self.blurriness[blur_value[0]] = blur_value[1]

    def threshold_calc(self):
        weightage = [0.5, 0.0, 0.5, 0.0]        
        # weightage = [0.6, 0.0, 0.4, 0.0]
        count = 0
        for i in self.overlapping:
                #print(i)
            self.threshold[count] = weightage[0] * (1/i)
            count += 1
        count = 0
        for i in self.blurriness:
            self.threshold[count] += weightage[1] * i
            count += 1
        count = 0
        for i in self.crowd_array:
            self.threshold[count] += weightage[2] * (1/i)
            count += 1
        count = 0
        for i in self.noises:
            self.threshold[count] += weightage[3] * i
            count += 1

        [self.threshold] = normalize([self.threshold],norm='l2')
        return self.threshold

# print("Enter the input images name for the three cameras")
# s1,s2,s3,s4,s5=raw_input().split()


def Reputation_threshold(image1,image2,image3,image4,image5,crowd_density_array):
    
    input_image1 = cv2.imread(str(image1))
    input_image2 = cv2.imread(str(image2))
    input_image3 = cv2.imread(str(image3))
    input_image4 = cv2.imread(str(image4))
    input_image5 = cv2.imread(str(image5))

    input_image_1 = fix_image_size(input_image1)
    blur_map_1, score_1, blurry_1 = estimate_blur(input_image_1)    

    input_image_2 = fix_image_size(input_image2)
    blur_map_2, score_2, blurry_2 = estimate_blur(input_image_2)    

    input_image_3 = fix_image_size(input_image3)
    blur_map_3, score_3, blurry_3 = estimate_blur(input_image_3)    

    input_image_4 = fix_image_size(input_image4)
    blur_map_4, score_4, blurry_4 = estimate_blur(input_image_4)    

    input_image_5 = fix_image_size(input_image5)
    blur_map_5, score_5, blurry_5 = estimate_blur(input_image_5)    
    

    # print("Quality_score1: {0}, blurry1: {1}".format(score_1, blurry_1))
    # print("Quality_score2: {0}, blurry2: {1}".format(score_2, blurry_2))
    # print("Quality_score3: {0}, blurry3: {1}".format(score_3, blurry_3))
    # print("Quality_score4: {0}, blurry4: {1}".format(score_4, blurry_4))
    # print("Quality_score5: {0}, blurry5: {1}".format(score_5, blurry_5))    
    
    

    """    if args.display:
            cv2.imshow("input", input_image1)
            cv2.imshow("result", pretty_blur_map(blur_map))
            cv2.waitKey(0)"""   

    img_gray = cv2.cvtColor(input_image1, cv2.COLOR_BGR2GRAY)
    noise_value_img_1 = estimate_noise(img_gray)    

    img_gray = cv2.cvtColor(input_image2, cv2.COLOR_BGR2GRAY)
    noise_value_img_2 = estimate_noise(img_gray)    

    img_gray = cv2.cvtColor(input_image3, cv2.COLOR_BGR2GRAY)
    noise_value_img_3 = estimate_noise(img_gray)    

    img_gray = cv2.cvtColor(input_image4, cv2.COLOR_BGR2GRAY)
    noise_value_img_4 = estimate_noise(img_gray)    

    img_gray = cv2.cvtColor(input_image5, cv2.COLOR_BGR2GRAY)
    noise_value_img_5 = estimate_noise(img_gray)    
    

    # print("Noise of camera 1 is ", noise_value_img_1)
    # print("Noise of camera 2 is ", noise_value_img_2)
    # print("Noise of camera 3 is ", noise_value_img_3)
    # print("Noise of camera 4 is ", noise_value_img_4)
    # print("Noise of camera 5 is ", noise_value_img_5)   
    
    
    global overlapping_array

    view_7 = cameras(0, overlapping_array)
    view_1 = cameras(1, [0.3, 1, 0.3, 0.5, 0.4])
    view_5 = cameras(2, [0.5, 0.3, 1, 0.4, 0.4])
    view_6 = cameras(3, [0.4, 0.5, 0.4, 1, 0.4])
    view_8 = cameras(4, [0.2, 0.5, 0.3, 0.4, 1])    

    # crowd density will be calculated from a separate code.These are dummy values  

    #scalar = StandardScaler(with_mean=False)
    #crowd_density_array = [[10], [5], [7]]
    crowd_density_array = [crowd_density_array]
    #scalar.fit(crowd_density_array)
    #crowd_density_array = scalar.transform(crowd_density_array)
    #[crowd_density_array] = np.array(crowd_density_array).reshape((1, 3))
    [crowd_density_array] = normalize(crowd_density_array,norm='l2')    
    

    view_7.crowd_density([view_7.cid, crowd_density_array[view_7.cid]])
    view_7.crowd_density([view_1.cid, crowd_density_array[view_1.cid]])
    view_7.crowd_density([view_5.cid, crowd_density_array[view_5.cid]])
    view_7.crowd_density([view_6.cid, crowd_density_array[view_6.cid]])
    view_7.crowd_density([view_8.cid, crowd_density_array[view_8.cid]]) 
    

    view_1.crowd_density([view_1.cid, crowd_density_array[view_1.cid]])
    view_1.crowd_density([view_7.cid, crowd_density_array[view_7.cid]])
    view_1.crowd_density([view_5.cid, crowd_density_array[view_5.cid]])
    view_1.crowd_density([view_6.cid, crowd_density_array[view_6.cid]])
    view_1.crowd_density([view_8.cid, crowd_density_array[view_8.cid]]) 
    

    view_5.crowd_density([view_5.cid, crowd_density_array[view_5.cid]])
    view_5.crowd_density([view_7.cid, crowd_density_array[view_7.cid]])
    view_5.crowd_density([view_1.cid, crowd_density_array[view_1.cid]])
    view_5.crowd_density([view_6.cid, crowd_density_array[view_6.cid]])
    view_5.crowd_density([view_8.cid, crowd_density_array[view_8.cid]]) 
    
    

    #scalar = StandardScaler(with_mean=False)
    #noise_array = [[noise_value_img_1], [noise_value_img_2], [noise_value_img_3]]
    noise_array = [[noise_value_img_1, noise_value_img_2, noise_value_img_3, noise_value_img_4, noise_value_img_5]]
    #scalar.fit(noise_array)
    #noise_array = scalar.transform(noise_array)
    #[noise_array] = np.array(noise_array).reshape((1, 3))
    [noise_array] = normalize(noise_array,norm='l2')    
    

    view_7.noise([view_7.cid, noise_array[view_7.cid]])
    view_7.noise([view_1.cid, noise_array[view_1.cid]])
    view_7.noise([view_5.cid, noise_array[view_5.cid]])
    view_7.noise([view_6.cid, noise_array[view_6.cid]])
    view_7.noise([view_8.cid, noise_array[view_8.cid]]) 
    

    view_1.noise([view_7.cid, noise_array[view_7.cid]])
    view_1.noise([view_1.cid, noise_array[view_1.cid]])
    view_1.noise([view_5.cid, noise_array[view_5.cid]])
    view_1.noise([view_6.cid, noise_array[view_6.cid]])
    view_1.noise([view_8.cid, noise_array[view_8.cid]]) 
    

    view_5.noise([view_7.cid, noise_array[view_7.cid]])
    view_5.noise([view_1.cid, noise_array[view_1.cid]])
    view_5.noise([view_5.cid, noise_array[view_5.cid]])
    view_5.noise([view_6.cid, noise_array[view_6.cid]])
    view_5.noise([view_8.cid, noise_array[view_8.cid]]) 
    
    

    #scalar = StandardScaler(with_mean=False)
    #blur_array = [[score_1], [score_2], [score_3]]
    blur_array = [[score_1, score_2, score_3, score_4, score_5]]
    #scalar.fit(blur_array)
    #blur_array = scalar.transform(blur_array)
    #[blur_array] = np.array(blur_array).reshape((1, 3))
    [blur_array] = normalize(blur_array,norm='l2')  
    

    view_7.blur([view_7.cid, blur_array[view_7.cid]])
    view_7.blur([view_1.cid, blur_array[view_1.cid]])
    view_7.blur([view_5.cid, blur_array[view_5.cid]])
    view_7.blur([view_6.cid, blur_array[view_6.cid]])
    view_7.blur([view_8.cid, blur_array[view_8.cid]])   
    

    view_1.blur([view_1.cid, blur_array[view_1.cid]])
    view_1.blur([view_7.cid, blur_array[view_7.cid]])
    view_1.blur([view_5.cid, blur_array[view_5.cid]])
    view_1.blur([view_6.cid, blur_array[view_6.cid]])
    view_1.blur([view_8.cid, blur_array[view_8.cid]])   
    

    view_5.blur([view_5.cid, blur_array[view_5.cid]])
    view_5.blur([view_7.cid, blur_array[view_7.cid]])
    view_5.blur([view_1.cid, blur_array[view_1.cid]])
    view_5.blur([view_6.cid, blur_array[view_6.cid]])
    view_5.blur([view_8.cid, blur_array[view_8.cid]])   
    

    thresh_view_7 = list(view_7.threshold_calc())
    #thresh_view_1 = view_1.threshold_calc()
    #thresh_view_5 = view_5.threshold_calc()    

    # print("Threshold of cameras w.r.t to View7 is ", thresh_view_7)
    return thresh_view_7
    #print("Threshold of cameras w.r.t to Camera2 is ", thresh_view_1)
    #print("Threshold of cameras w.r.t to Camera3 is ", thresh_view_5)