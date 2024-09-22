import functions as f
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rembg
import skimage.io
import skimage.data
import skimage.morphology as morph
from skimage.measure import label, regionprops
import math


def analyzer(image_path):

    ##import image
    img = f.im_read(image_path) #RGB
    # f.show_image(img)
    print(img.shape)
    
    #preprocess
    preprocess = cv2.GaussianBlur(f.brightness_correction(img.copy(),level='low',sat=3,val=0.9),(7,7),5)


    ##remove background

    hsv_image = f.convert_image_f(preprocess.copy(),'hsv') 

    lower_green = np.array([15, 20, 30])    # Lower bound for green (H=250, S=40, V=40)
    upper_green = np.array([96, 254, 220])  # Upper bound for green (H=85, S=255, V=255)
    # Create a mask that extracts the green color
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Apply the mask to the original image to get the green parts
    green_extracted = cv2.bitwise_and(preprocess, preprocess, mask=green_mask)
    # f.show_image(green_extracted)


    img_green_mask = green_mask.copy()

    # use de ai background remover to increase the probability of just appear the plant portion of the image
    img_rembg_mask = rembg.remove(preprocess.copy()[:,:,0])
    img_rembg_mask = f.convert_image_f(img_rembg_mask,'gray')

    img_rembg = cv2.bitwise_or(img_green_mask,img_rembg_mask)

    #refine the mask
    masks_filled = cv2.morphologyEx(img_rembg.copy(),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=5)
    masks_filled = cv2.morphologyEx(masks_filled,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=10)

    main_mask, _, _ = f.get_contours(masks_filled.copy(), np.zeros_like(preprocess),fill=True)
    main_mask = f.convert_image_f(main_mask,'gray')
    
    main_image = f.convert_image_f(img.copy(), 'rgb')
    
    #apply the mask to the image
    main_image[main_mask == 0] = 0
    # print('current')
    # f.show_image(main_mask)



    # img_rembg = preprocess.copy()
    # #remove background from red channel of rgb image
    # img_rembg = rembg.remove(img_rembg[:,:,0])
    # img_rembg = f.convert_image_f(img_rembg,'gray')
    # #f.show_image(img_rembg)
    # main_image = np.zeros_like(img) # image to show masked at the end
    # # remove not useful image
    # if np.mean(img_rembg.ravel()) > 5:
    #     print('Valid picture, procesing...')
    #     main_image = img.copy()
    # else:
    #     return np.zeros_like(img)
    # 
    #Create a mask of the leaf to use later
    # main_mask = cv2.adaptiveThreshold(img_rembg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # mask,mask_area,mask_contours = f.get_contours(main_mask, np.zeros_like(main_mask),fill=True)
    #clean the mask by morph operations
    # mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=15)
    # mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)),iterations=10)
    
    #Apply the mask to the original image to use it
    # main_image[mask == 0] = 0

    #Filter image features
    img_filtered = f.clahe_f(main_image, a=2, x=5,y=5)
    # print('filtered image')
    # f.show_image(img_filtered)
    
    ##segmentation

    img_spots,img_yellow, mask = segmentation_f(img_filtered)

    ###characterization and interpretation
    # to count spots and calculate yellow area
    m_spots,spots, contours_spot = f.get_contours(img_spots,main_image)
    m_yellow,yellows, contours_yellow = f.get_contours(img_yellow, main_image)
    m_mask,mask_area, contours_mask = f.get_contours(mask,np.zeros_like(mask),fill=True)
    # f.show_image(m_yellow)
    # print('yellows',yellows)
    # f.show_image(m_mask)

    #to calculate yellow area
    leaf_area = 0.0
    for i in range(len(mask_area)):
        leaf_area += mask_area[i]

    sum_yellows = 0.0
    for i in range(len(yellows)):
        sum_yellows += yellows[i]
    #   print('Yellow area:',sum)

    #to count sports
    # print('spots_dic',spots)
    # print('contours_spot', contours_spot[1])
    aux_spots = [(spots[spot]/leaf_area)*100 for spot in spots] # spot_area/leaf_area in percentage
    # print('aux_spots',aux_spots)# print('max spot', np.max(aux_spots))
    spots_to_draw = []
    for i in range(len(spots)):
        if aux_spots[i] > 0.01:
            spots_to_draw.append(contours_spot[i])

    # print('spots_to_draw',spots_to_draw)


    # sum_spots_areas = 0.0
    # for i in range(len(yellows)):
    #     sum_yellows += yellows[i]
    # print('Yellow area:',sum_yellows)

    m_spots = cv2.drawContours(main_image,spots_to_draw,-1,(255,102,0),3)

    #add if area of the spots are lower than some % of leaf area, dont count as spot

    number_of_spots = len(spots_to_draw)
    # print('Leaf area: ',leaf_area)
    # print('sum of yellows', sum_yellows)
    print('Number of spots in the leaf: ', number_of_spots)
    print('Yellow percentage: %2.2f%%' %((sum_yellows/leaf_area)*100))
    if leaf_area == 0:
        leaf_yellows_area = None
    else:
        leaf_yellows_area = (sum_yellows/leaf_area)*100

    res = [image_path, number_of_spots, leaf_yellows_area]

    return res

def segmentation_f(img_not_bg):
    size_kernel = 5
    aux_img = None
    tomato_mold_leaf_HLS = f.convert_image_f(img_not_bg, 'HLS')#HLS or LAB
    tomato_mold_leaf_LAB = f.convert_image_f(img_not_bg, 'LAB')#HLS or LAB

    img_spots = cv2.GaussianBlur(tomato_mold_leaf_HLS,(size_kernel,size_kernel),0)[:,:,0]#Hue channel
    # f.show_image (img_not_bg)
    img_yellow = f.convert_image_f(img_not_bg, 'LAB')[:,:,0]
    
    _,mask = cv2.threshold(tomato_mold_leaf_LAB[:,:,0],20,255,cv2.THRESH_BINARY)
    
    # f.show_image(mask)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(size_kernel+4,size_kernel+4)),iterations=1)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS,(size_kernel+4,size_kernel+4)),iterations=3)
    
    aux_img = img_spots.copy() # red color in Hue are between -25(225) to 25 aprox
    aux_img[img_spots<20] = 255 # dark areas
    aux_img[mask==0] = 0
    # f.show_image(aux_img)
    img_spots[img_spots>150] = 255 # light areas
    img_spots[mask == 0] = 0
    img_yellow[img_yellow > 200] = 255
    img_yellow[mask == 0] = 0
    
    # f.show_image(mask)
    # f.show_image(img_spots)
    # f.show_image(img_yellow)

    
    _,output_aux_spots = cv2.threshold(aux_img,240,255,cv2.THRESH_BINARY)
    # f.show_image(output_aux_spots)
    _,output_spots = cv2.threshold(img_spots,240,255,cv2.THRESH_BINARY)
    # f.show_image(output_spots)
    output_spots = cv2.bitwise_or(output_spots,output_aux_spots)

    _,output_yellow = cv2.threshold(img_yellow,250,255,cv2.THRESH_BINARY) #Yellow areas
    # f.show_image(output_spots)
    # f.show_image(output_yellow)

    output_spots = cv2.bitwise_and(output_spots,mask)
    output_yellow = cv2.bitwise_and(output_yellow,mask)

      
    output_yellow = cv2.morphologyEx(output_yellow,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_kernel-2,size_kernel-2)),iterations=4)
    output_yellow = cv2.morphologyEx(output_yellow,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_kernel,size_kernel)),iterations=2)

    output_spots = cv2.morphologyEx(output_spots,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_kernel-2,size_kernel-2)),iterations=1)
    output_spots = cv2.morphologyEx(output_spots,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_kernel,size_kernel)),iterations=5)
    # f.show_image(output_spots)
    # f.show_image(output_yellow)

    return [output_spots,output_yellow, mask]