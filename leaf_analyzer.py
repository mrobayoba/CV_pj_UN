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
    preprocess = cv2.GaussianBlur(cv2.multiply(img.copy(),1.2),(9,9),5)


    ##remove background
    img_rembg = preprocess.copy()
    #remove background from red channel of rgb image
    img_rembg = rembg.remove(img_rembg[:,:,0])
    img_rembg = f.convert_image_f(img_rembg,'gray')
    #f.show_image(img_rembg)
    main_image = np.zeros_like(img) # image to show masked at the end
    # remove not useful image
    if np.mean(img_rembg.ravel()) > 5:
        print('Valid picture, procesing...')
        main_image = img.copy()
    else:
        return np.zeros_like(img)
    
    #Create a mask of the leaf to use later
    main_mask = cv2.adaptiveThreshold(img_rembg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    mask,mask_area,mask_contours = f.get_contours(main_mask, np.zeros_like(main_mask),fill=True)
    #clean the mask by morph operations
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=15)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)),iterations=10)
    
    #Apply the mask to the original image to use it
    main_image[mask == 0] = 0
    
    ##segmentation

    img_spots,img_yellow, mask = segmentation_f(main_image)

    ###characterization and interpretation
    # to count spots and calculate yellow area
    m_spots,spots, contours_spot = f.get_contours(img_spots,main_image)
    m_yellow,yellows, contours_yellow = f.get_contours(img_yellow, main_image)
    m_mask,mask_area, contours_mask = f.get_contours(mask,np.zeros_like(mask))

    #to count sports
    number_of_spots = len(spots)
    print('Number of spots in the leaf: ', number_of_spots)

    #to calculate yellow area
    leaf_area = mask_area[0]
    sum = 0.0
    for i in range(len(yellows)):
        sum += yellows[i]
    # print('Yellow area:',sum)
    yellow_percentage = (sum/leaf_area)*100
    if yellow_percentage >100:
        print('Yellow percentage out of range')
    else:
        print('Yellow percentage: %2.2f%%' %(yellow_percentage))

    # fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 6))
    # fig.suptitle(f"Illness progression", fontsize=15, y=0.9)
    # ax1.imshow(m_spots)
    # ax2.imshow(m_yellow)

    # ax1.set_title('Mold spots')
    # ax2.set_title('Mold yellow area')

    res =[image_path, number_of_spots, yellow_percentage]

    return res
    
def segmentation_f(img_not_bg):
    size_kernel = 5
    tomato_mold_leaf = f.convert_image_f(f.clahe_f(img_not_bg), 'HLS')
    img_spots = cv2.GaussianBlur(f.convert_image_f(f.clahe_f(img_not_bg),'HLS'),(size_kernel,size_kernel),0)[:,:,0]
    img_spots[img_spots>145] = 255

    img_yellow = f.convert_image_f(f.clahe_f(img_not_bg), 'HLS')[:,:,1]
    img_yellow[img_yellow > 230] = 255
    # img_yellow = cv2.add(img_yellow,cv2.multiply(0.3,tomato_mold_leaf[:,:,0]))
    # img_yellow = cv2.GaussianBlur(img_yellow,(size_kernel,size_kernel),10)
    
    _,mask = cv2.threshold(tomato_mold_leaf[:,:,2],30,255,cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS,(size_kernel,size_kernel)),iterations=6)
    # output_spots = cv2.morphologyEx(output_spots,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(size_kernel,size_kernel)),iterations=6)
    # f.show_image(mask)

    # img_spots = cv2.bitwise_and(img_spots,mask)
    # f.show_image(img_spots)
    
    # f.show_image(img_yellow)

    _,output_yellow = cv2.threshold(img_yellow,250,255,cv2.THRESH_BINARY) #Yellow areas

    _,output_spots = cv2.threshold(img_spots,250,255,cv2.THRESH_BINARY)
    # output_spots = cv2.adaptiveThreshold(img_spots,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    output_spots = cv2.bitwise_and(output_spots,mask)
    output_yellow = cv2.bitwise_and(output_yellow,mask)

    # f.show_image(output_spots)
    # f.show_image(output_yellow)


    output_yellow = cv2.morphologyEx(output_yellow,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS,(size_kernel,size_kernel)),iterations=5)
    output_yellow = cv2.morphologyEx(output_yellow,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(size_kernel,size_kernel)),iterations=1)

    output_spots = cv2.morphologyEx(output_spots,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_CROSS,(size_kernel,size_kernel)),iterations=6)
    output_spots = cv2.morphologyEx(output_spots,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_CROSS,(size_kernel,size_kernel)),iterations=1)
    # f.show_image(output_spots)
    # f.show_image(output_yellow)

    return [output_spots,output_yellow, mask]