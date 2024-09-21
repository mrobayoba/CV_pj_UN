import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# import rembg
import skimage.io
import skimage.data
import skimage.morphology as morph
from skimage.measure import label, regionprops
# import csv
# import math

######### HAY QUE REVISAR EN TODAS LAS SESIONES DE CLASE PARA VER QUE FUNCIONES IMPLEMENTAR ########


# It is useful to create functions to perform various task in just one call

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def plot_multi(img_array, colums = 1,colorMap= None, figSize=(15,15), Title = 'Images'):
    rows = len(img_array)//colums
    if(rows == 0):
        rows =1
    fig, plots = plt.subplots(rows,colums, figsize=(15, 15))

    for i in range(len(img_array)):
        # plots[i].set_title('Imagen original')
        if (colorMap != None):
            plots[i].imshow(img_array[i], cmap = colorMap)
            plots[i].axis('off')
        elif (colorMap == None):
            plots[i].imshow(img_array[i])
            plots[i].axis('off')
    fig.suptitle(Title, fontsize=24, y=0.66)


def showHistogram_f (image_in_gray):

    #to organize from lower to greater values use .ravel()
    #calculate the histogram
    hist, bins = np.histogram(image_in_gray.ravel(), 256, [0,255], density= True)

    # to show the histogram and the image
    fig, (im, h) = plt.subplots(1,2, figsize=(10,5))

    # Show the image in the first subplot
    im.imshow(image_in_gray, cmap="gray")
    im.set_title('Image')

    # Calculate the histogram of the image
    h.plot(hist)
    h.set_title('Histogram')
    h.set_xlim([0,256])

    # Display the plot
    plt.show()

def showHistogramColor_f (image_color): #RGB
    # Calculate histogram for each color channel
    hist_red, bins = np.histogram(image_color[:,:,0].ravel(),256,[0,255],density= True)
    hist_green,bins = np.histogram(image_color[:,:,1].ravel(),256,[0,255], density= True)
    hist_blue, bins = np.histogram(image_color[:,:,2].ravel(),256,[0,255], density= True)

    # Create subplots figure
    fig, (img, h) = plt.subplots(1,2,figsize=(10,5))

    img.imshow(image_color)
    img.set_title('Image')

    h.plot(hist_red, color= 'red')
    h.plot(hist_green,color= 'green')
    h.plot(hist_blue, color= 'blue')
    h.set_title('Histogram')
    h.set_xlim([0,256])

    plt.show()

def showColorSpace(image_RGB, colorSpace='RGB'):

    if colorSpace in ('COLOR', 'RGB'): #RGB
        img = image_RGB

        fig, plots = plt.subplots(1,3, figsize=(40,40))

        for i in range(3):
            plots[i].imshow(img[:,:,i])
        fig.suptitle(f"Canales RGB", fontsize=36, y=0.7)

    elif colorSpace == 'HSV': #HSV
        img = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2HSV)

        fig, plots = plt.subplots(1,3, figsize=(40,40))

        for i in range(3):
            plots[i].imshow(img[:,:,i])
        fig.suptitle(f"Canales HSV", fontsize=36, y=0.7)

    elif colorSpace == 'XYZ': #XYZ
        img = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2XYZ)

        fig, plots = plt.subplots(1,3, figsize=(40,40))

        for i in range(3):
            plots[i].imshow(img[:,:,i])
        fig.suptitle(f"Canales XYZ", fontsize=36, y=0.7)

    elif colorSpace == 'LAB': #LAB
        img = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2LAB)

        fig, plots = plt.subplots(1,3, figsize=(40,40))

        for i in range(3):
            plots[i].imshow(img[:,:,i])
        fig.suptitle(f"Canales LAB", fontsize=36, y=0.7)

    elif colorSpace == 'YUV': #YUV
        img = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2YUV)

        fig, plots = plt.subplots(1,3, figsize=(40,40))

        for i in range(3):
            plots[i].imshow(img[:,:,i])
        fig.suptitle(f"Canales YUV", fontsize=36, y=0.7)

    elif colorSpace == 'YCRCB' :
        img = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2YCrCb)

        fig, plots = plt.subplots(1,3, figsize=(40,40))

        for i in range(3):
            plots[i].imshow(img[:,:,i])
        fig.suptitle(f"Canales YCRCB", fontsize=36, y=0.7)

    elif colorSpace == 'LUV' :
        img = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2Luv)

        fig, plots = plt.subplots(1,3, figsize=(40,40))

        for i in range(3):
            plots[i].imshow(img[:,:,i])
        fig.suptitle(f"Canales LUV", fontsize=36, y=0.7)

    elif colorSpace == 'HLS' :
        img = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2HLS)

        fig, plots = plt.subplots(1,3, figsize=(40,40))

        for i in range(3):
            plots[i].imshow(img[:,:,i])
        fig.suptitle(f"Canales HLS", fontsize=36, y=0.7)

    elif colorSpace == 'HSV_FULL' :
        img = cv2.cvtColor(image_RGB,cv2.COLOR_RGB2HSV_FULL)

        fig, plots = plt.subplots(1,3, figsize=(40,40))

        for i in range(3):
            plots[i].imshow(img[:,:,i])
        fig.suptitle(f"Canales HSV_FULL", fontsize=36, y=0.7)
        
    else:
        # invalid input 
        raise Exception('INPUT ERROR: color space do not exist')



def im_read(image_path, mode='color'): #to read an image with any colormap
    mode = mode.upper()
    if mode in ('COLOR', 'RGB'): #RGB
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    elif mode in ('GRAYSCALE','GRAY'): #grayscale
        return cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    elif mode == 'HSV': #HSV
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2HSV)
    elif mode == 'XYZ': #XYZ
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2XYZ)
    elif mode == 'LAB': #LAB
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2LAB)
    elif mode == 'YUV': #YUV
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2YUV)
    elif mode == 'YCRCB' :
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2YCrCb)
    else:
        # invalid input 
        raise Exception('INPUT ERROR: color space do not exist')

#if you need to go back from RGB to BGR write color_space='BGR' !!!
def convert_image_f(image, color_space='GRAY') : # to switch from RBG to another color space
    color_space = color_space.upper()
    if color_space == 'RGB' :
        return image
    elif color_space == 'BGR':
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # special case
    elif color_space in ('GRAYSCALE','GRAY') :
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif color_space == 'HSV' :
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif color_space == 'XYZ' :
        return cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    elif color_space == 'LAB' :
        return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)    
    elif color_space == 'YUV' :
        return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif color_space == 'YCRCB' :
        return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'LUV' :
        return cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
    elif color_space == 'HLS' :
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif color_space == 'HSV_FULL' :
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
    else :
        raise Exception('INPUT ERROR: color space do not exist')
    
    
################ Transform Functions ################
def fft_transform_f(image_gray, plot= False):
    res = np.fft.fftshift(np.fft.fft2(image_gray))

    if plot == True:
        plt.imshow(np.log(np.abs(res)), vmin=0, vmax=20)
        plt.title('FFT de la imagen original')
        plt.colorbar();
    return res
    
def apply_function_f(img, f, args):
    #Crear una matriz de ceros del tamaño de la imagen de entrada
    res = np.zeros(img.shape, np.uint8)
    #Aplicar la transformación f sobre cada canal del espacio de color RGB
    res[:,:,0] = f(img[:,:,0], *args)
    res[:,:,1] = f(img[:,:,1], *args)
    res[:,:,2] = f(img[:,:,2], *args)
    
    return res

#Función para interpolar los puntos entre 0 y 255
def img_scale(img, value_range = [0,255]):
    return (value_range[1] - value_range[0])*(img - np.min(img))/(np.max(img)-np.min(img)) + value_range[0]

#Función para enviar los extremos superior e inferior a 0 y 255 respectivamente en la matriz
def img_trim(img, value_range = [0, 255]):
    res = img.copy()
    res[res > 255] = 255
    res[res < 0] = 0
    return res

def histogram_expansion_f(img):
    
    res = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    
    #Get the min and max values of the image
    m = float(np.min(img))
    M = float(np.max(img))
    #Normalize the image and save it with data type as uint8
    res = (img-m)*255.0/(M-m)
    res = res.astype(np.uint8)
    
    return res

def histogram_equalization_f(img): #cv2.equalizeHist
    
    res = cv2.equalizeHist(img) #uniform equalization
 
    return res

def brightness_correction(img_RGB):
    res = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2HSV)
    res[:,:,2] = histogram_equalization_f(res[:,:,2])
    res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)

    return res

def resize_and_interpolation_f(img, mode='bilinear', resize=(0,0)): 
    # interpolation, resize image to resize=(x,y) and interpolate as mode='bilinear' or mode='bicubic'
    mode = mode.upper()
    
    if mode == 'BILINEAR':
        res = cv2.resize(img,None,fx=resize[0], fy=resize[1], interpolation = cv2.INTER_LINEAR)
    elif mode == 'BICUBIC':
        res = cv2.resize(img,None,fx=resize[0], fy=resize[1], interpolation = cv2.INTER_CUBIC)
 
    return res



def unsharp_mask_f(img, kernel_size= 5): # Perfilado

    #Se le aplica a la imagen un filtro Gaussiano con un kernel de tamaño k_size x k_size,
    #y con un sigma de 2.
    gaussian = cv2.GaussianBlur(img, (kernel_size,kernel_size), 2)

    #Crear la máscara que corresponde a la resta entre la imagen original y el filtro gaussiano
    mascara_unsharp = cv2.subtract(img, gaussian)

    #Sumar la imagen de entrada con la máscara
    res = cv2.add(img, mascara_unsharp)
 
    return res

def clahe_f(img, a=5, x=8, y=8) :
    clahe = cv2.createCLAHE(clipLimit=a, tileGridSize=(x,y))
    img = apply_function_f(img, clahe.apply, [])
    
    return img

def get_contours(img_thresh, img, fill= False):
    
    aux = cv2.GaussianBlur(img_thresh,(9,9),2)
    
    if fill == True:
        thickness =-1
    elif fill == False:
        thickness = 3
    else:
        thickness = 3

    contours,_ =cv2.findContours(aux,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    area = {}

    for i in range(len(contours)):
        area[i] = cv2.contourArea(contours[i])

    masked = cv2.drawContours(img.copy(),contours,-1,(255,0,0),thickness)

    # print(area)

    return (masked, area, contours)

def gammaCorrection(img, a, gamma):
    # img->image a-> between 0 and 1 gamma-> power
    
    img_copy = img.copy().astype(np.float32)/255.0
    res_gamma = cv2.pow(img_copy,gamma)
    res = cv2.multiply(res_gamma, a)
    
    res[res<0] = 0
    res = res*255.0
    res[res>255] = 255
    res = res.astype(np.uint8)
    
    return res