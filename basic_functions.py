import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import rembg
import skimage.io
import skimage.data
import skimage.morphology as morph
from skimage.measure import label, regionprops
import csv

MODES = ['RGB', 'YUV', 'HSV', 'LAB', 'HLS', 'XYZ', 'YCRCB', 'CMY', 'YIQ']

def img_read(filename, color_space='RGB') :
    """
    Leer una imagen con un espacio de color especificado.

    Input:
        - filename (str): Ruta de la imagen
        - color_space (str): Espacio de color de la imagen (RGB, GRAY, YUV, HSV, LAB, HLS, XYZ, YCrCb, YIQ, CMY) (default: 'RGB')

    Output
        - img (numpy.ndarray): Matriz de la imagen obtenida en el espacio de color deseado
    """
    color_space = color_space.upper()
    if color_space == 'RGB' :
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    elif color_space == 'GRAY' :
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
    elif color_space == 'YUV' :
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2YUV)
    elif color_space == 'HSV' :
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2HSV)
    elif color_space == 'LAB' :
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2LAB)
    elif color_space == 'HLS' :
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2HLS)
    elif color_space == 'XYZ' :
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2XYZ)
    elif color_space == 'YCRCB' :
        return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2YCrCb)
    elif color_space == 'CMY' :
        img = img_read(filename)
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        new_img = np.zeros(img.shape, np.uint8)

        C = 255 - R
        M = 255 - G
        Y = 255 - B

        new_img[:, :, 0] = C
        new_img[:, :, 1] = M
        new_img[:, :, 2] = Y

        return new_img
    elif color_space == 'YIQ' :
        img = img_read(filename)
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        new_img = np.zeros(img.shape, np.uint8)

        Y = 0.299*R + 0.587*G + 0.114*B
        I = 0.596*R - 0.274*G - 0.322*B
        Q = 0.211*R - 0.523*G + 0.312*B

        new_img[:, :, 0] = Y
        new_img[:, :, 1] = I
        new_img[:, :, 2] = Q

        return new_img
    else :
        raise Exception('INPUT ERROR: Espacio de color incorrecto')

def convert_image(img, color_space='GRAY') :
    """
    Convertir una imagen en RGB al espacio de color especificado.

    Input:
        - img (numpy.ndarray): Matriz de la imagen en RGB
        - color_space (str): Espacio de color al que se quiera convertir la imagen (RGB, GRAY, YUV, HSV, LAB, HLS, XYZ, YCrCb, YIQ, CMY) (default: 'GRAY')

    Output
        - img (numpy.ndarray): Matriz de la imagen obtenida en el espacio de color deseado
    """
    color_space = color_space.upper()
    if color_space == 'RGB' :
        return img
    elif color_space == 'GRAY' :
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif color_space == 'YUV' :
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif color_space == 'HSV' :
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif color_space == 'LAB' :
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    elif color_space == 'HLS' :
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif color_space == 'XYZ' :
        return cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
    elif color_space == 'YCRCB' :
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif color_space == 'CMY' :
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        new_img = np.zeros(img.shape, np.uint8)

        C = 255 - R
        M = 255 - G
        Y = 255 - B

        new_img[:, :, 0] = C
        new_img[:, :, 1] = M
        new_img[:, :, 2] = Y

        return new_img
    elif color_space == 'YIQ' :
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        new_img = np.zeros(img.shape, np.uint8)

        Y = 0.299*R + 0.587*G + 0.114*B
        I = 0.596*R - 0.274*G - 0.322*B
        Q = 0.211*R - 0.523*G + 0.312*B

        new_img[:, :, 0] = Y
        new_img[:, :, 1] = I
        new_img[:, :, 2] = Q

        return new_img
    else :
        raise Exception('INPUT ERROR: Espacio de color incorrecto')

# TRANSFORMACIONES
def apply_f(img, f, args):
    #Crear una matriz de ceros del tamaño de la imagen de entrada
    res = np.zeros(img.shape, np.uint8)
    #Aplicar la transformación f sobre cada canal del espacio de color RGB
    res[:,:,0] = f(img[:,:,0], *args)
    res[:,:,1] = f(img[:,:,1], *args)
    res[:,:,2] = f(img[:,:,2], *args)
    
    return res

def cuadraticTransform(img, a, b ,c):
    img_copy = img.copy().astype(np.float32)/255.0
    res_a = cv2.pow(img_copy, 2)
    res_a = cv2.multiply(res_a, a)
    res_b = cv2.multiply(img_copy, b)
    res = cv2.add(res_a, res_b)
    res = cv2.add(res, c)
    
    res[res < 0] = 0
    res = res * 255
    res[res > 255] = 255
    res = res.astype(np.uint8)
    
    return res

def rootTransform(img, a, b): 
    img_copy = img.astype(np.float32)/255.0
    res_a = cv2.pow(img_copy, 0.5)
    res_a = cv2.multiply(res_a, a)
    res = cv2.add(res_a, b)

    res[res < 0] = 0
    res = res * 255
    res[res > 255] = 255
    res = res.astype(np.uint8)
    
    return res

def gammaCorrection(img, a, gamma):
    
    img_copy = img.copy().astype(np.float32)/255.0
    res_gamma = cv2.pow(img_copy,gamma)
    res = cv2.multiply(res_gamma, a)
    
    res[res<0] = 0
    res = res*255.0
    res[res>255] = 255
    res = res.astype(np.uint8)
    
    return res

def histogram_expansion(img):
    
    #Crear matriz de ceros del tamaño de la imagen y tipo de dato flotante
    res = np.zeros([img.shape[0], img.shape[1]], dtype=np.float32)
    
    #Extraer el mínimo y el máximo del conjunto de datos
    m = float(np.min(img))
    M = float(np.max(img))
    #Aplicar la función de expansión(normalización) y asegurar datos uint8
    res = (img-m)*255.0/(M-m)
    res = res.astype(np.uint8)
    
    return res

def clahe_f(img, a=5, x=8, y=8) :
    clahe = cv2.createCLAHE(clipLimit=a, tileGridSize=(x,y))
    img = apply_f(img, clahe.apply, [])
    
    return img