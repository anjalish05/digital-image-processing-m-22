import numpy as np
import cv2 as cv
import mediapipe as mp


def get_mask(img):
    results = selfie_segmentation.process(img)
    mask = results.segmentation_mask
    mask = np.where(mask > 0.0005 , 1, 0)
    return mask

def getBackground(image, mask):
    img = image.copy()
    img[mask > 0 ] = [0, 0, 0]
    return img

def change_baground(image, mask, background):
    img = image.copy()
    img[mask == 0 ] = background[mask == 0 ]
    return img

def imperfection_mask(img,mask):
    new_img = np.zeros((img.shape[0], img.shape[1]))
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)+mask
   
    new_img[gray<=0] = 255
    new_img[gray>0]=0
    return np.uint8(new_img)


def exchange_baground(input_rgb,output_rgb,example_rgb):
    # plot_rgb(input_rgb)
    # plot_rgb(example_rgb)
    
    mask_input = get_mask(input_rgb)
    mask_example = get_mask(example_rgb)
    
    # plot_grayscale(mask_input)
    #print(mask_input)
    # plot_grayscale(mask_example)
    
    bag = getBackground(example_rgb, mask_example)
    # plot_rgb(bag)
    
    new_img = change_baground(output_rgb, mask_input, bag)
    # plot_rgb(new_img)
    
    mask_imperect = imperfection_mask(new_img,mask_input)
    # plot_grayscale(mask_imperect)
    
    final_baground_change = cv.inpaint(new_img, mask_imperect, 3, cv.INPAINT_NS)
    
    return final_baground_change
