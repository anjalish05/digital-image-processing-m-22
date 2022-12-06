import numpy as np
import cv2 as cv



## LAPLACIAN
def laplacian_stacks(img, n=6):
    img = img.copy()
    # img = (img - np.min()) / (np.max(img) - np.max(img))
    
    laplacian_stacks = [rescale(img)]
    
    sigma = 1
    for i in range(n+1):
        sigma <<= 1
        k = 5 * sigma + 1
        
        gauss_blur = cv.GaussianBlur(img, (k, k), sigma)
        laplacian_stacks.append(gauss_blur)
    
    for i in range(n+1):
        # laplacian_stacks[i] = rescale(cv.subtract(laplacian_stacks[i+1], laplacian_stacks[i]))
        # laplacian_stacks[i] = rescale(cv.subtract(laplacian_stacks[i], laplacian_stacks[i+1]))
        
        laplacian_stacks[i] = rescale(laplacian_stacks[i] - laplacian_stacks[i+1])
        # laplacian_stacks[i] = laplacian_stacks[i] - laplacian_stacks[i+1]
    
    laplacian_stacks = np.array(laplacian_stacks)
    
    # return laplacian_stacks[1:-1], residuals[1:]
    return laplacian_stacks[:-1]

def residual(img, n=6):
    img = img.copy()
    
    sigma = 2 ** n
    k = 5 * sigma + 1
    
    R = cv.GaussianBlur(img, (k, k), sigma)
    return R



## LOCAL ENERGY
def local_energy(laplacian_stacks):
    S = []
    sigma = 1
    
    for subband in laplacian_stacks:
        sigma <<= 1
        k = 5 * sigma + 1
        
        # S.append(rescale(cv.GaussianBlur(subband ** 2, (k, k), sigma)))
        S.append(cv.GaussianBlur(subband ** 2, (k, k), sigma))
        
    return np.array(S)



## WARP
def warp_stacks(energy_stacks, xx, yy, vx, vy):
    warp_energy = []
    for subband in energy_stacks:
        temp = np.ones(subband.shape)
        temp[yy, xx] = subband[vy, vx]
        
        warp_energy.append(temp)
    
    return np.array(warp_energy)

def warp_residual(residual, xx, yy, vx, vy):
    warp_residual = np.ones(residual.shape)
    warp_residual[yy, xx] = residual[vy, vx]
    
    return warp_residual



## GAIN
def gain(warp_style_energy, input_energy):
    e = (1e-2) ** 2.0
    G = np.sqrt(warp_style_energy/(input_energy + e))
    return G

def robust_gain(warp_style_energy, input_energy, theta_h=2.8, theta_l=0.9, beta=3):
    G = gain(warp_style_energy, input_energy)
    RG = []
    
    sigma = 1
    for subband in G:
        k = 2 * (beta * sigma) + 1
        
        subband[subband > theta_h] = theta_h
        subband[subband < theta_l] = theta_l
        
        r_subband = cv.GaussianBlur(subband, (k, k), beta * sigma)
        RG.append(r_subband)
        
        sigma <<= 1
        
    return np.array(RG)

def robust_transfer(input_laplacian_stacks, warp_style_energy, input_energy):
    theta_h=2.8
    theta_l=0.9
    beta=3
    
    RG = robust_gain(warp_style_energy, input_energy, theta_h = theta_h, theta_l = theta_l, beta = beta)
    output_laplacian_stacks = input_laplacian_stacks * RG
    
    return output_laplacian_stacks



## AGGREGATE
def aggregate_stacks(output_stacks, warp_style_residual):
    output = rescale(np.sum(output_stacks, axis=0) + warp_style_residual)
    return output