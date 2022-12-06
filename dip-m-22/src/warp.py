import numpy as np
import cv2 as cv

def warp_style(style_img, input_img, style_lm, input_lm):
    h, w, c = style_img.shape
    trans_coord = np.meshgrid(range(h), range(w), indexing='ij')
    yy, xx = trans_coord[0].astype(np.float64), trans_coord[1].astype(np.float64)  # might need to switch

    xsum = xx * 0
    ysum = yy * 0.
    wsum = xx * 0.
    
    # dx = np.arange(len(style_lm) - 1)
    dx = np.arange(len(style_lm))
    dy = dx + 1
    
    idx1 = np.array([16, 21, 26, 30, 35, 47])
    dy[idx1] = idx1
    
    idx2 = np.array([41, 47, 59, 67])
    dy[idx2] = np.array([36, 42, 48, 60])
    
    # idx2 = np.array([41, 47, 59])
    # dy[idx2] = np.array([36, 42, 48])

    # face.con
    for i in dx:
        j = dy[i]
        
        if i == j:
            continue

        # Computes u, v
        p_x1, p_y1 = (style_lm[i, 0], style_lm[i, 1])
        q_x1, q_y1 = (style_lm[j, 0], style_lm[j, 1])
        
        qp_x1 = q_x1 - p_x1
        qp_y1 = q_y1 - p_y1
        
        qpnorm1 = (qp_x1 ** 2 + qp_y1 ** 2) ** 0.5

        u = ((xx - p_x1) * qp_x1 + (yy - p_y1) * qp_y1) / qpnorm1 ** 2
        v = ((xx - p_x1) * -qp_y1 + (yy - p_y1) * qp_x1) / qpnorm1

        # Computes x', y'
        p_x2, p_y2 = (input_lm[i, 0], input_lm[i, 1])
        q_x2, q_y2 = (input_lm[j, 0], input_lm[j, 1])
        
        qp_x2 = q_x2 - p_x2
        qp_y2 = q_y2 - p_y2
        
        qpnorm2 = (qp_x2 ** 2 + qp_y2 ** 2) ** 0.5

        d_x = p_x2 + u * (q_x2 - p_x2) + (v * -qp_y2) / qpnorm2  # X'(x)
        d_y = p_y2 + u * (q_y2 - p_y2) + (v * qp_x2) / qpnorm2  # X'(y)

        # Computes weights
        d1 = ((xx - q_x1) ** 2 + (yy - q_y1) ** 2) ** 0.5
        d2 = ((xx - p_x1) ** 2 + (yy - p_y1) ** 2) ** 0.5
        
        d = np.abs(v)
        
        d[u > 1] = d1[u > 1]
        d[u < 0] = d2[u < 0]
        
        W = (qpnorm1 ** 1 / (10 + d)) ** 1

        wsum += W
        xsum += W * d_x
        ysum += W * d_y

    x_m = xsum / wsum
    y_m = ysum / wsum
    
    vx = xx - x_m
    vy = yy - y_m
    
    vx[x_m < 1] = 0
    vx[x_m > w] = 0
    vy[y_m < 1] = 0
    vy[y_m > h] = 0

    vx = (vx + xx).astype(int)
    vy = (vy + yy).astype(int)
    
    vx[vx >= w] = w - 1
    vy[vy >= h] = h - 1

    warp = np.ones(style_img.shape)
    warp[yy.astype(int), xx.astype(int)] = style_img[vy, vx]
    
    # from skimage.io import imsave
    # imsave('output/transformed.jpg', warp / 255)

    return warp, xx.astype(int), yy.astype(int), vx, vy