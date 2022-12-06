import numpy as np
import cv2 as cv

import dlib

def face_landmarks(img):
    img = np.copy(img)
    
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor('../data/inputs/shape_predictor_68_face_landmarks.dat')
    
    # Asks the detector to fing the bounding boxes of each face
    # The 1 in the second argument indicates that we should upsample the image 1 time
    # This will make everything bigger and allow us to detect more faces
    
    total_faces = face_detector(img, 1)
    
    total_landmarks = []
    
    for k, face in enumerate(total_faces):
        landmarks = landmark_detector(img, face)
        
        for i in range(0, 68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y

            total_landmarks.append([x, y])
            # cv.circle(img, (x, y), 4, (255, 0, 0), -1)
            
    total_landmarks = np.array(total_landmarks)
    # return img, total_landmarks
    return total_landmarks