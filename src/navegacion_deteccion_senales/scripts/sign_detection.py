import cv2
import numpy as np

'''
This program implements a traffic sign detection method. The method has been inspired by this following website: 
https://medium.com/@staytechrich/computer-vision-001-color-detection-with-opencv-58426c880449
'''

def _red_mask(image):
    '''Build binary red-color mask from a BGR image.'''
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def detection(image):
    '''
    Function to detect traffic speed signs. 
    Take in image as input
    If a traffic sign has been detected, return a zoom on it, otherwise return None.
    '''
    result = detection_with_bbox(image)
    if result is None:
        return None
    return result[0]


def detection_with_bbox(image):
    '''
    Detect the largest red traffic sign in image.
    Returns (cropped_sign, (x, y, w, h)) in original image coordinates,
    or None if no sign is found.
    '''
    if image is None:
        return None

    mask = _red_mask(image)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) <= 100:
        return None

    x, y, w, h = cv2.boundingRect(max_contour)
    stretch = 15
    h_img, w_img = image.shape[:2]
    y_start = max(0, y - stretch)
    y_end   = min(h_img, y + h + stretch)
    x_start = max(0, x - stretch)
    x_end   = min(w_img, x + w + stretch)

    cropped = image[y_start:y_end, x_start:x_end]
    bbox = (x_start, y_start, x_end - x_start, y_end - y_start)
    return cropped, bbox

def zoom(image, contour):
    '''
    Zoom on an image to only return the area within the contour.
    '''
    x, y, w, h = cv2.boundingRect(contour)
    stretch = 15
    
    # Getting the 4 corners of the rectangle
    h_img, w_img = image.shape[:2]
    y_start = max(0, y - stretch)
    y_end = min(h_img, y + h + stretch)
    x_start = max(0, x - stretch)
    x_end = min(w_img, x + w + stretch)
    
    # Zoom on the traffic sign
    zoomed_image = image[y_start:y_end, x_start:x_end]

    return zoomed_image