import cv2
import numpy as np
import seam_carving as sc
import matplotlib.pylab as plt

def expr2segm_img(img) -> list:
    # zaj csökkentese
    img = cv2.medianBlur(img, 3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kontur kiemelese
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    
    img_size = 45
    
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for c in contours:
        area = cv2.contourArea(c)
        if True:
            x, y, w, h = cv2.boundingRect(c)
            segment = img[y:y + h, x:x + w]
            if True:
                if w <= img_size:
                    blank = np.ones((h, img_size), dtype=np.uint8)*255
                    blank[0:h, int(img_size/2-w/2):int(img_size/2+w/2)] = segment
                    w = img_size
                    segment = blank
                if h <= img_size:
                    blank = np.ones((img_size, w), dtype=np.uint8)*255
                    blank[int(img_size/2-h/2):int(img_size/2+h/2), 0:w] = segment
                    h = img_size
                    segment = blank
                if w > img_size or h > img_size:
                    segment = sc.resize(segment, (img_size, img_size))
                    w = h = img_size
            else:
                segment = cv2.resize(segment, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            rs = np.zeros((segment.shape[0], segment.shape[1], 3))
            rs[:,:,0] = segment
            rs[:,:,1] = segment
            rs[:,:,2] = segment
            segments.append((rs[:,:,0]/255.0, x))
    segments.pop(0)
    return segments