import cv2
import numpy as np
import skimage
import matplotlib.pylab as plt

def expr2segm_img(img) -> list:
    img = cv2.medianBlur(img, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = 255 - img
    img = skimage.morphology.skeletonize(img)
    img = img.astype(np.uint8) * 255
    img = skimage.morphology.dilation(img, skimage.morphology.square(2))

    img_size = 45
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []

    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        segment = 255 - img[y:y + h, x:x + w]
        aspect_ratio = w / h
        if aspect_ratio > 1:
            new_w = img_size
            new_h = int(img_size / aspect_ratio)
        else:
            new_h = img_size
            new_w = int(img_size * aspect_ratio)

        resized_segment = cv2.resize(segment, (new_w, new_h), interpolation=cv2.INTER_AREA)
        blank = np.ones((img_size, img_size), dtype=np.uint8) * 255
        y_offset = (img_size - new_h) // 2
        x_offset = (img_size - new_w) // 2
        blank[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_segment

        rs = np.zeros((img_size, img_size, 3))
        rs[:, :, 0] = blank
        rs[:, :, 1] = blank
        rs[:, :, 2] = blank
        rs = rs[:, :, 0] / 255.0
        segments.append((rs, x, y, w, h, area))

    return segments