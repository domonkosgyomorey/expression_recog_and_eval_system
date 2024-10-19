import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import imgsolver.imgsolver as isol
import matplotlib.pyplot as plt

def run():
    img_solver = isol.ImgSolver()
    img = cv2.imread('test_img/d.png')
    img_solver.eval(img)

if __name__ == "__main__":
    run()