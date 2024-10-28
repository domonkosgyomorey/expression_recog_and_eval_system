import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import imgsolver.imgsolver as isol

def run():
    img_solver = isol.ImgSolver(4)
    img = cv2.imread('test_img/g.png')
    print(img_solver.eval(img))

if __name__ == "__main__":
    run()