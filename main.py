import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import imgsolver.imgsolver as isol
import sympy
import matplotlib.pyplot as plt

def cvt_str2op(predicted_class:str) -> str:
    if predicted_class == "times":
        return "*"
    elif predicted_class == "div":
        return "/"
    return predicted_class

def run():
    img_solver = isol.ImgSolver()
    img = cv2.imread('test_img/b.png')
    predicted_classes = img_solver.eval(img)
    expression = "".join(map(lambda x: cvt_str2op(x), predicted_classes))
    print(expression)
    print("Simplyfied expression: ",sympy.simplify(expression))

if __name__ == "__main__":
    run()