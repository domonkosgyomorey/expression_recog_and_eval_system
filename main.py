import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import imgsolver.imgsolver as isol
import sympy
import matplotlib.pyplot as plt
from expression_parser.expression_char import parse_expression

def cvt_str2op(predicted_class:str) -> str:
    if predicted_class == "times":
        return "*"
    elif predicted_class == "div":
        return "/"
    return predicted_class

def run():
    
    img_solver = isol.ImgSolver(4)
    img = cv2.imread('test_img/g.png')
    expression = img_solver.eval(img, cvt_str2op)
    #expression = map(lambda x: ExprChr(*x), expression)
    #expression = parse_expression(expression)
    #expression = "".join(map(lambda x: cvt_str2op(x), predicted_classes))
    expression = "".join(list(map(lambda x: x[0], expression)))
    print(expression)
    #print("Simplyfied expression: ",sympy.simplify(expression))

if __name__ == "__main__":
    run()