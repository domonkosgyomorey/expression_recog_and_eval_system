import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QPushButton, QLabel, QFrame, QHBoxLayout)
from PySide6.QtCore import Qt, QPoint, QTimer, Signal
from PySide6.QtGui import QPainter, QPen, QImage
import cv2
import numpy as np
from imgsolver import imgsolver as isol
import threading
import queue

class DrawingWidget(QWidget):
    drawing_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage(800, 150, QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.last_point = QPoint()
        self.setFixedSize(800, 150)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = event.pos()
            self.draw_point(event.pos())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.black, 4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.drawing_finished.emit()

    def draw_point(self, pos):
        painter = QPainter(self.image)
        painter.setPen(QPen(Qt.black, 4, Qt.SolidLine, Qt.RoundCap))
        painter.drawPoint(pos)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()

    def get_image(self):
        width = self.image.width()
        height = self.image.height()
        img = self.image.convertToFormat(QImage.Format_RGB888)
        ptr = img.constBits()
        arr = np.frombuffer(ptr, np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.im_solver = isol.ImgSolver('../models', verbose=True)
            self.result_queue = queue.Queue()
            self.init_ui()
            
            self.result_timer = QTimer()
            self.result_timer.timeout.connect(self.check_result)
            self.result_timer.start(100)
        except Exception as e:
            print(f"Initialization error: {e}")
            raise

    def init_ui(self):
        self.setWindowTitle("Mathematical Expression Recognition")
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #f5f5f5; 
            }
            QLabel { 
                color: #333333; 
            }
            QPushButton { 
                background-color: #2196F3; 
                color: white; 
                border: none; 
                padding: 10px 25px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover { 
                background-color: #1976D2; 
            }
            QFrame#canvas_frame {
                background-color: white;
                border: 2px solid #2196F3;
                border-radius: 8px;
            }
            QFrame#result_frame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)

        title = QLabel("Handwritten Mathematical Expression Recognition")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #1976D2; margin-bottom: 10px;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        subtitle = QLabel("Draw a mathematical expression in the area below")
        subtitle.setStyleSheet("font-size: 16px; color: #666666; margin-bottom: 20px;")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        
        canvas_container = QWidget()
        canvas_layout = QHBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        
        canvas_frame = QFrame()
        canvas_frame.setObjectName("canvas_frame")
        canvas_frame.setFixedSize(840, 190)  # Nagyobb a padding miatt
        canvas_frame_layout = QVBoxLayout(canvas_frame)
        canvas_frame_layout.setContentsMargins(20, 20, 20, 20)
        
        self.drawing_widget = DrawingWidget(self)
        canvas_frame_layout.addWidget(self.drawing_widget, alignment=Qt.AlignCenter)
        
        canvas_layout.addWidget(canvas_frame, alignment=Qt.AlignCenter)
        main_layout.addWidget(canvas_container)
        
        self.drawing_widget.drawing_finished.connect(self.on_drawing_finished)
        
        result_container = QWidget()
        result_layout = QVBoxLayout(result_container)
        result_layout.setSpacing(10)
        
        result_title = QLabel("Recognized Expression:")
        result_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        result_title.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(result_title)
        
        result_frame = QFrame()
        result_frame.setObjectName("result_frame")
        result_frame_layout = QVBoxLayout(result_frame)
        
        self.result_label = QLabel("The recognized expression will appear here...")
        self.result_label.setStyleSheet("""
            font-size: 24px;
            color: #333;
            padding: 15px;
        """)
        self.result_label.setFixedSize(800, 70)
        self.result_label.setAlignment(Qt.AlignCenter)
        result_frame_layout.addWidget(self.result_label, alignment=Qt.AlignCenter)
        
        result_layout.addWidget(result_frame)
        main_layout.addWidget(result_container)
        
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        
        clear_button = QPushButton("Clear Canvas")
        clear_button.clicked.connect(self.clear_canvas)
        clear_button.setCursor(Qt.PointingHandCursor)
        clear_button.setFixedWidth(200)
        button_layout.addWidget(clear_button, alignment=Qt.AlignCenter)
        
        main_layout.addWidget(button_container)
        
        self.setFixedSize(900, 700)

    def clear_canvas(self):
        self.drawing_widget.clear()
        self.result_label.setText("The recognized expression will appear here...")

    def on_drawing_finished(self):
        try:
            image = self.drawing_widget.get_image()
            print("Output image size:", image.shape)
            self.result_label.setText("Processing...")
            recognition_thread = threading.Thread(
                target=self.process_image,
                args=(image,),
                daemon=True
            )
            recognition_thread.start()
        except Exception as e:
            print(f"Drawing processing error: {e}")
            self.result_label.setText(f"Error processing drawing: {str(e)}")

    def process_image(self, image):
        try:
            result = self.im_solver.eval(image)
            self.result_queue.put(result)
        except Exception as e:
            print(f"Recognition error: {e}")
            self.result_queue.put(f"Error: {str(e)}")

    def check_result(self):
        try:
            result = self.result_queue.get_nowait()
            if isinstance(result, tuple):
                string = result[0]
                if result[1] is not None:
                    string += " = " + str(result[1])
                self.result_label.setText(string)
            else:
                self.result_label.setText(str(result))
        except queue.Empty:
            pass

    def closeEvent(self, event):
        self.result_timer.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Application error: {e}")