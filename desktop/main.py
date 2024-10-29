import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageGrab
from imgsolver import imgsolver as isol
import matplotlib.pyplot as plt
import threading
import queue
import os

class DrawingApp:
    def __init__(self, root):
        self.im_solver = isol.ImgSolver('../models')
        self.root = root
        self.root.title("Mathematical Expression Recognition")
        self.bg_color = "#f0f0f0"
        self.accent_color = "#2196F3"
        self.canvas_color = "#ffffff"
        self.root.configure(bg=self.bg_color)
        self.root.geometry("1200x600")
        self.result_queue = queue.Queue()
        
        title_frame = tk.Frame(root, bg=self.bg_color)
        title_frame.pack(pady=20)
        
        title_label = tk.Label(
            title_frame,
            text="Handwritten Mathematical Expression Recognition",
            font=('Helvetica', 24, 'bold'),
            bg=self.bg_color,
            fg="#333333"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Draw a mathematical expression in the area below",
            font=('Helvetica', 12),
            bg=self.bg_color,
            fg="#666666"
        )
        subtitle_label.pack(pady=5)
        
        self.frame = tk.Frame(
            root,
            bg=self.accent_color,
            padx=2,
            pady=2
        )
        self.frame.pack(pady=10)
        
        canvas_width = 800
        canvas_height = 150
        
        self.canvas = tk.Canvas(
            self.frame,
            width=canvas_width,
            height=canvas_height,
            bg=self.canvas_color,
            highlightthickness=0
        )
        self.canvas.pack()
        
        result_frame = tk.Frame(root, bg=self.bg_color, pady=20)
        result_frame.pack(fill='x', padx=50)
        
        result_title = tk.Label(
            result_frame,
            text="Recognized Expression:",
            font=('Helvetica', 14, 'bold'),
            bg=self.bg_color,
            fg="#333333"
        )
        result_title.pack(pady=(0, 10))
        
        self.result_label = tk.Label(
            result_frame,
            text="The recognized expression will appear here...",
            font=('Helvetica', 24),
            bg='white',
            fg="#333333",
            wraplength=canvas_width,
            height=2,
            relief='flat',
            border=1
        )
        self.result_label.pack(fill='x')
        
        clear_button = tk.Button(
            root,
            text="Clear",
            font=('Helvetica', 12),
            command=self.clear_canvas,
            bg=self.accent_color,
            fg='white',
            relief='flat',
            padx=20,
            pady=5,
            cursor='hand2'
        )
        clear_button.pack(pady=20)
        
        self.image = Image.new('L', (canvas_width, canvas_height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        self.last_x = None
        self.last_y = None
        
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        self.check_result_thread()

    def check_result_thread(self):
        try:
            result = self.result_queue.get_nowait()
            string = result[0]
            if result[1] is not None:
                string += " = "+str(result[1])
            self.result_label.config(text=string, fg="#333333")
        except queue.Empty:
            pass
        self.root.after(100, self.check_result_thread)

    def process_image(self, image):
        try:
            result = self.im_solver.eval(image)
            self.result_queue.put(result)
        except Exception as e:
            print(e)
            self.result_queue.put(f"Error: {str(e)}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (self.canvas.winfo_width(), self.canvas.winfo_height()), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="The result will appear here...")

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.draw_point(event.x, event.y)

    def draw_point(self, x, y):
        r = 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def draw_line(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y,
                event.x, event.y,
                width=4,
                fill='black',
                capstyle=tk.ROUND,
                smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill='black',
                width=4
            )
            self.draw_point(event.x, event.y)
        
        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None
        image = self.get_canvas_image()
        
        recognition_thread = threading.Thread(
            target=self.process_image,
            args=(image,),
            daemon=True
        )
        recognition_thread.start()
        
        self.result_label.config(text="Processing...", fg="#666666")

    def get_canvas_image(self):
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        image = ImageGrab.grab(bbox=(x, y, x+width, y+height))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()