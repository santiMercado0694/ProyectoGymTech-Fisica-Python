import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
import prueba

class VideoPlayerApp:
    def __init__(self, master, menu_window):
        self.master = master
        self.menu_window = menu_window
        self.master.title("Reproductor de Video")
        self.video_path = None
        self.video_cap = None
        self.is_playing = False
        self.frame_number = 0
        self.frames = 30
        self.create_widgets()

    def create_widgets(self):
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack(padx=10, pady=10)
        self.btn_open = tk.Button(self.video_frame, text="Seleccionar Video", command=self.open_video)
        self.btn_open.grid(column=0, row=0)
        self.btn_go_back = tk.Button(self.video_frame, text="Volver al Menú", command=self.back_to_menu)
        self.btn_go_back.grid(column=0, row=1)

        self.control_frame = tk.Frame(self.video_frame)
        self.control_frame.grid(column=1, row=0)
        self.btn_play = tk.Button(self.control_frame, text="\u23F5", command=self.play_video, state=tk.DISABLED)
        self.btn_play.pack(side=tk.LEFT)
        self.btn_stop = tk.Button(self.control_frame, text="\u23F8", command=self.stop_video, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT)
        self.btn_restart = tk.Button(self.control_frame, text="\u23F9", command=self.restart_video, state=tk.DISABLED)
        self.btn_restart.pack(side=tk.LEFT)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.grid(column=1, row=1)

        self.image_frame = tk.Frame(self.master)
        self.image_frame.pack(padx=10, pady=10)
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()
        self.load_image("loadImage.png")
        self.show_image()

    def back_to_menu(self):
        self.master.destroy()
        self.menu_window.deiconify()

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4;*.avi;*.mkv")])
        if self.video_path:
            prueba.video_ready_callback = self.video_ready_callback
            prueba.track_pose(self.video_path)
            if hasattr(self, 'slider') and self.slider is not None:
                self.slider.destroy()
                self.frame_number = 0
            self.video_cap = cv2.VideoCapture('resultados\\video\\tracked_video.mp4')
            self.frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.create_slider()
            self.load_and_show_image("resultados\\graficos\\posicion_x_muneca.png")
            self.create_dropdown()

    def create_slider(self):
        self.slider_frame = tk.Frame(self.master)
        self.slider_frame.pack(pady=10)
        slider_length = 466
        self.slider = tk.Scale(self.slider_frame, from_=0, to=self.frames - 1, orient=tk.HORIZONTAL, command=self.on_slider_changed, length=slider_length)
        self.slider.pack(fill=tk.X)
        self.slider_frame.place(in_=self.image_frame, relx=0.5, rely=1.0, anchor=tk.CENTER)

    def create_dropdown(self):
        self.options = ["posicion munieca x", "posicion munieca y", "velocidad munieca y", "velocidad munieca x"]
        self.selected_option = tk.StringVar(value=self.options[0])
        self.dropdown = ttk.Combobox(self.video_frame, textvariable=self.selected_option, values=self.options)
        self.dropdown.grid(column=2, row=0, padx=10)
        self.dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_changed)

    def on_dropdown_changed(self, event):
        selected = self.selected_option.get()
        image_paths = {
            "posicion munieca x": "resultados\\graficos\\posicion_x_muneca.png",
            "posicion munieca y": "resultados\\graficos\\posicion_y_muneca.png",
            "velocidad munieca y": "resultados\\graficos\\velocidad_y_muneca.png",
            "velocidad munieca x": "resultados\\graficos\\velocidad_x_muneca.png"
        }
        self.load_and_show_image(image_paths.get(selected, "loadImage.png"))

    def on_slider_changed(self, val):
        if not self.is_playing:
            self.frame_number = int(val)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            self.show_frame()
            self.btn_restart.config(state=tk.NORMAL)
            if self.frame_number == self.frames - 1:
                self.btn_play.config(state=tk.DISABLED)

    def video_ready_callback(self):
        self.btn_play.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_restart.config(state=tk.NORMAL)

    def play_video(self):
        if self.video_cap:
            self.is_playing = True
            self.btn_play.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_restart.config(state=tk.DISABLED)
            self.show_frame()

    def stop_video(self):
        if self.video_cap:
            self.is_playing = False
            if self.frame_number < self.frames - 1:
                self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def restart_video(self):
        if self.video_cap:
            self.frame_number = 0
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
            self.show_frame()
            self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def show_frame(self):
        ret, frame = self.video_cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (400, 250))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk)
            if self.is_playing:
                self.video_label.after(30, self.show_frame)
            else:
                self.btn_restart.config(state=tk.NORMAL)
            self.slider.set(self.frame_number)
            self.frame_number += 1
            self.draw_indicator_line()
            if self.frame_number == self.frames:
                self.btn_play.config(state=tk.DISABLED)
                self.btn_restart.config(state=tk.NORMAL)
        else:
            self.stop_video()

    def load_and_show_image(self, image_path):
        self.load_image(image_path)
        self.show_image()

    def load_image(self, image_path):
        self.image_path = image_path
        self.image = Image.open(self.image_path)

    def show_image(self):
        image_width, image_height = self.image.size
        resized_image = self.image.resize((int(image_width), int(image_height)))
        self.img_tk = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.img_tk)
        self.image_label.image = self.img_tk

    def draw_indicator_line(self):
        image_array = np.array(self.image)
        height, width, _ = image_array.shape
        indicator_x = int((self.frame_number / self.frames * (width * 0.74))) + int(width * 0.14)
        start_point = (indicator_x, 0)
        end_point = (indicator_x, height)
        cv2.line(image_array, start_point, end_point, (255, 0, 0, 255), thickness=2)
        image_with_line = Image.fromarray(image_array)
        img_tk_with_line = ImageTk.PhotoImage(image_with_line)
        self.image_label.config(image=img_tk_with_line)
        self.image_label.image = img_tk_with_line

def on_closing():
    sys.exit(0)

def main():
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = VideoPlayerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
