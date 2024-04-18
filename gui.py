import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

class VideoPlayerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Reproductor de Video")
        
        self.video_path = None
        self.video_cap = None
        self.is_playing = False
        
        self.create_widgets()

    def create_widgets(self):
        # Frame principal para el reproductor de video
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack(padx=10, pady=10)

        self.btn_open = tk.Button(self.video_frame, text="Seleccionar Video", command=self.open_video)
        self.btn_open.pack(pady=10)

        self.btn_play = tk.Button(self.video_frame, text="Reproducir", command=self.play_video, state=tk.DISABLED)
        self.btn_play.pack(pady=5)

        self.btn_stop = tk.Button(self.video_frame, text="Detener", command=self.stop_video, state=tk.DISABLED)
        self.btn_stop.pack(pady=5)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(side=tk.LEFT)

        # Frame para la imagen adicional a la derecha
        self.image_frame = tk.Frame(self.master)
        self.image_frame.pack(padx=10, pady=10)

        # Crear un label para la imagen
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack()

        # Cargar y mostrar la imagen en el frame
        self.load_image()
        self.show_image()

    def open_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4;*.avi;*.mkv")])
        if self.video_path:
            self.video_cap = cv2.VideoCapture(self.video_path)
            self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.NORMAL)

    def play_video(self):
        if self.video_cap:
            self.is_playing = True
            self.btn_play.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.show_frame()

    def stop_video(self):
        if self.video_cap:
            self.is_playing = False
            self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.video_cap.release()

    def show_frame(self):
        ret, frame = self.video_cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.img_tk = img_tk
            self.video_label.config(image=img_tk)
            if self.is_playing:
                self.video_label.after(30, self.show_frame)
        else:
            self.stop_video()

    def load_image(self):
        # Cargar una imagen desde el archivo
        self.image_path = "plot.png"  # aca va la ruta de los graficos
        self.image = Image.open(self.image_path)

    def show_image(self):
        # Mostrar la imagen en el frame
        image_width, image_height = self.image.size
        resized_image = self.image.resize((int(image_width / 2), int(image_height / 2)))  # Ajustar el tamaño de la imagen
        self.img_tk = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.img_tk)
        self.image_label.image = self.img_tk

def main():
    root = tk.Tk()
    app = VideoPlayerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
