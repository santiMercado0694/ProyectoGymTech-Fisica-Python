import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import prueba
import sys

class VideoPlayerApp:
    frameNumber = 0
    frames = 0

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
        self.btn_open.grid(column=0, row=0)

        # Frame para los botones de control (play, pause, restart)
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

        # Frame para la imagen adicional a la derecha
        self.image_frame = tk.Frame(self.master)
        self.image_frame.pack(padx=10, pady=10)

        # Crear un label para la imagen
        self.image_label = tk.Label(self.image_frame)
        self.image_label.pack() 
        
        # Cargamos la imagen de presentación
        self.load_image("loadImage.png")
        self.show_image()

    def update_slider_position(self, val):
        # Actualizar la posición del slider
        self.slider.set(val)

    def open_video(self):
     self.video_path = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4;*.avi;*.mkv")])
     if self.video_path:
        # Llama a la funcion track_pose de prueba.py
        prueba.video_ready_callback = self.video_ready_callback
        prueba.track_pose(self.video_path)
        
        # Verificar si ya existe una barra de reproducción y destruirla si es el caso
        if hasattr(self, 'slider') and self.slider is not None:
            self.slider.destroy()
            self.frameNumber = 0
                   
        # Abre el video generado
        self.video_cap = cv2.VideoCapture('resultados\\video\\tracked_video.mp4')

        # Consigue los frames totales para generar la slider con ese numero maximo
        self.frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider = tk.Scale(from_=0, to=self.frames - 1, orient=tk.HORIZONTAL, command=self.on_slider_changed)
        self.slider.pack(fill=tk.X)
        
        # Ejecuta graficos.py para generar los graficos correspondientes
        exec(open('graficos.py').read(),globals())
        
        # Cargar y mostrar el gráfico en el frame
        self.load_image("resultados/graficos/subgraficos.png" )
        self.show_image()
        
    def on_slider_changed(self, val):
        if not self.is_playing:
            self.frameNumber = int(val)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumber)
            self.show_frame()
            self.btn_restart.config(state=tk.NORMAL)  # Habilitar el botón de reinicio cuando se ajusta manualmente el fotograma
            if self.frameNumber == self.frames - 1:  # Verificar si se alcanza el último fotograma
                self.btn_play.config(state=tk.DISABLED)  # Deshabilitar el botón de play al llegar al final

    # Funcion auxiliar para que compruebe que se genero el video 
    def video_ready_callback(self):
        self.btn_play.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)     # Deshabilitar el botón de pausa al inicio
        self.btn_restart.config(state=tk.NORMAL)  

    def play_video(self):
        if self.video_cap:
            self.is_playing = True
            self.btn_play.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_restart.config(state=tk.DISABLED)  # Deshabilitar el botón de reinicio mientras se reproduce
            self.show_frame()

    def stop_video(self):
        if self.video_cap:
            self.is_playing = False
            if self.frameNumber < self.frames - 1:  # Verificar si no se ha llegado al final del video
                self.btn_play.config(state=tk.NORMAL)  # Habilitar el botón de play solo si no se ha llegado al final
            self.btn_stop.config(state=tk.DISABLED)

    def restart_video(self):
        if self.video_cap:
            self.frameNumber = 0
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumber)
            self.show_frame()
            self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def show_frame(self):
        ret, frame = self.video_cap.read()
        if ret:
            # Redimensionar el fotograma
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
            self.update_slider_position(self.frameNumber)
            self.frameNumber += 1

            # Verificar si se alcanza el último fotograma
            if self.frameNumber == self.frames:
                self.btn_play.config(state=tk.DISABLED)  # Deshabilitar el botón de play al llegar al final
                self.btn_restart.config(state=tk.NORMAL) # Habilitar el botón de reinicio al llegar al final del video
        else:
            self.stop_video()

    def load_image(self,ruta):
        # Cargar una imagen desde el archivo
        self.image_path = ruta 
        self.image = Image.open(self.image_path)

    def show_image(self):
        # Mostrar la imagen en el frame
        image_width, image_height = self.image.size
        resized_image = self.image.resize((int(image_width / 2), int(image_height / 2.5)))  # Ajustar el tamaño de la imagen
        self.img_tk = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.img_tk)
        self.image_label.image = self.img_tk

def on_closing():
    sys.exit(0)

def main():
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = VideoPlayerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
