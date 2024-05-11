import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import prueba
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class VideoPlayerApp:
    frameNumber = 0
    frames = 30

    def __init__(self, master,menu_window):
        self.master = master
        self.menu_window = menu_window
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

        self.btn_goBack = tk.Button(self.video_frame, text="Volver al Menú", command=self.back_to_menu)
        self.btn_goBack.grid(column=0, row=1)

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

    def back_to_menu(self):
        self.master.destroy()  # Cerrar la ventana actual (ventana de video)
        self.menu_window.deiconify()  # Mostrar la ventana del menú principal

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

       # Frame principal para el deslizador
        self.slider_frame = tk.Frame(self.master)
        self.slider_frame.pack(pady=10)

        # Crear el deslizador con una longitud fija
        slider_length = 466
        self.slider = tk.Scale(self.slider_frame, from_=0, to=self.frames - 1, orient=tk.HORIZONTAL, command=self.on_slider_changed, length=slider_length)
        self.slider.pack(fill=tk.X)

        # Centrar el frame del deslizador debajo del frame de la imagen
        self.slider_frame.place(in_=self.image_frame, relx=0.5, rely=1.0, anchor=tk.CENTER)
       
        # Ejecuta graficos.py para generar los graficos correspondientes
        self.createNewDataframe()
        
        #Abre el drop-down menu para seleccionar el grafico deseado
        self.seleccion_imagen()

        self.show_frame()

    

    def seleccion_imagen (self) :
        self.load_image("resultados\\graficos\\posicion_x_muneca.png")
        self.show_image()
        self.dropdown_frame = tk.Frame(self.video_frame)
        self.dropdown_frame.grid(column=2, row=0, padx=10)

        self.options = ["posicion munieca x", "posicion munieca y", "velocidad munieca y", "velocidad munieca x"]
        self.selected_option = tk.StringVar(value=self.options[0])

        self.dropdown = ttk.Combobox(self.dropdown_frame, textvariable=self.selected_option, values=self.options)
        self.dropdown.pack()
        
        self.dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_changed)
        
    #Encuentra la ruta de la imagen seleccionada 
    def on_dropdown_changed(self, event):
        selected = self.selected_option.get()
        image_paths = {
            "posicion munieca x": "resultados\\graficos\\posicion_x_muneca.png",
            "posicion munieca y": "resultados\\graficos\\posicion_y_muneca.png",
            "velocidad munieca y": "resultados\\graficos\\velocidad_y_muneca.png",
            "velocidad munieca x": "resultados\\graficos\\velocidad_x_muneca.png"
        }
        self.load_image(image_paths.get(selected, "loadImage.png"))
        self.show_image()
        self.draw_indicator_line()

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
            self.draw_indicator_line()
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
        resized_image = self.image.resize((int(image_width), int(image_height)))  # Ajustar el tamaño de la imagen
        self.img_tk = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=self.img_tk)
        self.image_label.image = self.img_tk

    def draw_indicator_line(self):
        # Crear una imagen a partir del archivo existente
        image_array = np.array(self.image)
        
        # Dimensiones de la imagen
        height, width, _ = image_array.shape
        
        # Calcular la posición x de la línea indicadora basada en el fotograma actual
        indicator_x = int((self.frameNumber / self.frames * (width * 0.74))) 
        indicator_x += int(width * 0.14)
        
        # Calcular el punto de inicio y final de la línea (y)
        start_point = (indicator_x, 0)
        end_point = (indicator_x, height)
        
        # Dibujar la línea indicadora (por ejemplo, en color rojo)
        cv2.line(image_array, start_point, end_point, (255, 0,0, 255), thickness=2)
        
        # Convertir la imagen modificada a formato ImageTk
        image_with_line = Image.fromarray(image_array)
        img_tk_with_line = ImageTk.PhotoImage(image_with_line)
        
        # Mostrar la imagen actualizada en el label
        self.image_label.config(image=img_tk_with_line)
        self.image_label.image = img_tk_with_line

    def generarSubgraficos(self, tiempo, datos, titulos, unidades):
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        for i, ax in enumerate(axs.flat):
            ax.plot(tiempo, datos[i], marker='o', linestyle='-')
            ax.set_title(titulos[i])
            ax.set_xlabel('Tiempo(seg)')
            ax.set_ylabel(unidades[i])
            ax.grid(True)
        plt.tight_layout()
        plt.savefig('resultados/graficos/subgraficos.png')
        print("Grafico de subgraficos guardado en resultados/graficos/subgraficos.png")

    def generarGraficos(self, tiempo, datos, titulos, unidades):
        # Crear el directorio si no existe
        for i, (dato, titulo, unidad) in enumerate(zip(datos, titulos, unidades)):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(tiempo, dato, marker='o', linestyle='-')
            ax.set_title(titulo)
            ax.set_xlabel('Tiempo(seg)')
            ax.set_ylabel(unidad)
            ax.grid(True)
            
            # Guardar el gráfico con el nombre correspondiente
            filename = f'resultados/graficos/{titulo.replace(" ", "_").lower()}.png'
            plt.savefig(filename)
            print(f"Grafico guardado en {filename}")
            plt.close()

    def createNewDataframe(self):
        dataframe = pd.read_csv('resultados/documents/data.csv', index_col=[0])
        df2 = pd.DataFrame()
        df2['posicion_x'] = dataframe['LEFT_WRIST_x(m)']
        df2['posicion_y'] = dataframe['LEFT_WRIST_y(m)']
        df2['tiempo'] = dataframe['tiempo(seg)']

        df2['dx'] = df2['posicion_x'].diff()
        df2['dy'] = df2['posicion_y'].diff()
        df2['dt'] = df2['tiempo'].diff()

        df2.dropna(inplace=True)

        df2['vx'] = df2['dx'] / df2['dt']
        df2['vy'] = df2['dy'] / df2['dt']

        tiempo = df2['tiempo']
        datos = [df2['posicion_x'], df2['posicion_y'], df2['vx'], df2['vy']]
        titulos = ['Posicion X Muneca', 'Posicion Y Muneca', 'Velocidad X Muneca', 'Velocidad Y Muneca']
        unidades = ['m', 'm', 'm/s', 'm/s']

        self.generarGraficos(tiempo, datos, titulos, unidades)

def on_closing():
    sys.exit(0)

def main():
    root = tk.Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = VideoPlayerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
