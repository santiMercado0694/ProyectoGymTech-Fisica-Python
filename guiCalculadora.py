import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import srcCalculadora
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

    # Frame principal para el reproductor de video
    def __create_frame_reproductor_video(self, gui):
        gui.video_frame = tk.Frame(gui.master)
        gui.video_frame.pack(padx=10, pady=10)

        gui.btn_open = tk.Button(gui.video_frame, text="Seleccionar Video", command=gui.open_video)
        gui.btn_open.grid(column=0, row=0)

        gui.btn_goBack = tk.Button(gui.video_frame, text="Volver al Menú", command=gui.back_to_menu)
        gui.btn_goBack.grid(column=0, row=1)
        
        # Agrega un campo de entrada para la masa de la pesa en la interfaz de usuario
        gui.masa_entry_label = tk.Label(gui.video_frame, text="Masa de la pesa:")
        gui.masa_entry_label.grid(column=0, row=2)
        gui.masa_entry = tk.Entry(gui.video_frame)
        gui.masa_entry.grid(column=1, row=2)

    #Crea el frame de los botones de control
    def __create_frame_botones_control(self,gui):
        gui.control_frame = tk.Frame(gui.video_frame)
        gui.control_frame.grid(column=1, row=0)

        gui.btn_play = tk.Button(gui.control_frame, text="\u23F5", command=gui.play_video, state=tk.DISABLED)
        gui.btn_play.pack(side=tk.LEFT)

        gui.btn_stop = tk.Button(gui.control_frame, text="\u23F8", command=gui.stop_video, state=tk.DISABLED)
        gui.btn_stop.pack(side=tk.LEFT)

        gui.btn_restart = tk.Button(gui.control_frame, text="\u23F9", command=gui.restart_video, state=tk.DISABLED)
        gui.btn_restart.pack(side=tk.LEFT)

        gui.video_label = tk.Label(gui.video_frame)
        gui.video_label.grid(column=1, row=1)

    # Frame para la imagen adicional a la derecha
    def __create_frame_imagen_derecha(self, gui):
        gui.image_frame = tk.Frame(gui.master)
        gui.image_frame.pack(padx=10, pady=10)

    # Crear un label para la imagen
    def __create_frame_image_label(self,gui):
        gui.image_label = tk.Label(gui.image_frame)
        gui.image_label.pack() 

    # Cargamos la imagen de presentación
    def __create_frame_imagen_presentacion(self,gui):
        gui.load_image("loadImage.png")
        gui.show_image()

    def create_widgets(self):
        self.__create_frame_reproductor_video(self)
        self.__create_frame_botones_control(self)
        self.__create_frame_imagen_derecha(self)
        self.__create_frame_image_label(self)
        self.__create_frame_imagen_presentacion(self)

    def update_slider_position(self, val):
        # Actualizar la posición del slider
        self.slider.set(val)

    def back_to_menu(self):
        self.master.destroy()  # Cerrar la ventana actual (ventana de video)
        self.menu_window.deiconify()  # Mostrar la ventana del menú principal

    def open_video(self):
     try:
            masa_pesa = float(self.masa_entry.get())
     except ValueError:
            tk.messagebox.showerror("Error", "¡Por favor, ingrese un valor numérico para la masa de la pesa!")
            return
     self.video_path = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4;*.avi;*.mkv")])
     if self.video_path:
        # Llama a la funcion track_pose de srcCalculadora.py
        srcCalculadora.video_ready_callback = self.video_ready_callback
        srcCalculadora.track_pose(self.video_path,masa_pesa)
        
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

    #Selecciona una imagen default para el video
    def seleccion_imagen (self) :
        self.load_image("resultados\\graficos\\posicion_x_muneca.png")
        self.show_image()
        self.dropdown_frame = tk.Frame(self.video_frame)
        self.dropdown_frame.grid(column=2, row=0, padx=10)

        self.options = ["Posicion Muñeca x", "Posicion Muñeca y", "Angulo del Brazo", "Velocidad Angular", "Aceleracion Angular"]
        self.selected_option = tk.StringVar(value=self.options[0])

        self.dropdown = ttk.Combobox(self.dropdown_frame, textvariable=self.selected_option, values=self.options)
        self.dropdown.pack()
        
        self.dropdown.bind("<<ComboboxSelected>>", self.on_dropdown_changed)
        
    #Encuentra la ruta de la imagen seleccionada 
    def on_dropdown_changed(self, event):
        selected = self.selected_option.get()
        image_paths = {
            "Posicion Muñeca x": "resultados\\graficos\\posicion_x_muneca.png",
            "Posicion Muñeca y": "resultados\\graficos\\posicion_y_muneca.png",
            "Angulo del Brazo": "resultados\\graficos\\angulo_del_brazo.png",
            "Velocidad Angular": "resultados\\graficos\\velocidad_angular.png",
            "Aceleracion Angular": "resultados\\graficos\\aceleracion_angular.png",
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

    # Reproduce el video
    def play_video(self):
        if self.video_cap:
            self.is_playing = True
            self.btn_play.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_restart.config(state=tk.DISABLED)  # Deshabilitar el botón de reinicio mientras se reproduce
            self.show_frame()

    #Pausa el video
    def stop_video(self):
        if self.video_cap:
            self.is_playing = False
            if self.frameNumber < self.frames - 1:  # Verificar si no se ha llegado al final del video
                self.btn_play.config(state=tk.NORMAL)  # Habilitar el botón de play solo si no se ha llegado al final
            self.btn_stop.config(state=tk.DISABLED)

    #Reinicia el video
    def restart_video(self):
        if self.video_cap:
            self.frameNumber = 0
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumber)
            self.show_frame()
            self.btn_play.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    #Muestra el fotograma en el video
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

    # Cargar una imagen desde el archivo
    def load_image(self,ruta):
        self.image_path = ruta 
        self.image = Image.open(self.image_path)

    # Mostrar la imagen en el frame
    def show_image(self):
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
        df2['angulo'] = dataframe['Angulo']
        df2['tiempo'] = dataframe['tiempo(seg)']

        df2.dropna(inplace=True)

        df2['dif_angular'] = df2['angulo'].diff()
        df2['dif_temporal'] = df2['tiempo'].diff()
        df2['vel_angular'] = abs(df2['dif_angular'] / df2['dif_temporal'])

        df2['dif_vel_angular'] = df2['vel_angular'].diff()
        df2['aceleracion_angular'] = abs(df2['dif_vel_angular'] / df2['dif_temporal'])

        tiempo = df2['tiempo']
        datos = [df2['posicion_x'], df2['posicion_y'],df2['angulo'],df2['vel_angular'],df2['aceleracion_angular']]
        titulos = ['Posicion X Muneca', 'Posicion Y Muneca','Angulo del brazo','Velocidad Angular','Aceleracion Angular']
        unidades = ['m', 'm', 'rad', 'rad/seg', 'rad/seg^2']

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
