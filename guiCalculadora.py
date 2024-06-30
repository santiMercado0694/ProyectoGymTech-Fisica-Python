import os
import sys
import cv2
import srcCalculadora
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import customtkinter as ctk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk


class VideoPlayerApp:
    frameNumber = 0
    frames = 30

    def __init__(self, master, menu_window):
        self.master = master
        self.menu_window = menu_window
        self.master.title("Reproductor de Video")

        self.video_path = None
        self.video_cap = None
        self.is_playing = False

        self.create_widgets()

    # Frame principal para el reproductor de video
    def __create_frame_reproductor_video(self, gui):
        self.video_frame = ctk.CTkFrame(self.master)
        self.video_frame.pack(padx=10, pady=10)

        self.btn_open = ctk.CTkButton(
            self.video_frame, text="Seleccionar Video", command=self.open_video
        )
        self.btn_open.grid(column=0, row=0)

        self.btn_goBack = ctk.CTkButton(
            self.video_frame, text="Volver al Menú", command=self.back_to_menu
        )
        self.btn_goBack.grid(column=0, row=1)

        # Agrega un campo de entrada para la masa de la pesa en la interfaz de usuario
        self.masa_entry_label = ctk.CTkLabel(self.video_frame, text="Masa de la pesa:")
        self.masa_entry_label.grid(column=0, row=2)
        self.masa_entry = ctk.CTkEntry(self.video_frame)
        self.masa_entry.grid(column=1, row=2)

    # Crea el frame de los botones de control
    def __create_frame_botones_control(self, gui):
        self.control_frame = ctk.CTkFrame(self.video_frame)
        self.control_frame.grid(column=1, row=0)

        self.btn_play = ctk.CTkButton(
            self.control_frame,
            text="\u23F5",
            command=self.play_video,
            state=ctk.DISABLED,
        )
        self.btn_play.pack(side=ctk.LEFT)

        self.btn_stop = ctk.CTkButton(
            self.control_frame,
            text="\u23F8",
            command=self.stop_video,
            state=ctk.DISABLED,
        )
        self.btn_stop.pack(side=ctk.LEFT)

        self.btn_restart = ctk.CTkButton(
            self.control_frame,
            text="\u23F9",
            command=self.restart_video,
            state=ctk.DISABLED,
        )
        self.btn_restart.pack(side=ctk.LEFT)

        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.grid(column=1, row=1)

    # Frame para la imagen adicional a la derecha
    def __create_frame_imagen_derecha(self, gui):
        self.image_frame = ctk.CTkFrame(self.master)
        self.image_frame.pack(padx=10, pady=10)

    # Crear un label para la imagen
    def __create_frame_image_label(self, gui):
        self.image_label = ctk.CTkLabel(self.image_frame, text="")
        self.image_label.pack()

    # Cargamos la imagen de presentación
    def __create_frame_imagen_presentacion(self, gui):
        self.load_image("loadImage.png")
        self.show_image()

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
            tk.messagebox.showerror(
                "Error",
                "¡Por favor, ingrese un valor numérico para la masa de la pesa!",
            )
            return
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Archivos de Video", "*.mp4;*.avi;*.mkv")]
        )
        if self.video_path:
            # Llama a la funcion track_pose de srcCalculadora.py
            srcCalculadora.video_ready_callback = self.video_ready_callback
            srcCalculadora.track_pose(self.video_path, masa_pesa)

            # Verificar si ya existe una barra de reproducción y destruirla si es el caso
            if hasattr(self, "slider") and self.slider is not None:
                self.slider.destroy()
                self.frameNumber = 0

            # Abre el video generado
            self.video_cap = cv2.VideoCapture("resultados\\video\\tracked_video.mp4")

            # Consigue los frames totales para generar la slider con ese numero maximo
            self.frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Frame principal para el deslizador
            self.slider_frame = ctk.CTkFrame(self.master)
            self.slider_frame.pack(pady=10)

            # Crear el deslizador con una longitud fija
            slider_length = 466
            self.slider = ctk.CTkSlider(
                self.slider_frame,
                from_=0,
                to=self.frames - 1,
                orientation="horizontal",
                command=self.on_slider_changed,
                width=slider_length,
            )
            self.slider.pack(fill="x", expand=True)

            # Centrar el frame del deslizador debajo del frame de la imagen
            self.slider_frame.place(
                in_=self.image_frame, relx=0.5, rely=1.0, anchor=ctk.CENTER
            )

            # Ejecuta graficos.py para generar los graficos correspondientes
            self.calcularVelocidadAceleracion()

            # Abre el drop-down menu para seleccionar el grafico deseado
            self.seleccion_imagen()

            self.show_frame()

    # Selecciona una imagen default para el video
    def seleccion_imagen(self):
        self.load_image("resultados\\graficos\\posicion_x_muneca.png")
        self.show_image()
        self.dropdown_frame = ctk.CTkFrame(self.video_frame)
        self.dropdown_frame.grid(column=2, row=0, padx=10)

        self.options = [
            "Posicion Muñeca x",
            "Posicion Muñeca y",
            "Angulo del Brazo",
            "Velocidad Angular",
            "Aceleracion Angular",
            "Fuerza Bicep",
            "Trabajo Bicep",
            "Energia cinetica",
            "Energia potencial",
            "Energia mecanica",
        ]
        self.selected_option = ctk.StringVar(value=self.options[0])

        self.dropdown = ctk.CTkComboBox(
            self.dropdown_frame,
            variable=self.selected_option,
            values=self.options,
            command=self.on_dropdown_changed,
        )
        self.dropdown.pack()

        self.dropdown.bind("<ComboboxSelected>", lambda e: self.on_dropdown_changed)

    # Encuentra la ruta de la imagen seleccionada
    def on_dropdown_changed(self, value):
        selected = self.selected_option.get()
        image_paths = {
            "Posicion Muñeca x": "resultados\\graficos\\posicion_x_muneca.png",
            "Posicion Muñeca y": "resultados\\graficos\\posicion_y_muneca.png",
            "Angulo del Brazo": "resultados\\graficos\\angulo_del_brazo.png",
            "Velocidad Angular": "resultados\\graficos\\velocidad_angular.png",
            "Aceleracion Angular": "resultados\\graficos\\aceleracion_angular.png",
            "Fuerza Bicep": "resultados\\graficos\\fuerza_bicep.png",
            "Trabajo Bicep": "resultados\\graficos\\trabajo_bicep.png",
            "Energia cinetica": "resultados\\graficos\\energia_cinetica.png",
            "Energia potencial": "resultados\\graficos\\energia_potencial.png",
            "Energia mecanica": "resultados\\graficos\\energia_mecanica.png",
        }
        self.load_image(image_paths.get(value, "loadImage.png"))
        self.show_image()
        self.draw_indicator_line()

    def on_slider_changed(self, val):
        if not self.is_playing:
            self.frameNumber = int(val)
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumber)
            self.show_frame()
            self.btn_restart.configure(
                state=ctk.NORMAL
            )  # Habilitar el botón de reinicio cuando se ajusta manualmente el fotograma
            if (
                self.frameNumber == self.frames - 1
            ):  # Verificar si se alcanza el último fotograma
                self.btn_play.configure(
                    state=ctk.DISABLED
                )  # Deshabilitar el botón de play al llegar al final

    # Funcion auxiliar para que compruebe que se genero el video
    def video_ready_callback(self):
        self.btn_play.configure(state=ctk.NORMAL)
        self.btn_stop.configure(
            state=ctk.DISABLED
        )  # Deshabilitar el botón de pausa al inicio
        self.btn_restart.configure(state=ctk.NORMAL)

    # Reproduce el video
    def play_video(self):
        if self.video_cap:
            self.is_playing = True
            self.btn_play.configure(state=ctk.DISABLED)
            self.btn_stop.configure(state=ctk.NORMAL)
            self.btn_restart.configure(
                state=ctk.DISABLED
            )  # Deshabilitar el botón de reinicio mientras se reproduce
            self.show_frame()

    # Pausa el video
    def stop_video(self):
        if self.video_cap:
            self.is_playing = False
            if (
                self.frameNumber < self.frames - 1
            ):  # Verificar si no se ha llegado al final del video
                self.btn_play.configure(
                    state=ctk.NORMAL
                )  # Habilitar el botón de play solo si no se ha llegado al final
            self.btn_stop.configure(state=ctk.DISABLED)

    # Reinicia el video
    def restart_video(self):
        if self.video_cap:
            self.frameNumber = 0
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.frameNumber)
            self.show_frame()
            self.btn_play.configure(state=ctk.NORMAL)
            self.btn_stop.configure(state=ctk.DISABLED)

    # Muestra el fotograma en el video
    def show_frame(self):
        ret, frame = self.video_cap.read()
        if ret:
            # Redimensionar el fotograma
            frame_resized = cv2.resize(frame, (400, 250))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            self.video_label.img_tk = img_tk
            self.video_label.configure(image=img_tk)
            if self.is_playing:
                self.video_label.after(30, self.show_frame)
            else:
                self.btn_restart.configure(state=ctk.NORMAL)
            self.update_slider_position(self.frameNumber)
            self.frameNumber += 1
            self.draw_indicator_line()
            # Verificar si se alcanza el último fotograma
            if self.frameNumber == self.frames:
                self.btn_play.configure(
                    state=ctk.DISABLED
                )  # Deshabilitar el botón de play al llegar al final
                self.btn_restart.configure(
                    state=ctk.NORMAL
                )  # Habilitar el botón de reinicio al llegar al final del video
        else:
            self.stop_video()

    # Cargar una imagen desde el archivo
    def load_image(self, ruta):
        self.image_path = ruta
        self.image = Image.open(self.image_path)

    # Mostrar la imagen en el frame
    def show_image(self):
        image_width, image_height = self.image.size
        resized_image = self.image.resize(
            (int(image_width), int(image_height))
        )  # Ajustar el tamaño de la imagen
        self.img_tk = ImageTk.PhotoImage(resized_image)
        self.image_label.configure(image=self.img_tk)
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
        cv2.line(image_array, start_point, end_point, (255, 0, 0, 255), thickness=2)

        # Convertir la imagen modificada a formato ImageTk
        image_with_line = Image.fromarray(image_array)
        img_tk_with_line = ImageTk.PhotoImage(image_with_line)

        # Mostrar la imagen actualizada en el label
        self.image_label.configure(image=img_tk_with_line)
        self.image_label.image = img_tk_with_line

    def generarGraficos(self, tiempo, datos, titulos, unidades):
        errory = 0.004
        errorx = 0.004
        errorangulo = 0.019

        if not os.path.exists("resultados/graficos"):
            os.makedirs("resultados/graficos")

        for i, (dato, titulo, unidad) in enumerate(zip(datos, titulos, unidades)):
            fig, ax = plt.subplots(figsize=(6, 4))

            # Suavizar los datos con una interpolación cúbica
            tiempo_suave = np.linspace(tiempo.min(), tiempo.max(), 500)
            datos_suave = np.interp(tiempo_suave, tiempo, dato)

            # Agrego errores
            if titulo == "Posicion X Muneca":
                ax.fill_between(
                    tiempo_suave,
                    datos_suave - errorx,
                    datos_suave + errorx,
                    color="r",
                    alpha=0.3,  # Aumenta la visibilidad de la banda de error
                    edgecolor="r",  # Añade un borde rojo para el error
                    linewidth=1,  # Grosor del borde de la banda de error
                    label="Error ±0.004",
                )
            elif titulo == "Posicion Y Muneca":
                ax.fill_between(
                    tiempo_suave,
                    datos_suave - errory,
                    datos_suave + errory,
                    color="r",
                    alpha=0.3,
                    edgecolor="r",
                    linewidth=1,
                    label="Error ±0.004",
                )
            elif titulo == "Angulo del brazo":
                ax.fill_between(
                    tiempo_suave,
                    datos_suave - errorangulo,
                    datos_suave + errorangulo,
                    color="r",
                    alpha=0.3,
                    edgecolor="r",
                    linewidth=1,
                    label="Error ±0.019",
                )

            # Aumenta el grosor de la línea del gráfico
            ax.plot(tiempo_suave, datos_suave, linestyle="-", color="b", linewidth=2)

            ax.set_title(titulo)
            ax.set_xlabel("Tiempo(seg)")
            ax.set_ylabel(unidad)
            ax.grid(True)

            # Guardar el gráfico con el nombre correspondiente
            filename = f'resultados/graficos/{titulo.replace(" ", "_").lower()}.png'
            plt.savefig(filename)
            print(f"Grafico guardado en {filename}")
            plt.close()

    def generarGraficoCombinado(self, tiempo, dataframe):
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Verificación de datos
        if dataframe.empty or len(tiempo) == 0:
            print("El dataframe o el tiempo están vacíos.")
            return

        # Posición en X e Y en el eje y principal (izquierdo)
        ax1.plot(tiempo, dataframe["LEFT_WRIST_x(m)"], label="Posición X (m)", linestyle='-', color='b', linewidth=2)
        ax1.plot(tiempo, dataframe["LEFT_WRIST_y(m)"], label="Posición Y (m)", linestyle='-', color='g', linewidth=2)
        ax1.set_xlabel("Tiempo (seg)", fontsize=14)
        ax1.set_ylabel("Posición (m)", fontsize=14)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.legend(loc="upper left", fontsize=12)

    # Crear segundo eje y para la velocidad y aceleración
        ax2 = ax1.twinx()
        ax2.plot(tiempo, dataframe["Velocidad_angular"], label="Velocidad Angular (rad/seg)", linestyle='--', color='r', linewidth=2)
        ax2.plot(tiempo, dataframe["Aceleracion_angular"], label="Aceleración Angular (rad/seg²)", linestyle='--', color='k', linewidth=2)
        ax2.set_ylabel("Velocidad y Aceleración (rad/seg y rad/seg²)", fontsize=14)
        ax2.plot(tiempo, dataframe["Angulo"], label="Angulo del brazo (rad)", linestyle='-.', color='c', linewidth=2)
        ax2.tick_params(axis='y', labelsize=12)
        ax2.legend(loc="upper right", fontsize=12)

        fig.tight_layout()
        ax1.grid(True)
        plt.title("Gráfico Combinado de Posición, Velocidad y Aceleración Angular", fontsize=16)

        filename = 'resultados/graficos/grafico_combinado.png'
        plt.savefig(filename)
        print(f"Grafico combinado guardado en {filename}")
        plt.close()

    def calcularVelocidadAceleracion(self):
        dataframe = pd.read_csv("resultados/documents/data.csv", index_col=[0])
        # Eliminar filas con valores NaN
        dataframe.dropna(inplace=True)

        # Guardar los datos actualizados en el mismo archivo CSV
        dataframe.to_csv("resultados/documents/data.csv")

        tiempo = dataframe["tiempo(seg)"]
        datos = [
            dataframe["LEFT_WRIST_x(m)"],
            dataframe["LEFT_WRIST_y(m)"],
            dataframe["Angulo"],
            dataframe["Velocidad_angular"],
            dataframe["Aceleracion_angular"],
            dataframe["Fuerza_bicep"],
            dataframe["Trabajo_bicep"],
            dataframe["Energia_cinetica"],
            dataframe["Energia_potencial"],
            dataframe["Energia_Mecanica"],
        ]
        titulos = [
            "Posicion X Muneca",
            "Posicion Y Muneca",
            "Angulo del brazo",
            "Velocidad Angular",
            "Aceleracion Angular",
            "Fuerza Bicep",
            "Trabajo Bicep",
            "Energia cinetica",
            "Energia potencial",
            "Energia mecanica",
        ]
        unidades = [
            "m",
            "m",
            "rad",
            "rad/seg",
            "rad/seg^2",
            "Newton",
            "J",
            "J",
            "J",
            "J",
        ]

        self.generarGraficos(tiempo, datos, titulos, unidades)
        self.generarGraficoCombinado(tiempo, dataframe)



def on_closing():
    sys.exit(0)


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    app = VideoPlayerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
