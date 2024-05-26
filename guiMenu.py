import tkinter as tk
from PIL import Image, ImageTk
from guiCalculadora import VideoPlayerApp 
import customtkinter as ctk
from PIL import Image, ImageTk

class MenuApp:
    def __init__(self, master):
        self.master = master
        self.master.geometry("640x360")
        self.createWidgets()

    def createWidgets(self):
        # Asignar tamaño inicial de la ventana
        self.master.geometry("640x360")

        # Crear el menú desplegable
        resoluciones = ["1080p (HD)", "720p (HD)", "480p (SD)", "360p (SD)"]
        self.selected_resolution = ctk.StringVar(self.master)
        self.selected_resolution.set(resoluciones[3]) 
        self.resolution_menu = ctk.CTkOptionMenu(self.master, variable=self.selected_resolution, values=resoluciones, command=self.setResolution)
        self.resolution_menu.pack(anchor="ne", padx=10, pady=10)  

        # Cargar la imagen
        self.original_image = Image.open("loadImage.png")

        # Crear un marco para la imagen
        self.image_frame = ctk.CTkFrame(self.master)
        self.image_frame.pack(side="left", padx=5, pady=5)

        # Convertir la imagen a un objeto tkinter PhotoImage
        self.image = ImageTk.PhotoImage(self.original_image)

        # Crear un widget de etiqueta para mostrar la imagen dentro del marco
        self.image_label = ctk.CTkLabel(self.image_frame, image=self.image)
        self.image_label.image = self.image 
        self.image_label.pack()

        # Lista de botones
        self.buttons = []
        botones_iniciales = [("Start", self.startFunction), ("Ajustes", self.ajustesFunction), ("Salir", self.close_menu)]
        self.createButtons(botones_iniciales)

        self.setResolution()

    def close_menu(self):
        self.master.destroy()

    # Se activa cuando se usa el desplegable de resolución
    def setResolution(self, *args):
        selected_resolution = self.selected_resolution.get()
        new_resolution = "640x360"

        if selected_resolution == "1080p (HD)":
            new_resolution = "1920x1080"
        elif selected_resolution == "720p (HD)":
            new_resolution = "1280x720"
        elif selected_resolution == "480p (SD)":
            new_resolution = "854x480"
        elif selected_resolution == "360p (SD)":
            new_resolution = "640x360"

        self.master.geometry(new_resolution)
        self.resizeRoot(new_resolution)

    def resizeRoot(self, resolution):
        # Convertir la resolución al formato "widthxheight"
        width, height = resolution.split('x')

        # Redimensionar la imagen con la resolución especificada
        new_image_width = round(int(width) / 3)
        new_image_height = int(height)

        # Redimensionar la imagen
        self.original_image_resized = self.original_image.resize((new_image_width, new_image_height))

        # Convertir la imagen redimensionada a un objeto tkinter PhotoImage
        self.image_resized = ImageTk.PhotoImage(self.original_image_resized)

        # Actualizar la imagen en el widget de etiqueta
        self.image_label.configure(image=self.image_resized)
        self.image_label.image = self.image_resized  # Actualizar la referencia de la imagen

    def createButtons(self, button_info):
        # Crear un marco para los botones
        button_frame = ctk.CTkFrame(self.master)
        button_frame.pack(side="right", fill="both", expand=True)

        for name, command in button_info:
            button = ctk.CTkButton(button_frame, text=name, command=lambda method=command: method())
            button.pack(side="top", padx=5, pady=5, fill="both", expand=True)
            self.buttons.append(button)

    def startFunction(self):
        self.master.withdraw()  # Ocultar la ventana del menú inicial
        player_window = ctk.CTkToplevel(self.master)  # Crear una nueva ventana para la aplicación de reproducción de video
        self.player_app = VideoPlayerApp(player_window, self.master)  # Pasar la ventana del menú principal como parámetro

    def ajustesFunction(self):
        print("Ajustes button clicked")

def main():
    ctk.set_appearance_mode("dark")  # Modos: "System" (default), "Dark", "Light"
    ctk.set_default_color_theme("blue")  # Temas: "blue" (default), "green", "dark-blue"
    
    root = ctk.CTk()
    app = MenuApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()