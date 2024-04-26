import tkinter as tk
from PIL import Image, ImageTk
from gui import VideoPlayerApp 

class MenuApp:
    def __init__(self, master):
        self.master = master
        self.master.title("GymTech")
        self.createWidgets()

    def createWidgets(self):
        # Asignar tamaño inicial de la ventana
        self.master.geometry("640x360")

        # Crear el menú desplegable
        resolutions = ["1080p (HD)", "720p (HD)", "480p (SD)", "360p (SD)"]
        self.selected_resolution = tk.StringVar(self.master)
        self.selected_resolution.set(resolutions[3]) 
        self.resolution_menu = tk.OptionMenu(self.master, self.selected_resolution, *resolutions, command=self.setResolution)
        self.resolution_menu.pack(anchor="ne", padx=10, pady=10)  

        # Cargar la imagen
        self.original_image = Image.open("loadImage.png")

        # Crear un marco para la imagen con margen
        self.image_frame = tk.Frame(self.master, padx=5, pady=5)
        self.image_frame.pack(side="left")

        # Convertir la imagen a un objeto tkinter PhotoImage
        self.image = ImageTk.PhotoImage(self.original_image)

        # Crear un widget de etiqueta para mostrar la imagen dentro del marco
        self.image_label = tk.Label(self.image_frame, image=self.image)
        self.image_label.image = self.image 
        self.image_label.pack()

        # Lista de botones
        self.buttons = []
        initial_buttons = [("Start", self.startFunction), ("Ajustes", self.ajustesFunction)]
        self.createButtons(initial_buttons)

        self.setResolution()

    # Se activa cuando se usa el desplegable de resolución
    def setResolution(self, *args):
        selected_resolution = self.selected_resolution.get()
        new_resolution = "0x0"

        if selected_resolution == "1080p (HD)":
            new_resolution = "1920x1080"
            self.master.geometry(new_resolution)
        elif selected_resolution == "720p (HD)":
            new_resolution = "1280x720"
            self.master.geometry(new_resolution)
        elif selected_resolution == "480p (SD)":
            new_resolution = "854x480"
            self.master.geometry(new_resolution)
        elif selected_resolution == "360p (SD)":
            new_resolution = "640x360"
            self.master.geometry(new_resolution)

        self.resizeRoot(new_resolution)

    def resizeRoot(self, resolution):
        # Convertir la resolución al formato "widthxheight"
        width, height = resolution.split('x')

        # Redimensionar la imagen con la resolución especificada
        new_image_width = round( int(width) / 3 )
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
        button_frame = tk.Frame(self.master)
        button_frame.pack(side="right", fill="both", expand=True)

        for name, command in button_info:
            button = tk.Button(button_frame, text=name, command=lambda method=command: method())
            button.pack(side="top", padx=5, pady=5, fill="both", expand=True)
            self.buttons.append(button)       

    def startFunction(self):
        self.master.destroy()
    
        root = tk.Tk()
        app = VideoPlayerApp(root)

    def ajustesFunction(self):
        print("Ajustes button clicked")

def main():
    root = tk.Tk()
    app = MenuApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
