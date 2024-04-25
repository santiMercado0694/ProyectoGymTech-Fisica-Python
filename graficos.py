import pandas as pd
import matplotlib.pyplot as plt

def generarSubgraficos(tiempo, datos, titulos, unidades):
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
    
    def generarGrafico(tiempo, posicion, string):
    ruta = 'resultados/graficos/'+string+'.png'
    #Crea una nueva figura
    plt.figure()
    plt.plot(tiempo, posicion,linestyle='-')
    plt.title(string)
    plt.xlabel('Tiempo')
    plt.ylabel('Metros')
    plt.grid(True)
    plt.savefig(ruta)
    #plt.show()

def createNewDataframe():
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

    generarSubgraficos(tiempo, datos, titulos, unidades)

createNewDataframe()
