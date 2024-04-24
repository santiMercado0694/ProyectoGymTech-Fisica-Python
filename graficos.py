import pandas as pd
import matplotlib.pyplot as plt

def generarGrafico(tiempo,posicion):
    plt.plot(tiempo, posicion, marker='o', linestyle='-')
    plt.title('Posicion_x_muñeca')
    plt.xlabel('Tiempo')
    plt.ylabel('Metros')
    plt.grid(True)
    plt.savefig('resultados/graficos/posicion_x_muñeca.png')
    #plt.show()

def createNewDataframe():
    dataframe = pd.read_csv('resultados/documents/data.csv',index_col=[0])
    #Nuevo dataframe solo con muñeca en el eje x
    df2=pd.DataFrame(dataframe.LEFT_WRIST_x)

    #voy a suponer que, por ejemplo la mitad de la pantalla (0.5, porque está normalizado, va de 0 a 1) son 2 metros, aca usen la diferencia de posicion
    #entre una articulación y otra, y mídanse, esto está de ejemplo.
    #regla de 3. Si 0.5 son 2 metros, mi posición actual normalizada es ** metros. posición actual normalizada*2/0.5.
    df2['posicion']=4*df2.LEFT_WRIST_x
 
    #pdiferencia entre la posición y tiempo actual y el anterior. .diff hace eso.
    df2['tiempo']=dataframe['tiempo(seg)']
    df2['dx'] = df2['posicion'].diff()
    df2['dt'] = df2['tiempo'].diff()

    #Quito las filas con nan (la primera, porque resto el valor anterior que no existe)
    df2.dropna(inplace=True)

    #Calculo la velocidad
    df2['v']=df2['dx']/df2['dt']
    tiempo = df2['tiempo']
    posicion = df2['posicion']

    generarGrafico(tiempo,posicion)

createNewDataframe()



