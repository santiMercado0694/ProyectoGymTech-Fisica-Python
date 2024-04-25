import pandas as pd
import matplotlib.pyplot as plt

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
    dataframe = pd.read_csv('resultados/documents/data.csv',index_col=[0])
    #Nuevo dataframe solo con muñeca
    df2=pd.DataFrame(dataframe.LEFT_WRIST_x)
    df2['LEFT_WRIST_y']=dataframe['LEFT_WRIST_y']

    #voy a suponer que, por ejemplo la mitad de la pantalla (0.5, porque está normalizado, va de 0 a 1) son 2 metros, aca usen la diferencia de posicion
    #entre una articulación y otra, y mídanse, esto está de ejemplo.
    #regla de 3. Si 0.5 son 2 metros, mi posición actual normalizada es ** metros. posición actual normalizada*2/0.5.
    df2['posicion_x']=4*df2.LEFT_WRIST_x
    df2['posicion_y']=4*df2.LEFT_WRIST_y
 
    #pdiferencia entre la posición y tiempo actual y el anterior. .diff hace eso.
    df2['tiempo']=dataframe['tiempo(seg)']
    df2['dx'] = df2['posicion_x'].diff()
    df2['dy'] = df2['posicion_y'].diff()
    df2['dt'] = df2['tiempo'].diff()

    #Quito las filas con nan (la primera, porque resto el valor anterior que no existe)
    df2.dropna(inplace=True)

    #Calculo la velocidad
    df2['vx']=df2['dx']/df2['dt']
    df2['vy']=df2['dy']/df2['dt']

    tiempo = df2['tiempo']
    posicion_x = df2['posicion_x']
    posicion_y = df2['posicion_y']
    velocidad_x = df2['vx']
    velocidad_y = df2['vy']

    generarGrafico(tiempo,posicion_x, 'posicion_x_muneca')
    generarGrafico(tiempo,posicion_y, 'posicion_y_muneca')
    generarGrafico(tiempo,velocidad_x, 'velocidad_x_muneca')
    generarGrafico(tiempo,velocidad_y, 'velocidad_y_muneca')

createNewDataframe()



