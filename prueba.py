import math

Promedio_x_munieca = -0.039034449
ErrorX_munieca = 0.072658137

Promedio_y_munieca = -0.208962806
ErrorY_munieca = 0.077962483

Promedio_x_hombro = -0.01621702
ErrorX_hombro = 0.102053173

Promedio_y_hombro = 0.242088395
ErrorY_hombro = 0.10809871

#arctan((y2-y1)/(x2-x1))

#X1
def derivadaAngulo_X1 (x1,y1,x2,y2):
    return ((y2-y1)/((((y2-y1)**2)/((x2-x1)**2))*((x2-x1)**2)))
#X2
def derivadaAngulo_X2 (x1,y1,x2,y2):
    return -((y2-y1)/((((y2-y1)**2)/((x2-x1)**2))*((x2-x1)**2)))
#Y1
def derivadaAngulo_Y1 (x1,y1,x2,y2):
    return (-1/((x2-x1)*(((y2-y1)**2)/((x2-x1)**2))+1))
#Y2
def derivadaAngulo_Y2 (x1,y1,x2,y2):
    return (1/((x2-x1)*(((y2-y1)**2)/((x2-x1)**2))+1))


def medicionIndirectaAngulo():
    derivadaAnguloX1 = derivadaAngulo_X1(Promedio_x_munieca,Promedio_y_munieca,Promedio_x_hombro,Promedio_y_hombro)
    derivadaAnguloY1 = derivadaAngulo_Y1(Promedio_x_munieca,Promedio_y_munieca,Promedio_x_hombro,Promedio_y_hombro)
    derivadaAnguloX2 = derivadaAngulo_X2(Promedio_x_munieca,Promedio_y_munieca,Promedio_x_hombro,Promedio_y_hombro)
    derivadaAnguloY2 = derivadaAngulo_Y2(Promedio_x_munieca,Promedio_y_munieca,Promedio_x_hombro,Promedio_y_hombro)
    return (derivadaAnguloX1**2)*(ErrorX_munieca**2) + (derivadaAnguloY1**2)*(ErrorY_munieca**2) + (derivadaAnguloX2**2)*(ErrorX_hombro**2) + (derivadaAnguloY2**2)*(ErrorY_hombro**2)

print(math.sqrt(medicionIndirectaAngulo()))
## RETORNA 0.27806713127603444


############################# ERROR DE VELOCIDAD ANGULAR #############################

#Calculo la veolcidad angular como el la diferencia de angulos entre dos puntos en un tiempo determinado Wi = (theta2 - theta1) / (t2 - t1)

Promedio_Theta1 = 2.890067047
ErrorTheta = 0.278
Promedio_Theta2 = 2.890161182


Promedio_tiempo1 = 3.56
Promedio_tiempo2 = 3.59
ErrorTiempo1 = 0.03

#Derivada 1
def derivadaVelocidadAngular_1 (theta1,theta2,t1,t2):
    return (1/(t2-t1))

#Derivada 2
def derivadaVelocidadAngular_2 (theta1,theta2,t1,t2):
    return -1/(t2-t1)

#Derivada 3
def derivadaVelocidadAngular_3 (theta1,theta2,t1,t2):
    return (theta2-theta1)/((t2-t1)**2)

#Derivada 4
def derivadaVelocidadAngular_4 (theta1,theta2,t1,t2):
    return (-1*(theta2-theta1))/((t2-t1)**2)

def medicionIndirectaVelocidadAngular():
    derivadaVelocidadAngular1 = derivadaVelocidadAngular_1(Promedio_Theta1,Promedio_Theta2,Promedio_tiempo1,Promedio_tiempo2)
    derivadaVelocidadAngular2 = derivadaVelocidadAngular_2(Promedio_Theta1,Promedio_Theta2,Promedio_tiempo1,Promedio_tiempo2)
    derivadaVelocidadAngular3 = derivadaVelocidadAngular_3(Promedio_Theta1,Promedio_Theta2,Promedio_tiempo1,Promedio_tiempo2)
    derivadaVelocidadAngular4 = derivadaVelocidadAngular_4(Promedio_Theta1,Promedio_Theta2,Promedio_tiempo1,Promedio_tiempo2)
    print(derivadaVelocidadAngular1**2*(ErrorTheta**2))
    print(derivadaVelocidadAngular2**2*(ErrorTheta**2))
    print(derivadaVelocidadAngular3**2*(ErrorTiempo1**2))
    print(derivadaVelocidadAngular4**2*(ErrorTiempo1**2))


    return (((derivadaVelocidadAngular1**2)*(ErrorTheta**2)) + ((derivadaVelocidadAngular2**2)*(ErrorTheta**2)) + ((derivadaVelocidadAngular3**2)*(ErrorTiempo1**2)) + ((derivadaVelocidadAngular4**2)*(ErrorTiempo1**2)))

print(medicionIndirectaVelocidadAngular())

## RETORNA 0.39317426161945046

############################# ERROR DE ACELERACION ANGULAR #############################

#Calculo la aceleracion angular como la diferencia de velocidades angulares entre dos puntos en un tiempo determinado Wi = (wi2 - wi1) / (t2 - t1)

Promedio_Wi1 = 0.
ErrorWi = 0.39317426161945046
Promedio_Wi2 = 0.1

Promedio_tiempo1 = 3.56
Promedio_tiempo2 = 3.59
ErrorTiempo1 = 0.03

#Derivada 1
def derivadaAceleracionAngular_1 (wi1,wi2,t1,t2):
    return (1/(t2-t1))

#Derivada 2
def derivadaAceleracionAngular_2 (wi1,wi2,t1,t2):
    return -1/(t2-t1)

#Derivada 3
def derivadaAceleracionAngular_3 (wi1,wi2,t1,t2):
    return (wi2-wi1)/((t2-t1)**2)

#Derivada 4
def derivadaAceleracionAngular_4 (wi1,wi2,t1,t2):
    return (-1*(wi2-wi1))/((t2-t1)**2)

def medicionIndirectaAceleracionAngular():
    derivadaAceleracionAngular1 = derivadaAceleracionAngular_1(Promedio_Wi1,Promedio_Wi2,Promedio_tiempo1,Promedio_tiempo2)
    derivadaAceleracionAngular2 = derivadaAceleracionAngular_2(Promedio_Wi1,Promedio_Wi2,Promedio_tiempo1,Promedio_tiempo2)
    derivadaAceleracionAngular3 = derivadaAceleracionAngular_3(Promedio_Wi1,Promedio_Wi2,Promedio_tiempo1,Promedio_tiempo2)
    derivadaAceleracionAngular4 = derivadaAceleracionAngular_4(Promedio_Wi1,Promedio_Wi2,Promedio_tiempo1,Promedio_tiempo2)
    print(derivadaAceleracionAngular1**2*(ErrorWi**2))
    print(derivadaAceleracionAngular2**2*(ErrorWi**2))
    print(derivadaAceleracionAngular3**2*(ErrorTiempo1**2))
    print(derivadaAceleracionAngular4**2*(ErrorTiempo1**2))


    return ((derivadaAceleracionAngular1**2)*(ErrorWi**2)) + ((derivadaAceleracionAngular2**2)*(ErrorWi**2)) + ((derivadaAceleracionAngular3**2)*(ErrorTiempo1**2)) + ((derivadaAceleracionAngular4**2)*(ErrorTiempo1**2))
            
print(medicionIndirectaAceleracionAngular())