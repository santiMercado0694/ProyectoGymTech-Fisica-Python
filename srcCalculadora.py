# -- LIBRERIAS QUE UTILIZAMOS --

import math
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from scipy.ndimage import uniform_filter1d 
from scipy.signal import savgol_filter

# -- CONSTANTES --

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
video_ready_callback = None
LARGO_ANTEBRAZO = 0.30
MASA_ANTEBRAZO = 1.8
RADIO_BICEP = 0.06
GRAVEDAD = 9.81

# -- FUNCIONES AXULIARES --

# Funcion que calcula el angulo_entre_vectores que aplica a = arcos ( (V1 * V2) / ( ||V1|| * ||V2|| ) )
def calcular_angulo(vector1, vector2):
    producto_punto = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitud1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitud2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    cos_theta = producto_punto / (magnitud1 * magnitud2)
    angulo_rad = math.acos(cos_theta)
    return angulo_rad

# Funcion que calcula el angulo en rads(radiantes), en base a tres puntos (x,y) siendo el 1ro la interseccion
def angulo_entre_vectores(codo_pos, muneca_pos, hombro_pos):
    vector_codo_muneca = (muneca_pos[0] - codo_pos[0], muneca_pos[1] - codo_pos[1])
    vector_codo_hombro = (hombro_pos[0] - codo_pos[0], hombro_pos[1] - codo_pos[1])
    angulo = calcular_angulo(vector_codo_muneca, vector_codo_hombro)
    return angulo

# Fncion que calcular la fuerza que ejerce el bicep
# Calcular inercias
# La fórmula anterior es esta inercia_pesa = (masa_pesa * LARGO_ANTEBRAZO**2) / 12,
# la cual es con el eje de rotacion (codo) ubicado en el centro de la varilla, no en un extremo
# La cambie por la siguiente:
# A= Inercia varilla delgada con el eje de rotacion en un extremo (codo)= (1/3) * MASA_ANTEBRAZO * LARGO_ANTEBRAZO ** 2
# B= Partícula de masa M a una distancia R del eje de rotación = M * R**2= MASA_PESA * LARGO_ANTEBRAZO ** 2
# Luego Inercia_total = A+B = ((MASA_ANTEBRAZO/3) + masa_pesa) * (LARGO_ANTEBRAZO ** 2)
def calcularFuerzaBicep(dataframe, masa_pesa):
    inercia_total = ((MASA_ANTEBRAZO / 3) + masa_pesa) * (LARGO_ANTEBRAZO**2)

    # Calcular suma de momentos
    dataframe["suma_momentos"] = inercia_total * dataframe["Aceleracion_angular"]
    dataframe["Fuerza_bicep"] = abs(
        -((dataframe["suma_momentos"] - dataframe["Momento_pesa"]) / (RADIO_BICEP))
    )



# Funcion que reliza el suavizado de los graficos
def suavizar_dataframe(df, max_window_length=5, polyorder=2):
    columnas_numericas = df.select_dtypes(include=[float, int]).columns
    for columna in columnas_numericas:
        datos_columna = df[columna].fillna(0)
        # Asegurarse de que window_length no sea mayor que la longitud de la columna y sea un número impar
        window_length = min(
            max_window_length,
            (
                len(datos_columna)
                if len(datos_columna) % 2 != 0
                else len(datos_columna) - 1
            ),
        )
        if window_length < polyorder + 2:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        # Aplicar el filtro de Savitzky-Golay solo si la columna tiene datos suficientes
        if len(datos_columna) > window_length:
            df[columna] = savgol_filter(datos_columna, window_length, polyorder)
    return df

# Funcion para dibujar los landmarks de la pose en la imagen
def dibujar_landarmks(image, results):
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )

# Funcion que calcula el momento de la pesa y se lo agrega al CSV
def calcular_momento_pesa(pose_row_cartesian, pesa_mancuerna, results):
    momento_de_la_pesa = (
        LARGO_ANTEBRAZO
        * pesa_mancuerna
        * GRAVEDAD
        * math.sin(
            angulo_entre_vectores(
                (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                ),
                (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                ),
                (
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                    1,
                ),
            )
        )   
    )
    pose_row_cartesian["Momento_pesa"] = momento_de_la_pesa

#Funcion para mostrar los datos en video
def mostrar_datos_en_video(
    FRAME_NUMERO,
    CAP,
    imagen,
    tiempo_en_segundos,
    repeticiones_ejercicio,
    calorias_quemadas_ej,
):
    cv2.putText(
        imagen,
        f"Frame: {FRAME_NUMERO}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        12,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        f"Segundo: {tiempo_en_segundos:.2f}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        12,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        f"Repeticiones: {repeticiones_ejercicio}",
        (50, int(CAP.get(4)) - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        12,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        f"Calorias quemadas (Kcal): {calorias_quemadas_ej}",
        (50, int(CAP.get(4)) - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        12,
        cv2.LINE_AA,
    )


    # Datos en video
    cv2.putText(
        imagen,
        f"Frame: {FRAME_NUMERO}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        f"Segundo: {tiempo_en_segundos:.2f}",
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        f"Repeticiones: {repeticiones_ejercicio}",
        (50, int(CAP.get(4)) - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        imagen,
        f"Calorias quemadas (Kcal): {calorias_quemadas_ej}",
        (50, int(CAP.get(4)) - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )


# Funcion para recolectar los datos de la pose (coordenadas cartesianas) en el DataFrame
def recolectar_datos_de_la_pose(
    pose_row_cartesian,
    landmarks_of_interest,
    results,
    previous_wrist_x,
    previous_wrist_y,
):
    elbow_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
    elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y

    # Obtener las coordenadas de la muñeca sin modificar
    pose_row_cartesian["Left_Wrist_x(m)_Sin_Modificar"] = (
        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
    )
    pose_row_cartesian["Left_Wrist_y(m)_Sin_Modificar"] = (
        (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y) * -1
    )

    for landmark in landmarks_of_interest:
        # Calcular las coordenadas relativas al codo y pasarlas a metros
        rel_x = (results.pose_landmarks.landmark[landmark].x - elbow_x) * 1.58
        rel_y = (results.pose_landmarks.landmark[landmark].y - elbow_y) * -1.58

        pose_row_cartesian[landmark.name + "_x(m)"] = rel_x
        pose_row_cartesian[landmark.name + "_y(m)"] = rel_y

    # Obtener las coordenadas relativas de la muñeca
    wrist_x = pose_row_cartesian["LEFT_WRIST_x(m)"]
    wrist_y = pose_row_cartesian["LEFT_WRIST_y(m)"]

    # Se calcula y agrega al csv el angulo del brazo en ese frame
    pose_row_cartesian["Angulo"] = angulo_entre_vectores(
        (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
        ),
        (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
        ),
        (
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
        ),
    )

    # Calcular la distancia recorrida por la pesa (muñeca) desde el frame anterior
    if previous_wrist_x is not None and previous_wrist_y is not None:
        distancia_recorrida = math.sqrt(
            (wrist_x - previous_wrist_x) ** 2 + (wrist_y - previous_wrist_y) ** 2
        )
        pose_row_cartesian["Distancia_recorrida(m)"] = distancia_recorrida
    else:
        pose_row_cartesian["Distancia_recorrida(m)"] = 0

    previous_wrist_x = wrist_x
    previous_wrist_y = wrist_y

    return previous_wrist_x, previous_wrist_y

# Funcion para recolectar los datos de la pose en caso de que no haya ningun resultado
def recolectar_datos_de_la_pose_no_results(pose_row_cartesian, landmarks_of_interest) :
    for landmark in landmarks_of_interest:
        pose_row_cartesian[landmark.name + "_x(m)"] = None
        pose_row_cartesian[landmark.name + "_y(m)"] = None
    pose_row_cartesian["Angulo"] = None
    pose_row_cartesian["Velocidad_angular"] = None
    pose_row_cartesian["Momento_pesa"] = None
    pose_row_cartesian["Distancia_recorrida(m)"] = None

# Funcion para calcular la cantidad de repeticiones que hace el usuario
def calcular_repeticiones(previous_Y, pose_row_cartesian, repeticiones):
    if previous_Y is not None:
        current_Y = pose_row_cartesian[mp_pose.PoseLandmark.LEFT_WRIST.name + "_y(m)"]
        if current_Y is not None and previous_Y is not None:
            if previous_Y < 0 and current_Y > 0:
                repeticiones += 1
            previous_Y = current_Y
    else:
        previous_Y = pose_row_cartesian[mp_pose.PoseLandmark.LEFT_WRIST.name + "_y(m)"]

    return repeticiones, previous_Y

# Funcion para crear el DataFrame para almacenar los datos de la pose (coordenadas cartesianas)
def crear_dataframe(landmarks_of_interest):
    columns_cartesian = ["frame_number", "tiempo(seg)", "repeticion"]
    for landmark in landmarks_of_interest:
        columns_cartesian.append(landmark.name + "_x(m)")
        columns_cartesian.append(landmark.name + "_y(m)")
    columns_cartesian.append("Angulo")
    columns_cartesian.append("Velocidad_angular")
    columns_cartesian.append("Momento_pesa")
    columns_cartesian.append("Distancia_recorrida(m)")
    pose_data_cartesian = pd.DataFrame(columns=columns_cartesian)

    return pose_data_cartesian

# Funcion para calcular la cantidad de calorias quemadas a lo largo del ejercicio de biceps
def calcular_calorias(calorias_quemadas, pose_data_cartesian):
    calorias_quemadas = round(
        pose_data_cartesian["Trabajo_bicep_positivo"].sum() / 4184, 2
    )

    return calorias_quemadas

# Funcion para cargar los datos del trackeo al CSV
def cargar_datos_al_csv(pose_data_cartesian, masa_pesa):
    pose_data_cartesian["dif_angular"] = pose_data_cartesian["Angulo"].diff()
    pose_data_cartesian["dif_temporal"] = pose_data_cartesian["tiempo(seg)"].diff()
    pose_data_cartesian["Velocidad_angular"] = abs(
        pose_data_cartesian["dif_angular"] / pose_data_cartesian["dif_temporal"]
    )
    pose_data_cartesian["dif_velocidad_angular"] = pose_data_cartesian[
        "Velocidad_angular"
    ].diff()
    pose_data_cartesian["Aceleracion_angular"] = abs(
        pose_data_cartesian["dif_velocidad_angular"]
        / pose_data_cartesian["dif_temporal"]
    )
    pose_data_cartesian["dif_x"] = pose_data_cartesian["LEFT_WRIST_x(m)"].diff()
    pose_data_cartesian["dif_y"] = pose_data_cartesian["LEFT_WRIST_y(m)"].diff()
    pose_data_cartesian["velocidad_munieca"] = (
        np.sqrt(
            pose_data_cartesian["dif_x"] ** 2 + pose_data_cartesian["dif_y"] ** 2
        )
        / pose_data_cartesian["dif_temporal"]
    )
    pose_data_cartesian["Energia_cinetica"] = (0.5 * masa_pesa) * (
        (pose_data_cartesian["velocidad_munieca"]) ** 2
    )
    
    # Encuentra el primer valor válido en la columna
    Y_inicial = pose_data_cartesian["Left_Wrist_y(m)_Sin_Modificar"].dropna().iloc[0]

    # Calcula la energía potencial restando el primer valor válido
    pose_data_cartesian["Energia_potencial"] = (
    masa_pesa
    * 9.8
    * (
        pose_data_cartesian["Left_Wrist_y(m)_Sin_Modificar"] - Y_inicial
      )
    )

    calcularFuerzaBicep(pose_data_cartesian, masa_pesa)
    pose_data_cartesian["Energia_Mecanica"] = (
        pose_data_cartesian["Energia_cinetica"]
        + pose_data_cartesian["Energia_potencial"]
    )

    pose_data_cartesian["Trabajo_bicep"] = pose_data_cartesian[
        "Energia_Mecanica"
    ].diff()

    pose_data_cartesian["Trabajo_bicep_positivo"] = pose_data_cartesian[
        "Trabajo_bicep"
    ].apply(lambda x: x if x > 0 else -x)
    


# Funcion Principal
def track_pose(video_path, peso_mancuerna):
    VIDEO_PATH = video_path
    OUTPUT_VIDEO_PATH = "resultados/video/tracked_video.mp4"
    OUTPUT_CSV_PATH = "resultados/documents/data.csv"
    FPS = 30
    FRAME_NUMBER = 0
    repeticiones = 0
    calorias_quemadas = 0
    previous_Y = None
    previous_wrist_x = None
    previous_wrist_y = None
    masa_pesa = peso_mancuerna
    cap = cv2.VideoCapture(VIDEO_PATH)

    # Definir los landmarks de interés
    landmarks_of_interest = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
    ]

    # Crear el DataFrame para almacenar los datos de la pose (coordenadas cartesianas)
    pose_data_cartesian = crear_dataframe(landmarks_of_interest)

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (int(cap.get(3)), int(cap.get(4))),
    )

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("No se pudo leer el video.")
            break

        # Convertir la imagen a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Procesar la imagen con MediaPipe
        results = pose.process(image)

        # Dibujar los landmarks de la pose en la imagen
        dibujar_landarmks(image, results)

        tiempo_segundos = FRAME_NUMBER / FPS
        pose_row_cartesian = {
            "frame_number": FRAME_NUMBER,
            "tiempo(seg)": tiempo_segundos,
            "repeticion": repeticiones,
        }

        # Contorno de los datos en video
        mostrar_datos_en_video(
            FRAME_NUMBER,
            cap,
            image,
            tiempo_segundos,
            repeticiones,
            calorias_quemadas
        )

        # Guardar el cuadro procesado en el video de salida
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Recolectar los datos de la pose (coordenadas cartesianas) en el DataFrame
        if results.pose_landmarks:

            previous_wrist_x, previous_wrist_y = recolectar_datos_de_la_pose(
                pose_row_cartesian,
                landmarks_of_interest,
                results,
                previous_wrist_x,
                previous_wrist_y,
            )

            # Se calcula el momento de la pesa y se lo agrega al csv
            
            calcular_momento_pesa(pose_row_cartesian, masa_pesa, results)

        else:
            recolectar_datos_de_la_pose_no_results(pose_row_cartesian, landmarks_of_interest)

        # Contador de repeticiones
        repeticiones, previous_Y = calcular_repeticiones(
            previous_Y, pose_row_cartesian, repeticiones
        )

        #Cargar los datos del trackeo al CSV
        pose_data_cartesian = pd.concat(
            [pose_data_cartesian, pd.DataFrame([pose_row_cartesian])], ignore_index=True
        )

        cargar_datos_al_csv(pose_data_cartesian, masa_pesa)

        #Actualizo el Frame del video
        FRAME_NUMBER += 1

        # Calcular las calorias quemadas
        calorias_quemadas = calcular_calorias(calorias_quemadas, pose_data_cartesian)

        #Suavizo los graficos
        pose_data_cartesian = suavizar_dataframe(pose_data_cartesian, max_window_length=5, polyorder=2)

    pose.close()
    video_writer.release()
    cap.release()

    # Guardar los DataFrames como archivos CSV
    pose_data_cartesian.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Proceso completado. Video trackeado guardado en:", OUTPUT_VIDEO_PATH)
    print("Datos de la pose (cartesianas) guardados en:", OUTPUT_CSV_PATH)
    if video_ready_callback:
        video_ready_callback()
