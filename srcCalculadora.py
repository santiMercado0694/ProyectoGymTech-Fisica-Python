import math
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
from scipy.ndimage import uniform_filter1d  # Importar para suavizado

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
video_ready_callback = None
LARGO_ANTEBRAZO = 0.30
MASA_ANTEBRAZO = 1.8
RADIO_BICEP = 0.06
GRAVEDAD = 9.81

def track_pose(video_path, peso_mancuerna):
    VIDEO_PATH = video_path
    OUTPUT_VIDEO_PATH = "resultados/video/tracked_video.mp4"
    OUTPUT_CSV_PATH = "resultados/documents/data.csv"
    FPS = 30
    masa_pesa = peso_mancuerna

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Definir los landmarks de interés
    landmarks_of_interest = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
    ]

    # Crear el DataFrame para almacenar los datos de la pose (coordenadas cartesianas)
    columns_cartesian = ["frame_number", "tiempo(seg)", "repeticion"]
    for landmark in landmarks_of_interest:
        columns_cartesian.append(landmark.name + "_x(m)")
        columns_cartesian.append(landmark.name + "_y(m)")
    columns_cartesian.append("Angulo")
    columns_cartesian.append("Velocidad_angular")
    columns_cartesian.append("Momento_pesa")
    columns_cartesian.append("Distancia_recorrida(m)")
    pose_data_cartesian = pd.DataFrame(columns=columns_cartesian)

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (int(cap.get(3)), int(cap.get(4))),
    )

    FRAME_NUMBER = 0
    repeticiones = 0
    previous_Y = None
    previous_wrist_x = None
    previous_wrist_y = None

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
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        tiempo_segundos = FRAME_NUMBER / FPS

        # Contorno de los datos en video
        cv2.putText(
            image,
            f"Frame: {FRAME_NUMBER}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            12,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Segundo: {tiempo_segundos:.2f}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            12,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Repeticiones: {repeticiones}",
            (50, int(cap.get(4)) - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            12,
            cv2.LINE_AA,
        )

        # Datos en video
        cv2.putText(
            image,
            f"Frame: {FRAME_NUMBER}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Segundo: {tiempo_segundos:.2f}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Repeticiones: {repeticiones}",
            (50, int(cap.get(4)) - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        # Guardar el cuadro procesado en el video de salida
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Recolectar los datos de la pose (coordenadas cartesianas) en el DataFrame
        pose_row_cartesian = {
            "frame_number": FRAME_NUMBER,
            "tiempo(seg)": tiempo_segundos,
            "repeticion": repeticiones,
        }
        if results.pose_landmarks:
            # Obtener las coordenadas del codo
            elbow_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
            elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
            
            # Obtener las coordenadas de la muñeca sin modificar
            pose_row_cartesian["Left_Wrist_x(m)_Sin_Modificar"] = (
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
            )
            pose_row_cartesian["Left_Wrist_y(m)_Sin_Modificar"] = (
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
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
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].x,
                    results.pose_landmarks.landmark[
                        mp_pose.PoseLandmark.LEFT_SHOULDER
                    ].y,
                ),
            )

            # Se calcula el momento de la pesa y se lo agrega al csv
            momento_pesa = (
                LARGO_ANTEBRAZO
                * masa_pesa
                * GRAVEDAD
                * math.sin(
                    angulo_entre_vectores(
                        (
                            results.pose_landmarks.landmark[
                                mp_pose.PoseLandmark.LEFT_ELBOW
                            ].x,
                            results.pose_landmarks.landmark[
                                mp_pose.PoseLandmark.LEFT_ELBOW
                            ].y,
                        ),
                        (
                            results.pose_landmarks.landmark[
                                mp_pose.PoseLandmark.LEFT_WRIST
                            ].x,
                            results.pose_landmarks.landmark[
                                mp_pose.PoseLandmark.LEFT_WRIST
                            ].y,
                        ),
                        (
                            results.pose_landmarks.landmark[
                                mp_pose.PoseLandmark.LEFT_ELBOW
                            ].x,
                            1,
                        ),
                    )
                )
            )
            pose_row_cartesian["Momento_pesa"] = momento_pesa

            pose_row_cartesian["Contraccion_bicep"] = (
                calcular_tamano_bicep_con_contraccion(pose_row_cartesian["Angulo"])
            )
            
            # Calcular la distancia recorrida por la pesa (muñeca) desde el frame anterior
            if previous_wrist_x is not None and previous_wrist_y is not None:
                distancia_recorrida = math.sqrt(
                    (wrist_x - previous_wrist_x)**2 + (wrist_y - previous_wrist_y)**2
                )
                pose_row_cartesian["Distancia_recorrida(m)"] = distancia_recorrida
            else:
                pose_row_cartesian["Distancia_recorrida(m)"] = 0

            previous_wrist_x = wrist_x
            previous_wrist_y = wrist_y
            
        else:
            for landmark in landmarks_of_interest:
                pose_row_cartesian[landmark.name + "_x(m)"] = None
                pose_row_cartesian[landmark.name + "_y(m)"] = None
            pose_row_cartesian["Angulo"] = None
            pose_row_cartesian["Velocidad_angular"] = None
            pose_row_cartesian["Momento_pesa"] = None
            pose_row_cartesian["Distancia_recorrida(m)"] = None

        # Contador de repeticiones
        if previous_Y is not None:
            current_Y = pose_row_cartesian[
                mp_pose.PoseLandmark.LEFT_WRIST.name + "_y(m)"
            ]
            if current_Y is not None and previous_Y is not None:
                if previous_Y < 0 and current_Y > 0:
                    repeticiones += 1
                previous_Y = current_Y
        else:
            previous_Y = pose_row_cartesian[
                mp_pose.PoseLandmark.LEFT_WRIST.name + "_y(m)"
            ]

        pose_data_cartesian = pd.concat(
            [pose_data_cartesian, pd.DataFrame([pose_row_cartesian])], ignore_index=True
        )

        FRAME_NUMBER += 1

    pose.close()
    video_writer.release()
    cap.release()

    # Suavizar la columna "Angulo"
    pose_data_cartesian["Angulo"] = uniform_filter1d(pose_data_cartesian["Angulo"].dropna(), size=10)

    # Guardar los DataFrames como archivos CSV
    pose_data_cartesian.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Proceso completado. Video trackeado guardado en:", OUTPUT_VIDEO_PATH)
    print("Datos de la pose (cartesianas) guardados en:", OUTPUT_CSV_PATH)
    if video_ready_callback:
        video_ready_callback()


# Metodo auxiliar de angulo_entre_vectores que aplica a = arcos ( (V1 * V2) / ( ||V1|| * ||V2|| ) )
def calcular_angulo(vector1, vector2):
    producto_punto = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitud1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitud2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    cos_theta = producto_punto / (magnitud1 * magnitud2)
    angulo_rad = math.acos(cos_theta)
    return angulo_rad


# Se calcula el angulo en rads, en base a tres puntos (x,y) siendo el 1ro la interseccion
def angulo_entre_vectores(codo_pos, muneca_pos, hombro_pos):
    vector_codo_muneca = (muneca_pos[0] - codo_pos[0], muneca_pos[1] - codo_pos[1])
    vector_codo_hombro = (hombro_pos[0] - codo_pos[0], hombro_pos[1] - codo_pos[1])
    angulo = calcular_angulo(vector_codo_muneca, vector_codo_hombro)
    return angulo


def calcularFuerzaBicep(dataframe, masa_pesa):
    # Calcular inercias
    # La fórmula anterior es esta inercia_pesa = (masa_pesa * LARGO_ANTEBRAZO**2) / 12,
    # la cual es con el eje de rotacion (codo) ubicado en el centro de la varilla, no en un extremo
    # La cambie por la siguiente:
    # A= Inercia varilla delgada con el eje de rotacion en un extremo (codo)= (1/3) * MASA_ANTEBRAZO * LARGO_ANTEBRAZO ** 2
    # B= Partícula de masa M a una distancia R del eje de rotación = M * R**2= MASA_PESA * LARGO_ANTEBRAZO ** 2
    # Luego Inercia_total = A+B = ((MASA_ANTEBRAZO/3) + masa_pesa) * (LARGO_ANTEBRAZO ** 2)
    inercia_total = ((MASA_ANTEBRAZO/3) + masa_pesa) * (LARGO_ANTEBRAZO ** 2)
    
    # Calcular suma de momentos
    dataframe["suma_momentos"] = inercia_total * dataframe["Aceleracion_angular"]
    dataframe["Fuerza_bicep"] = abs(-(
        (
            dataframe["suma_momentos"]
            - dataframe["Momento_pesa"]
        )
        / (RADIO_BICEP)
    ))


def calcular_tamano_bicep_con_contraccion(angulo):

    # Estimamos la que contraccion del bicep como un %35 del largo del bicep, por lo tanto si tomamos un bicep de tamaño 0.3m,
    # la contraccion maxima seria de 0.105m (10.5cm) y la minima de 0m (completamente estirado)
    # Por lo tanto, la longitud del bicep en base al angulo se calcula como:
    # longitud_bicep_contraido + proporcion_contraccion * (longitud_bicep_estirado - longitud_bicep_contraido)

    # Longitud del bíceps cuando está completamente estirado y contraído
    longitud_bicep_estirado = LARGO_ANTEBRAZO  # en m
    longitud_bicep_contraido = 0.195  # en m
    # Supongamos que la contracción máxima ocurre a π radianes (180 grados)
    angulo_max_contraccion = np.pi  # radianes (180 grados)

    # Calcular la proporción de contracción basada en el ángulo
    proporcion_contraccion = angulo / angulo_max_contraccion

    # Calcular la longitud actual del bíceps
    longitud_actual = longitud_bicep_contraido + proporcion_contraccion * (
        longitud_bicep_estirado - longitud_bicep_contraido
    )

    return longitud_actual


# Se calcula el trabajo del bicep en base a la fuerza, la distancia recorrida y el angulo
# Dado que la velocidad es tangente a la trayectoria y la fuerza del bíceps es perpendicular al antebrazo, el ángulo
# es 0, luego cos(0)=1
def calcularTrabajoBicep(dataframe):
    dataframe["Trabajo_Fuerza_bicep"] = (
        dataframe["Fuerza_bicep"]
        * dataframe["Distancia_recorrida(m)"]
    )
