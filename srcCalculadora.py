import glob
import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
video_ready_callback = None

def track_pose(video_path):

    VIDEO_PATH = video_path
    OUTPUT_VIDEO_PATH = 'resultados/video/tracked_video.mp4'
    OUTPUT_CSV_PATH = 'resultados/documents/data.csv'
    FPS = 30

    cap = cv2.VideoCapture(VIDEO_PATH)

    # Definir los landmarks de interés
    landmarks_of_interest = [mp_pose.PoseLandmark.LEFT_SHOULDER,
                            mp_pose.PoseLandmark.LEFT_ELBOW,
                            mp_pose.PoseLandmark.LEFT_WRIST]

    # Crear el DataFrame para almacenar los datos de la pose (coordenadas cartesianas)
    columns_cartesian = ['frame_number', 'tiempo(seg)']
    for landmark in landmarks_of_interest:
        columns_cartesian.append(landmark.name + '_x(m)')
        columns_cartesian.append(landmark.name + '_y(m)')
    columns_cartesian.append("Angulo")
    pose_data_cartesian = pd.DataFrame(columns=columns_cartesian)

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FPS,
                                    (int(cap.get(3)), int(cap.get(4))))

    FRAME_NUMBER = 0

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
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                )
        
        # Dibujar el número de frame y segundo (Para relacionar con precisión los datos obtenidos con el momento exacto del video)
        tiempo_segundos = FRAME_NUMBER / FPS
        cv2.putText(image, f'Frame: {FRAME_NUMBER}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Segundo: {tiempo_segundos:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # Guardar el cuadro procesado en el video de salida
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Recolectar los datos de la pose (coordenadas cartesianas) en el DataFrame
        pose_row_cartesian = {'frame_number': FRAME_NUMBER, 'tiempo(seg)': tiempo_segundos}
        if results.pose_landmarks:
            # Obtener las coordenadas del codo
            elbow_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
            elbow_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y

            for landmark in landmarks_of_interest:
                # Calcular las coordenadas relativas al codo y pasarlas a metros
                rel_x = (results.pose_landmarks.landmark[landmark].x - elbow_x) * 1.58
                rel_y = (results.pose_landmarks.landmark[landmark].y - elbow_y) * -1.58

                pose_row_cartesian[landmark.name + '_x(m)'] = rel_x
                pose_row_cartesian[landmark.name + '_y(m)'] = rel_y

            pose_row_cartesian["Angulo"] = angulo_entre_vectores((results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y),
                                                                 (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y),
                                                                 (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y))
        else:
            for landmark in landmarks_of_interest:
                pose_row_cartesian[landmark.name + '_x(m)'] = None
                pose_row_cartesian[landmark.name + '_y(m)'] = None
            pose_row_cartesian["Angulo"]

        pose_data_cartesian = pd.concat([pose_data_cartesian, pd.DataFrame([pose_row_cartesian])], ignore_index=True)

        FRAME_NUMBER += 1

    pose.close()
    video_writer.release()
    cap.release()

    # Guardar los DataFrames como archivos CSV
    pose_data_cartesian.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Proceso completado. Video trackeado guardado en:", OUTPUT_VIDEO_PATH)
    print("Datos de la pose (cartesianas) guardados en:", OUTPUT_CSV_PATH)
    if video_ready_callback:
        video_ready_callback()

def calcular_angulo(vector1, vector2):
    producto_punto = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitud1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitud2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
    cos_theta = producto_punto / (magnitud1 * magnitud2)
    angulo_rad = math.acos(cos_theta)
    angulo_deg = math.degrees(angulo_rad)
    return angulo_deg

def angulo_entre_vectores(codo_pos, muneca_pos, hombro_pos):
    vector_codo_muneca = (muneca_pos[0] - codo_pos[0], muneca_pos[1] - codo_pos[1])
    vector_codo_hombro = (hombro_pos[0] - codo_pos[0], hombro_pos[1] - codo_pos[1])
    angulo = calcular_angulo(vector_codo_muneca, vector_codo_hombro)
    return angulo

