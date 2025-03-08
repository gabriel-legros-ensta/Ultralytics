from PIL import Image
import glob
import zipfile
import os
import cv2

"""# **ULTRALYTICS**"""

import ultralytics
from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Open video
video_path = 'basket.mp4' 
cap = cv2.VideoCapture(video_path)

# Load a model
#model = YOLO("yolo11n.pt").to(device)
print(f"Modèle chargé sur : {device}")

"""# **Entrainement**"""

#model.train(data="dataset3/data.yaml", epochs=200, cache=True, device="cuda", batch=32, imgsz=640, patience=20)
#device="cuda"

"""# **Validation**"""

# Charger le modèle entraîné
model = YOLO('runs/detect/train5/weights/best.pt').to(device)
print(f"Modèle chargé sur : {device}")

# Effectuer l'évaluation sur le jeu de validation
#metrics = model.val(data="dataset/dataset/data.yaml", imgsz=640, batch=32, device=device)

# Afficher les résultats des métriques
#print(metrics)

"""# **Fine-Tune**"""
#!yolo task=detect mode=train model='runs/detect/train2/weights/best.pt' data='dataset2/data.yaml' imgsz=640 epochs=100 batch=32 lr0=0.01 device="cuda" optimizer='Adam' cache_images=True patience=15

"""# **Video**"""
# Obtenir les informations de la vidéo (largeur, hauteur, FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Créer un objet VideoWriter pour enregistrer la vidéo de sortie
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convertir l'image en format RGB pour YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Faire une prédiction avec YOLO
    results = model(frame_rgb)

    # Afficher les résultats sur l'image
    annotated_frame = results[0].plot()  # Dessine les boîtes de détection sur l'image

    # Convertir l'image annotée en format BGR pour OpenCV
    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Enregistrer la frame annotée
    out.write(annotated_frame_bgr)

    # Afficher la vidéo (optionnel)
    #cv2.imshow('Frame', annotated_frame_bgr)


# Libérer les ressources
cap.release()
out.release()
#cv2.destroyAllWindows()
