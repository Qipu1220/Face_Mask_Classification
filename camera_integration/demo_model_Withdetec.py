#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import joblib
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog
import os

# --- General Configuration ---
IMAGE_SIZE = (128, 128)
HOG_PARAMS = {
    'orientations':    9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm':      'L2-Hys'
}

# Labels for the classification model
LABELS = ['WithMask', 'WithoutMask']

# --- Function to extract HOG features from a grayscale image ---
def extract_hog_features(gray_img: np.ndarray) -> np.ndarray:
    img_resized = resize(gray_img, IMAGE_SIZE, anti_aliasing=True)
    return hog(img_resized, **HOG_PARAMS).reshape(1, -1)

# --- Step 1: Download and load YOLOv8-Face detector ---
repo_id    = "arnabdhar/YOLOv8-Face-Detection"
filename   = "model.pt"
model_path = hf_hub_download(repo_id=repo_id, filename=filename)
face_detector = YOLO(model_path)

# --- Step 2: Load your HOG+RandomForest model ---
clf_path = 'models\HOG_RandomForest_8x2.joblib'
if not os.path.exists(clf_path):
    raise FileNotFoundError(f"Classification model file not found at: {clf_path}")

data = joblib.load(clf_path)
clf = data['model']
le  = data['label_encoder']

# --- Step 3: Open the camera and process each frame ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Unable to open the camera. Please check the connection.")

print("Running Real-Time Face Detection + Mask Classification. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference face detection (YOLOv8-Face)
    results = face_detector(frame)[0]  
    detections = Detections.from_ultralytics(results)

    # Với mỗi face detection, crop ROI và phân loại
    for box, score in zip(detections.xyxy, detections.confidence):
        x1, y1, x2, y2 = map(int, box)
        face_roi_color = frame[y1:y2, x1:x2]
        if face_roi_color.size == 0:
            continue

        # Chuyển ROI sang grayscale
        face_gray = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2GRAY)

        # Trích xuất HOG và phân loại
        feat = extract_hog_features(face_gray)
        pred_idx = clf.predict(feat)[0]
        prob     = np.max(clf.predict_proba(feat))
        label    = le.inverse_transform([int(pred_idx)])[0]

        # Vẽ bounding box và nhãn
        color = (0,255,0) if label == 'WithMask' else (0,0,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label}: {prob*100:.1f}%"
        cv2.putText(frame, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Hiển thị kết quả
    cv2.imshow('Face Detection + Mask Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
