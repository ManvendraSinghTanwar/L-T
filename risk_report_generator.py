import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load trained models
yolo_model = YOLO("models/best_2.pt")  # YOLO for safety violations
best_model = XGBClassifier()  # XGBoost model (replace with trained model)
best_model.load_model("best_xgb_model.json")  # Load trained XGBoost model

# Load test dataset
df_test = pd.read_csv("test_construction_delay.csv")  # Test dataset

# Function to extract safety violations from images
def extract_risk_features(image_path):
    results = yolo_model(image_path)
    detections = results[0].boxes.cls.cpu().numpy()  # Extract class IDs
    hazard_counts = {cls: (detections == cls).sum() for cls in range(len(results[0].names))}
    return hazard_counts

# Function to generate risk report
def generate_risk_report(project_id, image_path, project_data):
    print("\n" + "="*50)
    print(f"📌 **Risk Report for Project ID: {project_id}**")
    print("="*50)

    # Extract safety hazards from image
    hazard_counts = extract_risk_features(image_path)
    print("⚠ **Safety Hazards Detected:**")
    for hazard, count in hazard_counts.items():
        if count > 0:
            print(f"   - {hazard}: {count} violations")

    # Prepare input for delay prediction
    project_data = project_data.drop(columns=["Project ID", "Start Date", "End Date"], errors='ignore')
    delay_prediction = best_model.predict(project_data)[0]
    delay_probability = best_model.predict_proba(project_data)[0][1]  # Probability of delay

    print("\n🔍 **Project Delay Prediction:**")
    if delay_prediction == 1:
        print(f"   ❌ **Prediction: Delayed** (⏳ Probability: {delay_probability*100:.2f}%)")
    else:
        print(f"   ✅ **Prediction: On Time** (🟢 Probability: {(1 - delay_probability)*100:.2f}%)")

    print("\n📊 **Key Contributing Factors:**")
    feature_importance = best_model.get_booster().get_score(importance_type='weight')
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:5]:  # Top 5 most important features
        print(f"   - {feature}: Importance Score = {importance}")

    print("="*50 + "\n")

# Iterate through test projects and generate reports
for i, row in df_test.iterrows():
    project_id = row["Project ID"]
    image_path = f"test.jpg"  # Assuming images are stored by Project ID
    project_data = row.to_frame().T  # Convert row to DataFrame
    generate_risk_report(project_id, image_path, project_data)
