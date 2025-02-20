import os
from ultralytics import YOLO

if __name__ == '__main__':
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # Using pre-trained YOLOv8n model

    # Train the model on your dataset with improved parameters
    results = model.train(
        data="D:/L and T/data.yaml",
        epochs=50,  # Increase the number of epochs
        imgsz=416,  # Increase the image size
        batch=16,  # Increase the batch size
        device=0,  # Use GPU (device=0)
        lr0=0.01,  # Adjust the learning rate
        augment=True  # Enable data augmentation
    )

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Save trained model
    model.save("models/best_2.pt")

    # Save training results to a log file
    with open("training_results.log", "w") as f:
        f.write("Training Complete! Model saved at models/best.pt\n")
        f.write("Training Results:\n")
        f.write(f"Final Epoch: {results.epoch}\n")
        f.write(f"Training Loss: {results.loss}\n")
        f.write(f"Validation Loss: {results.val_loss}\n")
        f.write(f"mAP@0.5: {results.metrics['mAP_0.5']}\n")
        f.write(f"mAP@0.5:0.95: {results.metrics['mAP_0.5:0.95']}\n")
        f.write(f"Precision: {results.metrics['precision']}\n")
        f.write(f"Recall: {results.metrics['recall']}\n")

    print("Training Complete! Model saved at models/best.pt")
    print("Training results have been saved to training_results.log")
