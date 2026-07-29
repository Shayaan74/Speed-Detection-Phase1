from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="Vehicle Detection.yolo26/data.yaml",      # Path to your data.yaml
        epochs=50,             # Number of training epochs
        imgsz=640,             # Image size
        batch=16,              # Batch size
        name="vehicle_yolo_v1",
        device=0,              # use RTX 4080
        workers=0,             # <- important on Windows to avoid spawn error
    )

if __name__ == "__main__":
    main()
