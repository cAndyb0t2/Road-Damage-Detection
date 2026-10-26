from ultralytics import YOLO

model_path = r"D:\RoadDamageDetection\Models\best.pt"

print("Loading trained YOLO model...")

model = YOLO(model_path)

print("\nSUCCESS! Model loaded.")
print("Task:", model.task)
print("Number of classes:", len(model.names))
print("Class names:", model.names)