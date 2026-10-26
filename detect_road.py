
from ultralytics import YOLO

model = YOLO(r"D:\RoadDamageDetection\Models\best.pt")

image_path = r"D:\RoadDamageDetection\road.jpg"

results = model.predict(
    source=image_path,
    conf=0.25,
    imgsz=640,
    save=True,
    verbose=True
)

for result in results:
    print("\nDetected objects:", len(result.boxes))

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        print(
            f"Class ID: {class_id}, "
            f"Confidence: {confidence:.3f}"
        )

print("\nCheck the saved prediction image.")
