from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolov8n.pt")

# Resmi tahmin et
source="https://user-images.githubusercontent.com/54944384/78804250-7fb77f00-79f2-11ea-9fa1-8c7253eed09d.png"

# Resmi tahmin et
# results = model.predict(r"C:\Users\Gokce\Documents\bilgisayar_görmesi\yolo\people.jpg")

results = model(source)
for result in results:
    result.show()
