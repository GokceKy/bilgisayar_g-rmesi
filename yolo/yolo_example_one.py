from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolov8n.pt")

# Resmi tahmin et
results = model.predict(r"C:\Users\Gokce\Documents\bilgisayar_görmesi\yolo\people.jpg")

# Sonuçları göster
for result in results:
    result.show()
