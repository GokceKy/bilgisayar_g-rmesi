import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO

# Modeli yükle
model = YOLO("yolov8n.pt")  # YOLOv8 model dosyasının bulunduğu yol

# Belirtilen URL'ler
urls = [
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/25/671bf82ac8a50557.jpg",
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/25/671bedbd7e0e2430.jpg",
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/25/671bf633e1ac5722.jpg",
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/25/671b30a7154ef988.jpg",
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/25/671ba31489aa6724.jpg",
    "https://icdn.ensonhaber.com/crop/1200x675/resimler/diger/kok/2024/10/21/67161b44668b4105.jpg",
    
]

# Her URL için görüntüyü indir ve nesne tespiti yap
for url in urls:
    # Görüntüyü indir
    resp = urllib.request.urlopen(url)
    image_array = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Resmi tahmin et
    results = model(image)

    # Sonuçları işleme
    for result in results:
        # Tahmin edilen sınıflar
        for box in result.boxes:
            class_id = int(box.cls)  # Sınıf ID'si
            confidence = box.conf.item()  # Güven oranını float'a çevir
            label = model.names[class_id]  # Sınıf adı

            if confidence > 0.5:  # Güven eşik değeri
                # Nesne konumunu al
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Tespit edilen kutunun koordinatları

                # Nesne etiketini ve güven oranını yazdır
                text = f"{label} {confidence:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Tespit edilen kutuyu çiz
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Metni yaz

    # Sonuçları görüntüle
    cv2.imshow("Image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
