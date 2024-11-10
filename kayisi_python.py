# %%

import torch
from ultralytics import YOLO
from roboflow import Roboflow

# CUDA ve GPU kullanılabilirliğini kontrol et
# bilgisayarınızda komut satırında nvidia-smi yazdığınızda indirebileceğiniz en son cuda sürümü çıkıyor.
# kullandığınız python versiyonu ile uyumunu kontrol edip uygun olan cudayı indirebilirsiniz
print("CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())

from roboflow import Roboflow
rf = Roboflow(api_key="vMTpFkYXfkzkBVWou711")
project = rf.workspace("gokce").project("kayisi-iwsen")
version = project.version(2)
dataset = version.download("yolov8")
                

# %%
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli
if __name__ == "__main__" :
    model.train(data=f"{dataset.location}/data.yaml", epochs=10)

# %%
import os
from ultralytics import YOLO

# Kendi eğittiğiniz modeli yükleyin
model = YOLO('C:/Users/Gokce/runs/detect/train5/weights/best.pt')

# Test klasörünün yolunu belirleyin
test_images_folder = r"C:\Users\Gokce\Documents\bilgisayar_görmesi\kayisi-2\test\images"
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

# Test klasöründeki her bir görüntüde tahmin yap ve kaydet
for idx, image_file in enumerate(os.listdir(test_images_folder), start=1):
    image_path = os.path.join(test_images_folder, image_file)

    # Tahminleri yap
    results = model.predict(source=image_path, conf=0.45, iou=0.45)

    # Tahmin edilen görüntüyü kaydet
    results[0].save(os.path.join(results_folder, f"prediction_{idx}.jpg"))
    print(f"Processed {image_file} - Results saved as prediction_{idx}.jpg in 'results' folder")


