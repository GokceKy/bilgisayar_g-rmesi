import cv2
import numpy as np
import matplotlib.pyplot as plt

#  Görüntüyü yükle
image_path = 'C:\Users\Gokce\Pictures\wallpaper\solo.jpg'
image = cv2.imread(image_path)

# Orijinal görüntü-> RGB -
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#  Ortalama filtre 
# 5x5 kernel boyutunda ortalama filtre uyguluyoruz
blurred_image = cv2.blur(image_rgb, (5, 5))


# Laplace filtresi 
laplace_image = cv2.Laplacian(blurred_image, cv2.CV_64F)


# görüntüyü yan yana koymak için 1 satır ve 3 sütun olacak şekilde bir figür 
plt.figure(figsize=(15, 5))  # Pencere boyutu

#  Orijinal görüntü
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Orijinal Görüntü")
plt.axis('off')  

#  Ortalama filtre
plt.subplot(1, 3, 2)
plt.imshow(blurred_image)
plt.title("Ortalama Filtrelenmiş")
plt.axis('off')

#  Laplace filtre
plt.subplot(1, 3, 3)
plt.imshow(laplace_image, cmap='gray')
plt.title("Laplace Filtrelenmiş")
plt.axis('off')

#  Sonuç
plt.tight_layout() 
plt.show()


