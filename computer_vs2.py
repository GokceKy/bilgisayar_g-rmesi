import cv2
import numpy as np

#  Webcam başlat
cap = cv2.VideoCapture(0)


while True:
    # Webcam
    ret, frame = cap.read()

    
    if not ret:
        break

    # Orijinal görüntü -> RGB 
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #  Ortalama filtre
    blurred_image = cv2.blur(image_rgb, (5, 5))

    #  Laplace filtre
    laplace_image = cv2.Laplacian(blurred_image, cv2.CV_64F)
    laplace_image = cv2.convertScaleAbs(laplace_image)  # Laplace görüntüsünü normalize et

    
    combined_image = np.hstack((image_rgb, blurred_image, laplace_image))

    #  Görüntüler
    cv2.imshow("Orijinal, Ortalama ve Laplace Filtreleri", combined_image)

    # 7. 'ESC' veya 'Backspace' 
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == 8:  # 27: ESC, 8: Backspace
        break


cap.release()
cv2.destroyAllWindows()
