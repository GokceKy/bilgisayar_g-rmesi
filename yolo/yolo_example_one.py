from ultralytics import YOLO
import cv2
import requests
import numpy as np

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Function to perform live prediction and display results on the webcam
def predict_on_webcam(source=0):
    cap = cv2.VideoCapture(source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform prediction on the current frame
        results = model(frame)

        # Loop through each detection result and draw it on the frame
        for result in results:
            annotated_frame = result.plot()  # Annotate the frame with bounding boxes and labels

        # Display the annotated frame
        cv2.imshow("YOLO Webcam Detection", annotated_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("YOLO Webcam Detection", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the function to start webcam detection
predict_on_webcam()
def predict_on_image(source):
    # Check if the source is a URL
    if source.startswith("http"):
        response = requests.get(source)
        image = np.asarray(bytearray(response.content), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    else:
        # Load a local image file
        image = cv2.imread(source)
    
    # Perform prediction on the image
    results = model(image)
    
    # Display the annotated image with bounding boxes
    for result in results:
        annotated_image = result.plot()
        cv2.imshow("YOLO Image Detection", annotated_image)
        cv2.waitKey(0)  # Wait for a key press to close the image window

    cv2.destroyAllWindows()

predict_on_image("https://user-images.githubusercontent.com/54944384/78804250-7fb77f00-79f2-11ea-9fa1-8c7253eed09d.png")

# For a local disk image
predict_on_image(r"C:\Users\Gokce\Documents\bilgisayar_görmesi\yolo\people.jpg")