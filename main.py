from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=True)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)

# Set confidence threshold
confidence_threshold = 0.5  # 50%

while True:
    # Grab the webcam's image
    ret, image = camera.read()
    if not ret:
        break  # Exit if the camera fails

    # Resize the raw image into (224-height,224-width) pixels
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Make the image a numpy array and reshape it to the model's input shape
    normalized_image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
    normalized_image = (normalized_image / 127.5) - 1

    # Predict the model
    prediction = model.predict(normalized_image)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    # Display prediction and confidence only if above threshold
    if confidence_score >= confidence_threshold:
        class_name = class_names[index].strip()
        text = f"Class: {class_name[2:]} | Confidence: {confidence_score * 100:.2f}%"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Listen for 'Q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Check if 'Q' key is pressed
        break

# Release the camera and close the window
camera.release()
cv2.destroyAllWindows()



