import cv2
import os
import numpy as np
from datetime import datetime
from twilio.rest import Client

# Twilio credentials
TWILIO_PHONE_NUMBER = "+13344182669"
TWILIO_ACCOUNT_SID = "AC62b536493ab06afc3918ed76bb4b14dd"
TWILIO_AUTH_TOKEN = "a0529ee9970ede763689258ce51bd44d"

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load Haar cascade for face detection
xml_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(xml_path)

if face_cascade.empty():
    print("Error: Could not load Haar Cascade file.")
    exit()

# Check if OpenCV has the 'face' module (needed for face recognition)
if not hasattr(cv2, 'face'):
    print("Error: OpenCV does not have the face module. Please install opencv-contrib-python.")
    exit()

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Function to send SMS
def send_sms(message, recipients):
    for number in recipients:
        try:
            msg = client.messages.create(
                from_=TWILIO_PHONE_NUMBER,
                to=number,
                body=message
            )
            print(f"Message sent to {number}: {msg.sid}")
        except Exception as e:
            print(f"Failed to send message to {number}: {e}")

# Function to train the recognizer using a predefined authorized face image
def train_recognizer():
    faces = []
    labels = []
    
    # Add the authorized person's face image to the training dataset
    label = "Authorized_Person"  # Label for your face
    
    # Path to the authorized person's image
    img_path = 'dataset/authorized_person.jpg'  # Make sure this image exists
    if not os.path.exists(img_path):
        print(f"Error: {img_path} does not exist.")
        exit()
        
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_in_img = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    
    for (x, y, w, h) in faces_in_img:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))  # Resize to a fixed size (100x100)
        faces.append(face_resized)
        labels.append(0)  # Only one label for the authorized person (0)
    
    faces = np.array(faces)
    labels = np.array(labels)

    if len(faces) > 0 and len(labels) > 0:
        recognizer.train(faces, labels)
        recognizer.save('trainer.yml')
        print("Model trained and saved as trainer.yml.")
    else:
        print("No faces found for training.")

# Train the recognizer using the predefined image
train_recognizer()

# Start video capture for real-time face detection
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

        for x, y, w, h in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            face_img = gray[y:y+h, x:x+w]

            # Recognize the face using the trained recognizer
            label, confidence = recognizer.predict(face_img)

            # If confidence is above a certain threshold (e.g., 100), it is an unknown face
            if confidence > 100:  # Confidence threshold for unknown face
                exact_time = datetime.now().strftime('%Y-%b-%d-%H-%S-%f')
                image_path = f"face_detected_{exact_time}.jpg"
                cv2.imwrite(image_path, img)

                # Send SMS notification for unknown face
                message = f"Alert: Unknown person entered your home. Face detected at {exact_time}."
                recipients = ["+919492592373"]  # Replace with recipient numbers
                send_sms(message, recipients)

        cv2.imshow("Home Surveillance", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

video.release()
cv2.destroyAllWindows()