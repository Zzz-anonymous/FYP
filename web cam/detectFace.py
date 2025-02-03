import cv2
import numpy as np
import pickle
import paho.mqtt.client as mqtt

# MQTT setup
mqtt_client = mqtt.Client()
mqtt_broker = "broker.mqtt-dashboard.com"
mqtt_topic = "open_door"
mqtt_client.connect(mqtt_broker, 1883, 60)

def alert():
    mqtt_client.publish(mqtt_topic, "alert")
    print("Unauthorized Face Detected!")

def open_door():
    mqtt_client.publish(mqtt_topic, "open")
    print("Door open signal sent!")

# Load face names from names.pkl file
with open('data/names.pkl', 'rb') as f:
    known_face_names = pickle.load(f)

# Initialize video capture and setup
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('data/trainer.yml')  # Load the pre-trained model

detected_attendance = []
threshold = 100  # Lower threshold for confidence

face_detected = False  # Flag to stop scanning after detection

while not face_detected:  # Continue scanning until a face is detected
    success, frame = video.read()
    frame = cv2.flip(frame, 1)
    if not success:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face, (100, 100))  # Resize to match training size
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        label, confidence = recognizer.predict(gray_face)
        print(f"Predicted label: {label}, Confidence: {confidence}")

        if label >= len(known_face_names):
            print("Label is out of range!")
            name = "Unauthorized"
        else:
            if confidence < threshold:
                name = known_face_names[label]
            else:
                name = "Unauthorized"
        
        print(f"Recognized Name: {name}")

        if name != "Unauthorized" and name not in detected_attendance:
            detected_attendance.append(name)
            open_door()  # Trigger the door opening
            face_detected = True  # Set flag to stop scanning
            
            # Draw rectangle and label around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y + h - 6), font, 0.5, (255, 255, 255), 1)
            cv2.imshow('Face Recognition', frame)
            break  # Break out of the face detection loop
        else:
            detected_attendance.append(name)
            alert()
            face_detected = True
            # Draw rectangle and label around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y + h - 35), (x + w, y + h), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (x + 6, y + h - 6), font, 0.5, (255, 255, 255), 1)
            cv2.imshow('Face Recognition', frame)
            break  # Break out of the face detection loop

    cv2.imshow('Face Recognition', frame)

    k = cv2.waitKey(1)

    if k == ord('q'):
        print("Exiting program...")
        break

video.release()
cv2.destroyAllWindows()
mqtt_client.disconnect()
