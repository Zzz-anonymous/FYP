import pickle
import numpy as np
import cv2
import os

# Ensure 'data' directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Load or initialize data for face encodings and names
faces_data_path = 'data/faces_data.pkl'
names_data_path = 'data/names.pkl'

# Create empty files if they don't exist
if not os.path.exists(faces_data_path):
    with open(faces_data_path, 'wb') as f:
        pickle.dump([], f)

if not os.path.exists(names_data_path):
    with open(names_data_path, 'wb') as f:
        pickle.dump([], f)

# Initialize face recognizer and face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8)


# Path to save the trainer model
trainer_path = 'data/trainer.yml'

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces_data = []
names = []
i = 0

# Capture faces from webcam
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (100, 100))  # Resize to 100x100
        # Convert the resized image to grayscale before appending to faces_data
        gray_resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Capture face data (maximum of 50 faces)
        if len(faces_data) < 50 and i % 10 == 0:
            faces_data.append(gray_resized_img)  # Append the RGB image
            names.append("temporary_name")  # Add a placeholder name temporarily
            print(f"Captured face data: {len(faces_data)}")  # Debug: check how many faces have been captured
        i += 1
        
        # Display face count on the screen
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)

    # Stop the capture if 'q' is pressed or 50 faces have been captured
    if k == ord('q') or len(faces_data) == 50:
        name = input("Enter your name: ")
        
        # Update names with the actual name instead of placeholder
        names = [name] * len(faces_data)

        # Save captured faces and names
        with open(faces_data_path, 'rb') as f:
            existing_faces = pickle.load(f)
        existing_faces.extend(faces_data)
        with open(faces_data_path, 'wb') as f:
            pickle.dump(existing_faces, f)

        with open(names_data_path, 'rb') as f:
            existing_names = pickle.load(f)
        existing_names.extend(names)  # Now the names will match the faces correctly
        with open(names_data_path, 'wb') as f:
            pickle.dump(existing_names, f)

        print(f"Captured {len(faces_data)} faces for {name}. Now training the model...")
        break

video.release()
cv2.destroyAllWindows()

# Now train the model
with open(faces_data_path, 'rb') as f:
    faces = pickle.load(f)

with open(names_data_path, 'rb') as f:
    names = pickle.load(f)

# Check if faces and names lists are not empty and have the same length
if len(faces) == 0 or len(names) == 0:
    print("Error: No faces or names to train the model. Please capture faces first.")
else:
    # Check if the length of faces and names match
    if len(faces) == len(names):
        print(f"Training with {len(faces)} faces...")
        
        # Debug: Print the contents of faces and names to check
        print(f"Faces: {len(faces)} samples")
        print(f"Names: {len(names)} samples")
        
        # Train the model with the faces and corresponding names
        recognizer.train(faces, np.array([names.index(name) for name in names]))  # Use the index as label
        recognizer.save(trainer_path)
        print("Model trained and saved as 'trainer.yml'")
    else:
        print(f"Error: The number of faces ({len(faces)}) does not match the number of names ({len(names)}). Please check the data.")
