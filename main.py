import cv2
import face_recognition
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(0)

obama_image = face_recognition.load_image_file("obama.png")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

biden_image = face_recognition.load_image_file("biden.png")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

prince_image = face_recognition.load_image_file("prince.png")
prince_face_encoding = face_recognition.face_encodings(prince_image)[0]

doni_image = face_recognition.load_image_file("doni.png")
doni_face_encoding = face_recognition.face_encodings(doni_image)[0]

bro_image = face_recognition.load_image_file("bro.png")
bro_face_encoding = face_recognition.face_encodings(bro_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    prince_face_encoding,
    doni_face_encoding,
    bro_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "Prince",
    "Doni",
    "Bro"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Read the frame
    _, img = cap.read()
    print("Width: %d, Height: %d, FPS: %d" % (cap.get(3), cap.get(4), cap.get(5)))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_encodings = face_recognition.face_encodings(img)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.insert(0, name)

    process_this_frame = not process_this_frame

    for (x, y, w, h), name in zip(faces, face_names):
        print(face_names)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (x + 6, y - 6), font, 1.0, (0, 0, 255), 1)

    # Display
    cv2.imshow('Face Recognition', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the VideoCapture object
cap.release()
