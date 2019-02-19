import face_recognition
import cv2

def check_unpresent_faces(known_face_encodings,face_encoding):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    # If a match was found in known_face_encodings, just use the first one.
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
    return name

def check_if_present(present_faces_encodings,face_encoding):
    matches = face_recognition.compare_faces(present_faces_encodings, face_encoding)
    # If a match was found in known_face_encodings, return true
    if True in matches:
        return True
    return False

print("Preparando camera...")
video_capture = cv2.VideoCapture(0)
# TODO: TURN THIS IMAGE FILE AUTOMATIC

print("Carregando lista de images...")
my_image = face_recognition.load_image_file("class/me.jpg")
classmate_image = face_recognition.load_image_file("class/face 2.jpg")

my_face_encoding = face_recognition.face_encodings(my_image)[0]
classmate_face_encoding = face_recognition.face_encodings(classmate_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    my_face_encoding,
    classmate_face_encoding
]
known_face_names = [
    "Guilherme",
    "Francine"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
presents_face_encoding = []
present_names = []

print("Iniciando sistema de chamada")
while True:
    # CAPTURE VIDEO FRAME
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Only process every other frame of video to save time
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        for face_encoding in face_encodings:
            if not check_if_present(presents_face_encoding,face_encoding):
                name = check_unpresent_faces(known_face_encodings,face_encoding)
                if name != "Unknown":
                    print(name+" presente")
                    present_names.append(name)
                    presents_face_encoding.append(face_encoding)
    process_this_frame = not process_this_frame


    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

print("As seguintes pessoas estavam presentes:")
print(present_names)