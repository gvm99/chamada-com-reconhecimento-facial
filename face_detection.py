import cv2
import face_recognition
import face_recognizer as cr

print("Preparando camera...")
video_capture = cv2.VideoCapture(0)

print("Carregando o modelo...")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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


print("INICIANDO...")
last_frame = 0 
already_present_list = []
while True:
    _, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]
    faces = face_cascade.detectMultiScale(rgb_frame, 1.3, 5)
    faces_encoded = []
    if last_frame != len(faces):
        for (x,y,w,h) in faces:
            enc = face_recognition.face_encodings(frame[y:y+h, x:x+w])
            if(len(enc)):
                faces_encoded.append(enc[0])
        name,already_present_list = cr.face_recognizer(known_face_encodings,known_face_names,faces_encoded,already_present_list)
        if len(name) != 0:
            print(name)
    last_frame = len(faces)

    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

