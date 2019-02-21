import face_recognition
import numpy as np

def check_unpresent_faces(known_face_encodings,known_faces_names,face_encoding,present_faces):
    matches = face_recognition.compare_faces([known_face_encodings], face_encoding)
    name = "Unknown"
    # If a match was found in known_face_encodings, just use the first one.
    if np.any(matches[0]):
        first_match_index = np.where(matches)[0][0]
        name = known_faces_names[first_match_index]
        present_faces.append(face_encoding)
    return name,present_faces

def check_if_present(present_faces_encodings,face_encoding):
    return np.any(face_recognition.compare_faces([present_faces_encodings], face_encoding)[0])

def face_recognizer(list_faces,known_faces_names,encoded_faces,present_faces):
    presence_in_frame = []
    encoded_present_list = present_faces
    
    for encoded_face in encoded_faces:
        if len(present_faces) == 0:
            name,encoded_present_list = check_unpresent_faces(list_faces,known_faces_names,encoded_face,present_faces)
            if name !="Unknown":
                presence_in_frame.append(name)
        elif check_if_present(present_faces,encoded_face) == False:
            name,encoded_present_list = check_unpresent_faces(list_faces,known_faces_names,encoded_face,present_faces)
            if name !="Unknown":
                presence_in_frame.append(name)
    
    return presence_in_frame,encoded_present_list
    