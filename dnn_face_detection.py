import cv2
import math

DNN = "TF"
if DNN == "CAFFE":
    modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "models/deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

conf_threshold = 0.7

def detectFaceOpenCVDnn(net, frame):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


print("Preparando camera...")
video_capture = cv2.VideoCapture(0)
face_points = []
print("Euclidean")
while True:
    _, frame = video_capture.read()
    outOpencvDnn, bboxes = detectFaceOpenCVDnn(net,frame)
    if(len(bboxes) > 0):
        medium_point_x = int((bboxes[0][2] + bboxes[0][0])/ 2)
        medium_point_y = int((bboxes[0][3] + bboxes[0][1])/ 2)
        if(len(face_points)>0):
            euclidean = math.sqrt((medium_point_x - face_points[0])**2 + (medium_point_y - face_points[1])**2)
            if euclidean > 10 :
                print("Pessoa diferente")
        cv2.circle(frame,(medium_point_x, medium_point_y), 5, (0,255,0), -1)
        face_points = [medium_point_x,medium_point_y] 
    cv2.imshow('Video', frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

