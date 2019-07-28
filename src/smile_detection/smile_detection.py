import tensorflow as tf
from tensorflow.keras.models import model_from_json
import cv2 as cv
import numpy as np
import imutils
import argparse
import os

def smile_detection_image(face_model, confidence, smile_model, img):
    img = cv.imread(img)
    if img is None:
        print("Image could not be read!")
        exit()
    img = imutils.resize(img, width=600)
    h, w = img.shape[:2]
    imageBlob = cv.dnn.blobFromImage(cv.resize(
        img, (300, 300)), 1.0, (300, 300), 
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_model.setInput(imageBlob)
    detections = face_model.forward()
    i = np.argmax(detections[0, 0, :, 2])
    detection_confidence = detections[0, 0, i, 2]
    if detection_confidence >= confidence:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = img[startY:endY, startX:endX]
    if face.shape[0] == 0 or face.shape[1] == 0: face = img
    gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY) 
    data = cv.resize(gray, (64, 64))
    data = data.astype(np.float32) / 255.
    output = np.array(smile_model(np.expand_dims(np.expand_dims(data, axis=0), axis=-1)))
    print("Smile prob: {}".format(output[0][1]*100))
    return output[0][1]

def smile_detection_webcam(face_model, confidence, smile_model):
    cam = cv.VideoCapture(0)
    _, frame_video = cam.read()
    frame = imutils.resize(frame_video, width=600)
    h, w = frame.shape[:2]
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter('output.avi', fourcc, 20.0, (w,h))
    try:
        while True:
            last_frame = np.copy(frame_video)
            grabbed, frame_video = cam.read()
            if frame_video is None or not grabbed: break
            face = np.copy(frame_video)
            frame = imutils.resize(frame_video, width=600)
            h, w = frame.shape[:2]
            
            imageBlob = cv.dnn.blobFromImage(cv.resize(
                frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            face_model.setInput(imageBlob)
            detections = face_model.forward()
            i = np.argmax(detections[0, 0, :, 2])
            detection_confidence = detections[0, 0, i, 2]
            if detection_confidence >= confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = np.copy(frame[startY:endY, startX:endX])
            if face.shape[0] == 0 or face.shape[1] == 0: face = np.copy(last_frame)

            # desenha a bounding box do rosto com a probabilidade associada
            gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY) 
            data = cv.resize(gray, (64, 64))
            data = data.astype(np.float32) / 255.
            output = np.array(smile_model(np.expand_dims(np.expand_dims(data, axis=0), axis=-1)))
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv.putText(frame, "Sorriso: {}".format(output[0][1]), (startX, y),
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
            cv.imshow("Webcam", frame)
            k = cv.waitKey(1) & 0xFF  
            if k == ord('q'): break
    except KeyboardInterrupt:
        pass
    cam.release()
    out.release()
    cv.destroyAllWindows()


def read_args():
    parser = argparse.ArgumentParser(description='Detect face and smile from webcam')
    parser.add_argument("-d", "--detector", default='../../face_detection_model', 
                        help="path to OpenCV's deep learning face detector")
    parser.add_argument("-c", "--confidence", type=float, default=0.4, 
                        help="minimum probability to filter weak detections")
    parser.add_argument("-m", "--model", default='../../notebooks/model.json', 
                        help="path to model's json")
    parser.add_argument("-w", "--weights", default='../../notebooks/weights.h5', 
                        help="path to model's weight")
    parser.add_argument("-i", "--image", required=True, help="path to the image")
    args = parser.parse_args()
    model = model_from_json(open(args.model).read())
    model.load_weights(args.weights)
    protoPath = os.path.sep.join([args.detector, "deploy.prototxt"])
    modelPath = os.path.sep.join([args.detector, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv.dnn.readNetFromCaffe(protoPath, modelPath)
    return detector, args.confidence, model, args.image 

def main():
    detector, confidence, model, img = read_args()
    # smile_detection_webcam(detector, confidence, model)
    smile_detection_image(detector, confidence, model, img)

if __name__ == '__main__':
    main()
