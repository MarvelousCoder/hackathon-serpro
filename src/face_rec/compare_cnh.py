import cv2
import numpy as np
import imutils
import argparse
import os
import math

def get_blob(detector, img, confidence):
    image = cv2.imread(img)
    if image is None:
        print("{} is a invalid path".format(img))
        exit()
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # constrói um blob da imagem
    imageBlob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 1.0, (300, 300),
    (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # aplica o face detector baseado em deep learning da OpenCV para localizar
    # rostos na imagem de entrada 
    detector.setInput(imageBlob)
    detections = detector.forward()

    # garante que pelo menos uma face foi encontrada
    if len(detections) > 0:
        # Assume-se que cada imagem possui apenas um rosto, então encontra a bounding box
        # com a maior probabilidade
        i = np.argmax(detections[0, 0, :, 2])
        detection_confidence = detections[0, 0, i, 2]

        # Garante que a detecção com a maior probabilidade também significa
        # nossa teste mínimo de probabilidade (filtra detecções fracas)
        if detection_confidence >= confidence:
            # computa as coordenadas da bounding box para o rosto
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extrai a ROI da face e pega as dimensões
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # constrói um blob para a ROI da face, então passa o
            # blob no face embedding model para obter a 128-d 
            # quantificação da face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
        else:
            print("Image quality {} is too bad for {} percent of confidence".format(img, confidence))
            exit()
    else:
        print("No face found in {}".format(img))
        exit()
    return faceBlob


def face_compare(detector, confidence, base_img, test_img, embedder):
    blob = get_blob(detector, base_img, confidence)
    embedder.setInput(blob)
    vec_base = embedder.forward()
    blob = get_blob(detector, test_img, confidence)
    embedder.setInput(blob)
    vec_test = embedder.forward()
    face_distance = np.linalg.norm(np.asarray(vec_base) - np.asarray(vec_test))

    print('[INFO] Faces distance: {:.2f}'.format(face_distance))
    print('[INFO] Same person probability 1: {:.2f}'.format(match_probability_1(face_distance)))
    print('[INFO] Same person probability 2: {:.2f}'.format(match_probability_2(face_distance)))
    return match_probability_1(face_distance)

def match_probability_1(face_distance, face_match_threshold=0.6):
    # Taken from: https://github.com/ageitgey/face_recognition/wiki/Calculating-Accuracy-as-a-Percentage
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return max(0., linear_val)
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return max(0., linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2)))

def match_probability_2(face_distance, low_threshold=0.5, high_threshold=1.0):
    '''
    Translate the face distance in a match probability to facilitate it use
    According to http://dlib.net/face_recognition.py.html dlib has a 99.38% accuracy in LFW dataset for a 0.6 threshold
    So, we will use:
        face_distance < low_threshold --> 100% 
        low_threshold >= face_distance <= high_threshold --> 1 - ((face_distance - low_threshold) / (high_threshold - low_threshold))
        face_distace > high_threshold --> 0% 
    TODO: Get some examples and tune that numbers
    '''
    if face_distance < low_threshold:
        return 1.0
    elif face_distance > high_threshold:
        return 0.0
    else:
        return (1 - ((face_distance - low_threshold) / (high_threshold - low_threshold)))


def read_args():
    parser = argparse.ArgumentParser(description='Detect face and smile from webcam')
    parser.add_argument("-d", "--detector", default='../../face_detection_model', 
                        help="path to OpenCV's deep learning face detector")
    parser.add_argument("-c", "--confidence", type=float, default=0.5, 
                        help="minimum probability to filter weak detections")
    parser.add_argument("-ib", "--imageb", required=True, help="path to the base image")
    parser.add_argument("-it", "--imaget", required=True, help="path to the test image")
    args = parser.parse_args()
    protoPath = os.path.sep.join([args.detector, "deploy.prototxt"])
    modelPath = os.path.sep.join([args.detector, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    embedder = os.path.sep.join([args.detector, "openface_nn4.small2.v1.t7"])
    embedder = cv2.dnn.readNetFromTorch(embedder)
    return detector, args.confidence, args.imageb, args.imaget, embedder 

def main():
    detector, confidence, base_img, test_img, embedder = read_args()
    face_compare(detector, confidence, base_img, test_img, embedder)

if __name__ == '__main__':
    main()
