#import packages
import argparse
import cv2
import face_recognition
import numpy as np
from .utils import read_base64_image

#TODO: Implement a threshold to know if the face is in center or biased to some direction

# landmark color mapping for drawing
LANDMARK_COLOR_MAP = {
    "left_eye": (255, 0, 0), #Desenha os pontos do olho esquerdo de azul
    "right_eye": (255, 0, 0), #Desenha os pontos do olho direito de azul
    "left_eyebrow": (0, 0, 255), #Desenha os pontos da sobrancelha esquerda de vermelho
    "right_eyebrow": (0, 0, 255), #Desenha os pontos da sobrancelha direita de vermelho
    "nose_tip": (0, 255, 0), #Desenha os pontos da ponta do nariz de verde
    "nose_bridge": (255, 255, 0), #Desenha os pontos do corpo do nariz de ciano
    "bottom_lip": (0, 127, 255), #Desenha os pontos do labio de amarelo
    "top_lip": (0, 255, 127), #Desenha os pontos do labio de amarelo
    "chin": (0, 0, 0) #Desenha os pontos da bochecha de preto
}

def Eye_Closed(args=None, base64image=None):
    """
    Main function.
    """
    if args is not None and 'image' in args:
        image = cv2.imread(args['image'])
    elif base64image is not None:
        image = read_base64_image(base64image)
    else:
        print('[ERROR] You need to pass the images as parameters')
        return None

    height, width = image.shape[0:2]

    #We need, at least a 80x80 pixels face. So, if the image resolution is lower than 160x160 it can't be used
    if height < 160 or width < 160:
        print('[ERROR] Image resolution to low, please use a higher resolution image')
        return None
    
    image = cv2.flip(image, 1)

    image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face = find_main_face(image_RGB, 'cnn')

    if face is None:
        print('[ERROR] No faces found in db image')
        return None
    
    # try to find the landmarks on the detected face
    landmarks = find_landmarks(image, face)
    if landmarks is None:
        print('[ERROR] Couldnt find landmarks on the face image')
        return None

    state = eye_state(landmarks)

    print('Olho esquerdo fechado com probabilidade: ', state[0])
    print('Olho direito fechado com probabilidade: ', state[1])

    if args is not None:
        if args['show']:
            draw_bbox(image, face)
            draw_landmarks(image, landmarks)

            cv2.imshow('image', image)

            cv2.waitKey(0)

    return state

def eye_aspect_ratio(eye):
    """
    Compute the eye aspect ratio.
    """
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(np.array(eye[1])-np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2])-np.array(eye[4]))

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(np.array(eye[0])-np.array(eye[3]))

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def eye_state(landmarks):
    '''
    determine eye state

    ear_thresh --> de 0 a 1 (1 é muito facil, 0 muito dificil)
    '''
    eye = [0, 0]
    ear_left = eye_aspect_ratio(landmarks['left_eye'])
    ear_right = eye_aspect_ratio(landmarks['right_eye'])
    
    eye[0] = closeness_probability(ear_left)
    eye[1] = closeness_probability(ear_right)

    return eye

def closeness_probability(eye_ar, low_threshold=0.2, high_threshold=0.3):
    '''
    We will use:
        eye_ar < low_threshold --> 100% 
        low_threshold >= eye_ar <= high_threshold --> 1 - ((eye_ar - low_threshold) / (high_threshold - low_threshold))
        face_distace > high_threshold --> 0% 
    TODO: Get some examples and tune that numbers
    '''
    if eye_ar < low_threshold:
        return 1.0
    elif eye_ar > high_threshold:
        return 0.0
    else:
        return (1 - ((eye_ar - low_threshold) / (high_threshold - low_threshold)))

def draw_landmarks(image, landmarks):
    """
    Draws the detected face landmarks on the image according to the given color map.
    """
    for landmark, color in LANDMARK_COLOR_MAP.items():
        for i, point in enumerate(landmarks[landmark]):
            cv2.circle(image, point, 1, color, -1)

def find_main_face(image, model):
    """
    Try to find the main face on the image (the largest one).
    """
    # recognize faces on the image
    face_locations = face_recognition.face_locations(image, model=model, number_of_times_to_upsample=0)

    # if no faces were found, indicate the failure
    if len(face_locations) == 0:
        return None

    # if there is more than one face on the image, pick the largest one
    elif len(face_locations) > 1:
        max_size = 0
        max_location = None
        for (top, right, bottom, left) in face_locations:
            size = (bottom - top) * (right - left)
            if size > max_size:
                max_size = size
                max_location = (top, right, bottom, left)
        return [max_location]

    # if there is only one, return the list element
    else:
        return face_locations

def find_landmarks(image, location):
    """
    Try to find the landmarks on the detected face.
    """
    landmarks = face_recognition.face_landmarks(image, face_locations=location)
    if len(landmarks) == 0:
        return None
    else:
        return landmarks[0]

def draw_bbox(image, location):
    """
    Draws a bounding box on the image for each face on the list of given face locations.
    """
    (top, right, bottom, left) = location[0]
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0))

if __name__ == "__main__":
    # setup argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True,
        help='Path to the image.')
    ap.add_argument('--show', action='store_true',
                    help='display the images being compared')
    args = vars(ap.parse_args())

    Eye_Closed(args=args)