#import packages
import argparse
import cv2
import face_recognition
import numpy as np
from utils import read_base64_image
import imutils
import os

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

def face_webcam():
    cam = cv2.VideoCapture(0)
    confidence = 0.4
    threshold = 15 #15% do tamanho da imagem
    while True:
        ret, frame_video = cam.read()
        img = face_angle(frame_video)
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF  
        if k == ord('q'): 
            break

    cam.release()
    cv2.destroyAllWindows()


def face_angle(image):
    threshold = 15 #15% do tamanho da imagem

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

    nose_point, face_projection_point = face_direction_points(landmarks, image)
    face_still, rec_point1, rec_point2 = face_direction_threshold(nose_point, face_projection_point, width, 
    height, threshold)

    draw_bbox(image, face)
    draw_landmarks(image, landmarks)
    cv2.line(image, nose_point, face_projection_point, (255,0,0), 2)
    cv2.line(image, nose_point, (width - 1, nose_point[1]), (0,0,255), 2)
    cv2.rectangle(image, rec_point1, rec_point2, (0,0,255), 2)

    if face_still:
        print('[INFO] The face is in the middle')
        return image
    else:
        angle = face_angle(nose_point, face_projection_point)
        print('angle: ', angle)
        if angle > 315 or angle < 45:
            print('Direita')
        if angle > 225 and angle < 315:
            print('Cima')
        if angle > 45 and angle < 135:
            print('Baixo')
        if angle > 135 and angle < 225:
            print('Esquerda')
        return image

def Face_angle(args=None, base64image=None):
    """
    Main function.
    """
    threshold = 15 #15% do tamanho da imagem

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

    nose_point, face_projection_point = face_direction_points(landmarks, image)
    face_still, rec_point1, rec_point2 = face_direction_threshold(nose_point, face_projection_point, width, height, threshold)

    if args is not None:
        if args['show']:
            draw_bbox(image, face)
            draw_landmarks(image, landmarks)
            cv2.line(image, nose_point, face_projection_point, (255,0,0), 2)
            cv2.line(image, nose_point, (width - 1, nose_point[1]), (0,0,255), 2)
            cv2.rectangle(image, rec_point1, rec_point2, (0,0,255), 2)

            cv2.imshow('image', image)

            cv2.waitKey(0)

    if face_still:
        print('[INFO] The face is in the middle')
        return None
    else:
        angle = face_angle(nose_point, face_projection_point)
        print('angle: ', angle)
        if angle > 315 or angle < 45:
            print('Direita')
        if angle > 225 and angle < 315:
            print('Cima')
        if angle > 45 and angle < 135:
            print('Baixo')
        if angle > 135 and angle < 225:
            print('Esquerda')
        return angle

def face_direction_threshold(nose_point, face_projection_point, width, height, threshold):
    '''
    Check if the face is really biased to some direction or if it is still
    '''
    #calc the threshold rectangle
    top = int(nose_point[1] - (height * (threshold / 100) / 2))
    bottom = int(nose_point[1] + (height * (threshold / 100) / 2))
    left = int(nose_point[0] - (width * (threshold / 100) / 2))
    right = int(nose_point[0] + (width * (threshold / 100) / 2))

    #check if the projection is inside the rectangle
    if (face_projection_point[0] >= left and face_projection_point[0] <= right 
        and face_projection_point[1] >= top and face_projection_point[1] <= bottom):
        return True, (left, top), (right, bottom)
    else:
        return False, (left, top), (right, bottom) 

def face_angle(nose_point, face_projection_point):
    '''
    Estimate the face angle
    '''
    #exception cases
    if nose_point[1] == face_projection_point[1]:       # eixo y igual
        if face_projection_point[0] > nose_point[0]:    # olhando para direita
            return 0
        elif face_projection_point[0] < nose_point[0]:  # olhando para esquerda
            return 180
        else:
            return 0
    elif nose_point[0] == face_projection_point[0]:     #eixo x igual
        if face_projection_point[1] > nose_point[1]:    # olhando para baixo
            return 270
        elif face_projection_point[1] < nose_point[1]:  # olhando para cima
            return 90
        else:
            return 0
    #normal cases
    else:
        m_face = (face_projection_point[1] - nose_point[1]) / (face_projection_point[0] - nose_point[0])
    
        if face_projection_point[1] > nose_point[1]:    #olhando para 'baixo'
            if face_projection_point[0] > nose_point[0]: #olhando para 'direita'
                return np.degrees(np.arctan(m_face))
            else:    
                return np.degrees(np.arctan(m_face)) + 180
        else:    #olhando para 'cima'
            if face_projection_point[0] > nose_point[0]: #olhando para 'direita'
                return np.degrees(np.arctan(m_face)) + 360
            else:    
                return np.degrees(np.arctan(m_face)) + 180

def face_direction_points(landmarks, image):
    '''
    Estimate the face direction point

    TODO: Tune sens parameter
    '''
    sens = 1

    # 2D image points for the detected face
    image_points = np.array([
                                landmarks['nose_bridge'][3], # Nose tip
                                landmarks['chin'][8],        # Chin
                                landmarks['left_eye'][0],    # Left eye left corner
                                landmarks['right_eye'][3],   # Right eye right corne
                                landmarks['top_lip'][0],     # Left Mouth corner
                                landmarks['top_lip'][6]      # Right mouth corner
                            ], dtype=np.float)
    
    # 3D model points for a face in an arbitrary world frame
    model_points = np.array([
                                (0.0, 0.0, 0.0),          # Nose tip
                                (0.0, -66.0, -13.0),      # Chin
                                (-45.0, 34.0, -27.0),     # Left eye left corner
                                (45.0, 34.0, -27.0),      # Right eye right corne
                                (-30.0, -30.0, -25.0),    # Left Mouth corner
                                (30.0, -30.0, -25.0)      # Right mouth corner
                            ])
    
    # internal parameters for the camera (approximated)
    f = image.shape[1]
    c_x, c_y = (image.shape[1]/2, image.shape[0]/2)
    mtx = np.array([[f, 0, c_x],
                    [0, f, c_y],
                    [0, 0, 1]], dtype=np.float)
    dist = np.zeros((4,1))
    (ret, rvec, tvec) = cv2.solvePnP(model_points, image_points, mtx, dist)
    
    # project a 3D point (defined by the sensibility) onto the image plane
    # this is used to draw a line sticking out of the nose
    nose_end_3D = np.array([(0.0, 0.0, 100.0*sens)])
    (nose_end_2D, _) = cv2.projectPoints(nose_end_3D, rvec, tvec, mtx, dist)
    focus_x = int(nose_end_2D[0][0][0])
    focus_y = int(nose_end_2D[0][0][1])
    p1 = (int(image_points[0][0]), int(image_points[0][1])) #nose point
    p2 = (focus_x, focus_y)                                 #face direction point

    return p1, p2

def draw_landmarks(image, landmarks):
    """
    Draws the detected face landmarks on the image according to the given color map.
    """
    for landmark, color in LANDMARK_COLOR_MAP.items():
        for point in landmarks[landmark]:
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

    Face_angle(args=args)
    # face_webcam()