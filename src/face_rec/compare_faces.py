#import packages
import argparse
import cv2
import face_recognition
import numpy as np
from utils import read_base64_image

def Compare_Faces(args=None, base64_db_image=None, base64_test_image=None):
    """
    Main function.
    """
    if args is not None and 'base_image' in args and 'compare_image' in args:
        db_image = cv2.imread(args['base_image'])
        new_image = cv2.imread(args['compare_image'])
    elif base64_db_image is not None and base64_test_image is not None:
        db_image =  read_base64_image(base64_db_image)
        new_image = read_base64_image(base64_test_image)
    else:
        print('[ERROR] You need to pass the images as parameter')
        return None

    db_image_height, db_image_width = db_image.shape[0:2]
    new_image_height, new_image_width = new_image.shape[0:2]

    #We need, at least a 80x80 pixels face. So, if the image resolution is lower than 160x160 it can't be used
    if db_image_height < 160 or db_image_width < 160 or new_image_height < 160 or new_image_width < 160:
        print('[ERROR] Image resolution to low, please use a higher resolution image')
        return None
    
    db_image_RGB = cv2.cvtColor(db_image, cv2.COLOR_BGR2RGB)
    new_image_RGB = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    db_faces = find_main_face(db_image_RGB, 'cnn')
    new_faces = find_main_face(new_image_RGB, 'cnn')

    if db_faces is None:
        print('[ERROR] No faces found in db image')
        return None
    elif new_faces is None:
        print('[ERROR] No faces found in test image')
        return None

    db_face_encoding = face_recognition.face_encodings(db_image_RGB, known_face_locations=db_faces, num_jitters=1)
    new_face_encoding = face_recognition.face_encodings(new_image_RGB, known_face_locations=new_faces, num_jitters=1)

    face_distance = np.linalg.norm(np.asarray(db_face_encoding) - np.asarray(new_face_encoding))

    print('[INFO] Faces distance: {:.2f}'.format(face_distance))
    print('[INFO] Same person probability: {:.2f}'.format(match_probability(face_distance)))

    if args is not None:
        if args['show']:
            for face in db_faces:
                draw_bbox(db_image, face)
            for face in new_faces:
                draw_bbox(new_image, face)
                
            cv2.imshow('db image', db_image)
            cv2.imshow('teste image', new_image)

            cv2.waitKey(0)

    return match_probability(face_distance)

def find_main_face(img, model):
    """
    Try to find the main face on the image (the largest one).
    """
    # recognize faces on the image
    face_locations = face_recognition.face_locations(img, model=model, number_of_times_to_upsample=0)

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

def draw_bbox(img, location):
    """
    Draws a bounding box on the image for each face on the list of given face locations.
    """
    (top, right, bottom, left) = location
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

def match_probability(face_distance, low_threshold=0.5, high_threshold=1.0):
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

if __name__ == "__main__":
    # setup argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-image', type=str, required=True,
        help='Path to the database image for comparation')
    ap.add_argument('--compare-image', type=str, required=True,
        help='Path to the new image to compare with the database image.')
    ap.add_argument('--show', type=bool,
                    help='display the images being compared')
    args = vars(ap.parse_args())

    Compare_Faces(args=args)