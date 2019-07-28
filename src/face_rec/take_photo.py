#import packages
import argparse
import cv2
import face_recognition
import os 

def main(args):
    """
    Main function.
    """
    webcam = cv2.VideoCapture(0)
    imagePath = os.path.join('./figs', args['image_name']) + '.png'

    print('Instructions\n'
          'P -> take photo\n'
          'Q -> exit program\n')
    
    while True:
        ret, img = webcam.read()

        if not ret or img is None:
            print('Problem while capturing the webcam image')

        flipped_img = cv2.flip(img, 1)

        cv2.imshow('webcam', flipped_img)

        k = cv2.waitKey(1)

        if k == ord('q'):
            break
        elif k == ord('p'):
            cv2.imwrite(imagePath, img)
            break

if __name__ == "__main__":
    # setup argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-name', type=str, required=True,
        help='Name to save the image')
    args = vars(ap.parse_args())

    main(args)