import cv2

def read_base64_image(imageName):
    '''
    Read a base64 image and return as an numpy array like OpenCV expects
    '''
    return cv2.imread(imageName)