import numpy as np
import cv2

def recognize_text(original):
    # Morphological gradient:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(original, cv2.MORPH_GRADIENT, kernel)

    # Binarization
    ret, binarization = cv2.threshold(opening, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Connected horizontally oriented regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(binarization, cv2.MORPH_CLOSE, kernel)

    # find countours
    contours, hierarchy = cv2.findContours(
        connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours, hierarchy

def recognize_card(imgPath, destPath):	
	img = cv2.imread(imgPath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	denoised = cv2.fastNlMeansDenoising(gray, None, 3, 7, 21)
	contours, hierarchy = recognize_text(gray)
	mask = np.zeros(gray.shape, np.uint8)
	max_tam = 0

	for contour in contours:
		[x, y, w, h] = cv2.boundingRect(contour)

		mskRoi = mask[y:y+h, x:x+w]
		
		cv2.drawContours(mask, [contour], 0, 255, -1) #CV_FILLED
		nz = cv2.countNonZero(mskRoi)
		ratio = (float)(nz) / (float)(h*w)

		# got this value from left heel
		if ratio > 0.5 and ratio < 1.5:
			if h + w > max_tam:
				max_tam = h + w 
				max_x = x
				max_y = y
				max_h = h
				max_w = w

	if max_tam != 0:
		final_image = img[max_y:max_y+max_h, max_x:max_x+max_w]
		cv2.imwrite(destPath, final_image)
		return True
	else:
		return False

if __name__ == '__main__':
	print(recognize_card('../../data/raphael_cnh.jpg', '../../data/raphael_cnh.png'))
