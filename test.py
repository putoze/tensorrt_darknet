import cv2

from utils.fitEllipse import find_eye_roi,find_max_contour

img = cv2.imread("eye.jpg")  

flag_list = [1,1,0,1,0,1,1]
target_img,contours = find_eye_roi(img,flag_list)
target_img = find_max_contour(target_img,contours)

cv2.imshow("Original Image", img)
cv2.imshow("Contours Image", target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
