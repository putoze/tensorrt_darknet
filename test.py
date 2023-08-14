import cv2

from utils.fitEllipse import find_max_contour

img = cv2.imread("./test_image/3.png")  
img = cv2.resize(img,(400,400))
flag_list = [1,1,1,1,1,1,1]
target_img = find_max_contour(img,flag_list)

cv2.imshow("Original Image", img)
cv2.imshow("fitEllipse Image", target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
