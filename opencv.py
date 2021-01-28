import cv2

print (cv2.__version__)

image = cv2.imread("clouds.jpg")
smaller = cv2.resize(image, (400,300))
gray_image = cv2.cvtColor(smaller, cv2.COLOR_BGR2GRAY)
cv2.imshow("Over the clouds", smaller)
cv2.imshow("Over the clouds - gray", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
