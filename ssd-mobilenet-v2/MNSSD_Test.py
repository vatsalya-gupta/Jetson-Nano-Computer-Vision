import cv2
import MobileNetSSDModule as mnSSD


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
myModel = mnSSD.mnSSD("ssd-mobilenet-v2", threshold=0.5)

while True:
    success, img = cap.read()
    objects = myModel.detect(img, True)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
