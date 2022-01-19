import jetson.inference
import jetson.utils
import cv2


class mnSSD:
    def __init__(self, path, threshold):
        self.path = path
        self.threshold = threshold
        self.net = jetson.inference.detectNet(self.path, self.threshold)

    def detect(self, img, display=False):
        imgCuda = jetson.utils.cudaFromNumpy(img)
        detections = self.net.Detect(imgCuda, overlay="OVERLAY_NONE")

        objects = []
        for d in detections:
            className = self.net.GetClassDesc(d.ClassID)
            objects.append([className, d])
            if display:
                x1, y1, x2, y2 = int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)
                cx, cy = int(d.Center[0]), int(d.Center[1])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, cy), (x2, cy), (255, 0, 255), 1)
                cv2.line(img, (cx, y1), (cx, y2), (255, 0, 255), 1)
                cv2.putText(img, className, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 255), 2)
                cv2.putText(img, f"FPS: {int(self.net.GetNetworkFPS())}", (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        return objects


def main():
    width, height = 640, 480
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    myModel = mnSSD("ssd-mobilenet-v2", threshold=0.5)
    while True:
        success, img = cap.read()
        objects = myModel.detect(img, True)
        if len(objects) != 0:
            print(objects[0][0])
        cv2.imshow("Window", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
