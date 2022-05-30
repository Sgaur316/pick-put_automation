import time
import cv2 as cv

TOP_ID = 0
SIDE_ID = 1

YELLOW_LOWER = (20, 100, 100)
YELLOW_UPPER = (30, 255, 255)

RED = [0,0,255]
GREEN = [0,255,0]

class FPS:
    def __init__(self):
        self.prev_time = time.time()

    def updateFps(self,frame=[]):
        curr_time = time.time()
        fps = int(1/(curr_time-self.prev_time))
        if frame != []:
            cv.putText(frame, "{} fps".format(fps), (5,15),
                   cv.FONT_HERSHEY_PLAIN, 1,
                   RED, thickness=1)
        self.prev_time = curr_time
        return fps
