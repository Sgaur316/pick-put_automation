import imutils
import cv2 as cv
import numpy as np
import mediapipe as mp
from common import *

class HandDetect:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

    def detect_object(self,frame,**kwargs):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    # print(id,lm)
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    # if id ==0:
                    cv.circle(frame, (cx, cy), 3, (255, 0, 255), cv.FILLED)

                self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)

class ObjectDetect:
    def __init__(self,history,threshold):
        x=1
        # self.object_detector = cv.createBackgroundSubtractorMOG2(history=20, varThreshold=15)
        self.object_detector = cv.createBackgroundSubtractorKNN(history=history, dist2Threshold=threshold)

        self.index = 0
        self.last_pos = None

    def detect_object(self,frame,show_mask:str='',show_controur:bool=False):
        blurImg = cv.GaussianBlur(frame, (7, 7),cv.BORDER_DEFAULT)

        # mask = self.object_detector.apply(blurImg)
        mask = cv.Canny(image = blurImg, threshold1=100, threshold2=150)
        # kernel = np.ones((3, 3), np.uint8)
        # mask = cv.dilate(edge, kernel, iterations=1)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        self.index += 1

        count = 0
        ret_data = {}
        for cnt in contours:
            # calculate area and remove small
            area = cv.contourArea(cnt)
            if area > 600:
                count += 1
                if False:
                    M = cv.moments(cnt)
                    # print(M)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    cv.rectangle(frame, (cx, cy), (cx + 5, cy + 5), GREEN, 3)

                if True:
                    x, y, w, h = cv.boundingRect(cnt)
                    # cv.rectangle(frame,(x,y),(x+5,y+5),GREEN,3)
                    ret_data['rect'] = [x, y, w, h]
                    if show_controur == True:
                        cv.rectangle(frame, (x, y), (x + w, y + h), GREEN, 1)
                        cv.putText(frame, "{},{}:{},{}:{}".format(area, x, y, w, h), (x, y - 10),
                                   cv.FONT_HERSHEY_PLAIN, 1,
                                   (0, 0, 255), thickness=1)
                        cv.circle(frame, (x, y), 2, (0, 0, 255), 2)

                if False:
                    (x, y), radius = cv.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    ret_data['circle'] = [center,radius]
                    if show_controur == True:
                        cv.circle(frame, center, radius, (0, 255, 0), 2)

                if False:
                    rect = cv.minAreaRect(cnt)
                    box = cv.boxPoints(rect)
                    box = np.int0(box)
                    if show_controur == True:
                        cv.drawContours(frame, [box], 0, (0, 0, 255), 2)

                if True:
                    rows, cols = frame.shape[:2]
                    [vx, vy, x, y] = cv.fitLine(cnt, cv.DIST_L2, 0, 0.01, 0.01)
                    lefty = int((-x * vy / vx) + y)
                    righty = int(((cols - x) * vy / vx) + y)

                    ret_data['line'] = [(cols - 1, righty), (0, lefty)]

                    if show_controur == True:
                        cv.line(mask, (cols - 1, righty), (0, lefty), (0, 255, 0), 1)
                        cv.line(frame, (cols - 1, righty), (0, lefty), (0, 255, 0), 1)


                    # cv.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

        if show_mask != '':
            cv.imshow(show_mask, mask)
        return ret_data

class ColorTrack:
    def __init__(self,maxCol,minCol):
        self.minCol = minCol
        self.maxCol = maxCol

    def detect_object(self,frame,add_ref=False,show_mask=False,show_controur=False):
        blurred = cv.GaussianBlur(frame, (11, 11), 0)
        hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

        mask = cv.inRange(hsv, self.minCol, self.maxCol)
        mask = cv.erode(mask, None, iterations=2)
        mask = cv.dilate(mask, None, iterations=2)
        if show_mask == True:
            cv.imshow('masked',mask)

        cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                                cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv.contourArea)
            ((x, y), radius) = cv.minEnclosingCircle(c)
            M = cv.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv.circle(frame, (int(x), int(y)), int(radius),
                           (0, 255, 255), 2)
                cv.circle(frame, center, 5, (0, 0, 255), -1)

DETECTION_CLASS = ObjectDetect
