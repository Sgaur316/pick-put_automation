# import necessary libraries

import cv2 as cv
import numpy as np
import os
import pickle,enum
import imutils
from common import *


class CalType(enum.Enum):
    CROP = 1
    PERSPECTIVE = 2
    NoCal = 3

class CalState(enum.Enum):
    Stopped = 1
    SelectArea = 2
    Tilt = 3
    Point3 = 4
    Point4 = 5

class Calibrate_v2:
    def __init__(self,name,cal_type=CalType.CROP,rotate=-1):
        self.name = name
        self.rotate = rotate
        self.fname = 'cal_{}.bin'.format(name)
        self.calType = cal_type
        self.cal_state = CalState.Stopped
        self.curr_cal_point = 0
        if os.path.isfile(os.path.join(os.getcwd(),'data',self.fname)):
            fo = open(os.path.join(os.getcwd(),'data',self.fname),'rb')
            self.cal_data = pickle.load(fo)
            print(self.cal_data)
            fo.close()
        else:
            self.cal_data = [[10,10],[300,10],[10,300],[300,300],0]
        if len(self.cal_data) < 5:
            self.cal_data.append(0)

    def keyHandler(self,frame):
        frame = imutils.rotate(frame, self.cal_data[4])
        if self.rotate >= cv.ROTATE_90_CLOCKWISE and self.rotate <= cv.ROTATE_90_COUNTERCLOCKWISE:
            frame = cv.rotate(frame,self.rotate)
        key = cv.waitKey(1)
        if key != -1:
            try:
                key_str = chr(key)
            except Exception as e:
                key_str = ''
            key_str = key_str.lower()
            self.key_task(key_str,frame.shape)
        if self.cal_state == CalState.SelectArea:
            cv.putText(frame, "{}".format('Select Area'), (10,10),
                       cv.FONT_HERSHEY_PLAIN, 1,
                       (0, 0, 0), thickness=2)
            self.add_cal_window(frame)

        elif self.cal_state == CalState.Tilt:
            cv.putText(frame, "Tilt:{}".format(self.cal_data[4]), (10, 10),
                       cv.FONT_HERSHEY_PLAIN, 1,
                       (0, 0, 0), thickness=2)

        if self.cal_state != CalState.Stopped:
            cv.imshow('Calibrate_{}'.format(self.name), frame)

        if self.calType == CalType.CROP:
            rFrame = self.__crop(frame)
        elif self.calType == CalType.PERSPECTIVE:
            rFrame = self.update_Perspective(frame)
        else:
            rFrame = frame
        return rFrame

    def add_cal_window(self,frame):
        radius = 5
        cv.line(frame, tuple(self.cal_data[0]), tuple(self.cal_data[1]), color=(0, 0, 0), thickness=1)
        cv.line(frame, tuple(self.cal_data[0]), tuple(self.cal_data[2]), color=(0, 0, 0), thickness=1)
        cv.line(frame, tuple(self.cal_data[1]), tuple(self.cal_data[3]), color=(0, 0, 0), thickness=1)
        cv.line(frame, tuple(self.cal_data[2]), tuple(self.cal_data[3]), color=(0, 0, 0), thickness=1)
        for index,val in enumerate(self.cal_data):
            if index > 3:
                break
            color = RED if index == self.curr_cal_point else GREEN
            cv.circle(frame, tuple(val), color=color, radius=radius, thickness=-1)
            scale = -1 if index < 2 else 1
            cv.putText(frame, "{}".format(val), (val[0] + (scale * radius * 2), val[1] + (scale * radius * 2)),
                       cv.FONT_HERSHEY_PLAIN, 1,
                       (0, 0, 0), thickness=1)

    def key_task(self,key_str,size):
        if self.cal_state == CalState.Stopped:
            if key_str == 'c':
                self.cal_state = CalState.SelectArea
                self.curr_cal_point = 0

        else:
            curr_point = self.cal_data[self.curr_cal_point]
            if key_str == 'a':
                if self.cal_state == CalState.SelectArea:
                    curr_point[0] -= 5
                elif self.cal_state == CalState.Tilt:
                    self.cal_data[4] -= 1
            elif key_str == 's':
                if self.cal_state == CalState.SelectArea:
                    curr_point[1] += 5
            elif key_str == 'd':
                if self.cal_state == CalState.SelectArea:
                    curr_point[0] += 5
                elif self.cal_state == CalState.Tilt:
                    self.cal_data[4] += 1
            elif key_str == 'w':
                if self.cal_state == CalState.SelectArea:
                    curr_point[1] -= 5
            elif key_str == 'q':
                fo = open(os.path.join(os.getcwd(),'data',self.fname), 'wb')
                pickle.dump(self.cal_data, fo)
                fo.close()
                self.cal_state = CalState.Stopped
                cv.destroyWindow('Calibrate_{}'.format(self.name))
            elif key_str == 'n':
                self.curr_cal_point += 1
                self.curr_cal_point = 0 if self.curr_cal_point > 3 else self.curr_cal_point
            elif key_str == 't':
                self.cal_state = CalState.Tilt
            else:
                print('Invalid Key:{}'.format(key_str))

            if curr_point[0] > size[1]:
                curr_point[0] = size[1]
            if curr_point[1] > size[0]:
                curr_point[1] = size[0]

            if curr_point[0] < 0:
                curr_point[0] = 0
            if curr_point[1] < 0:
                curr_point[1] = 0

            if self.cal_data[4] > 360:
                self.cal_data[4] = 0
            elif self.cal_data[4] < 0:
                self.cal_data[4] = 360

    def __crop(self,frame):
        h1 = abs(self.cal_data[0][1] - self.cal_data[2][1])
        h2 = abs(self.cal_data[1][1] - self.cal_data[3][1])

        w1 = abs(self.cal_data[0][0] - self.cal_data[1][0])
        w2 = abs(self.cal_data[2][0] - self.cal_data[3][0])

        # if h1 < h2
        x1 = self.cal_data[0][1]
        x2 = x1 + (h1 if h1<h2 else h2)
        y1 = self.cal_data[0][0]
        y2 = y1 + (w1 if w1<w2 else w2)

        frame = frame[x1:x2,y1:y2]
        return frame

    def update_Perspective(self,frame):
        pts1 = np.float32(self.cal_data) #actual Image
        size_w = 300
        size_h = 600
        pts2 = np.float32([[0,0],[size_w,0],[0,size_h],[size_w,size_h]]) #perspective Image

        matrix = cv.getPerspectiveTransform(pts1,pts2)

        final_result = cv.warpPerspective(frame,matrix,(size_w,size_h))

        return final_result

    # def order_points(self, pts):
    #     # initialzie a list of coordinates that will be ordered
    #     # such that the first entry in the list is the top-left,
    #     # the second entry is the top-right, the third is the
    #     # bottom-right, and the fourth is the bottom-left
    #     rect = np.zeros((4, 2), dtype="float32")
    #     # the top-left point will have the smallest sum, whereas
    #     # the bottom-right point will have the largest sum
    #     # ptsarr = np.array(pts)
    #     s = pts.sum(axis=1)
    #     rect[0] = pts[np.argmin(s)]
    #     rect[2] = pts[np.argmax(s)]
    #     # now, compute the difference between the points, the
    #     # top-right point will have the smallest difference,
    #     # whereas the bottom-left will have the largest difference
    #     diff = np.diff(pts, axis=1)
    #     rect[1] = pts[np.argmin(diff)]
    #     rect[3] = pts[np.argmax(diff)]
    #     # return the ordered coordinates
    #     return rect
    #
    # def update_Perspective(self, frame):
    #     # obtain a consistent order of the points and unpack them
    #     # individually
    #     pts = np.array(self.cal_data, np.float32)
    #     rect = self.order_points(pts)
    #     (tl, tr, br, bl) = rect
    #     # compute the width of the new image, which will be the
    #     # maximum distance between bottom-right and bottom-left
    #     # x-coordiates or the top-right and top-left x-coordinates
    #     widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    #     widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    #     maxWidth = max(int(widthA), int(widthB))
    #
    #     # compute the height of the new image, which will be the
    #     # maximum distance between the top-right and bottom-right
    #     # y-coordinates or the top-left and bottom-left y-coordinates
    #     heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    #     heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    #     maxHeight = max(int(heightA), int(heightB))
    #
    #     # now that we have the dimensions of the new image, construct
    #     # the set of destination points to obtain a "birds eye view",
    #     # (i.e. top-down view) of the image, again specifying points
    #     # in the top-left, top-right, bottom-right, and bottom-left
    #     # order
    #     dst = np.array([
    #         [0, 0],
    #         [maxWidth - 1, 0],
    #         [maxWidth - 1, maxHeight - 1],
    #         [0, maxHeight - 1]], dtype="float32")
    #
    #     # compute the perspective transform matrix and then apply it
    #     # image = cv.resize(frame, (720, 400))
    #     M = cv.getPerspectiveTransform(pts, dst)
    #     warped = cv.warpPerspective(frame, M, (maxWidth, maxHeight))
    #     return warped
    #     # print("Original points: ", pts, "data type: ", type(pts))
    #     # print("dst points: ", dst, "data type: ", type(dst))
    #     # print("Transformation: ", M)
    #     # break

