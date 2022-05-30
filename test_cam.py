import cv2 as cv
print('Test Started')
Video = cv.VideoCapture(2, cv.CAP_DSHOW)
print('Video Captured, Changing Resolution')
Video.set(3, 1280)
Video.set(4, 720)
print('Resolution Changed, Capturing Frame')

while True:
    ret,frame = Video.read()
    if ret == True:
        cv.imshow('camTest',frame)
        print(frame.shape)
    else:
        print('No Frame Captured')
    cv.waitKey(1)