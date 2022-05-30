import os
import threading
import socket
import time
from datetime import datetime
from threading import Thread, Lock, Event
import multiprocessing
from common import *
from algo.calibration import *
from algo.capture_video import *
from algo.detection import *



class OperatorTracker:

    def __init__(self):
        self.vLock = Lock()
        self.hlock = Lock()
        self.init_threads()

    def init_threads(self):
        self.capKill = Event()
        self.capKill.clear()

        if SIDE_ID !=-1 :
            vParam = {}
            vParam['name'] = 'side'
            vParam['cal_type'] = CalType.CROP
            vParam['rotate'] = cv.ROTATE_90_COUNTERCLOCKWISE
            vParam['lock'] = multiprocessing.Lock()
            vParam['qu'] = multiprocessing.Queue()
            vParam['cam_id'] = SIDE_ID
            vParam['resolution'] = [1280,720]
            vParam['contour'] = {}
            vParam['contour']['mask'] = 'side_mask'
            vParam['contour']['history'] = 10
            vParam['contour']['threshold'] = 150
            self.side_param = vParam

            self.vThread = multiprocessing.Process(target=CaptureProcess, kwargs={'param': vParam})
            self.vThread.start()

        if TOP_ID !=-1:
            hParam = {}
            hParam['name'] = 'top'
            hParam['cal_type'] = CalType.CROP
            hParam['rotate'] = -1
            hParam['lock'] = multiprocessing.Lock()
            hParam['qu'] = multiprocessing.Queue()
            hParam['cam_id'] = TOP_ID
            hParam['resolution'] = [1280,720]
            hParam['contour'] = {}
            hParam['contour']['mask'] = 'top_mask'
            hParam['contour']['history'] = 10
            hParam['contour']['threshold'] = 150
            self.hThread = multiprocessing.Process(target=CaptureProcess,kwargs={'param':hParam})

            self.hThread.start()
            self.top_param = hParam

        t1 = threading.Thread(target=self.syncThread)
        t2 = threading.Thread(target=self.UserInteraction)
        t1.start()
        t2.start()

        self.vIndex = 0
        self.hIndex = 0

    def UserInteraction(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        port_num = 4096
        while True:
            try:
                s.bind(('', port_num))
                break
            except Exception as e:
                port_num+=1
        print(f'Listening at Port: {port_num}')

        while True:
            try:
                data, address = s.recvfrom(port_num)
                user_ip = data.decode('utf-8')
                print(f'User_Input : {user_ip}')
                user_ip = user_ip.split(',')
                qu = None
                if user_ip[0].lower() == 'top':
                    qu = self.top_param['qu']
                elif user_ip[0].lower() == 'side':
                    qu = self.side_param['qu']

                if qu != None:
                    qu.put(user_ip[1])

            except Exception as e:
                pass
    def syncThread(self):
        def add_info(data,index):
            cv.putText(data['frame'], "{} fps".format(data['fps']), (5, 15),
                       cv.FONT_HERSHEY_PLAIN, 1,
                       RED, thickness=1)
            cv.putText(data['frame'], "{}".format(index), (5, 25),
                       cv.FONT_HERSHEY_PLAIN, 1,
                       RED, thickness=1)

        top_obj_det = DETECTION_CLASS(history=self.top_param['contour']['history'],
                                      threshold=self.top_param['contour']['threshold'])
        side_obj_det = DETECTION_CLASS(history=self.side_param['contour']['history'],
                                      threshold=self.side_param['contour']['threshold'])
        top_file = os.path.join(os.getcwd(),'data',f"{self.top_param['name']}_data.bin")
        side_file = os.path.join(os.getcwd(), 'data', f"{self.side_param['name']}_data.bin")

        curr_dump_dir = os.path.join(os.getcwd(),'dumps',str(datetime.now()).replace(':','_'))
        os.makedirs(curr_dump_dir, exist_ok=True)
        os.makedirs(os.path.join(curr_dump_dir,'top'),exist_ok=True)
        os.makedirs(os.path.join(curr_dump_dir,'side'),exist_ok=True)
        index = 0

        log_fo = open(os.path.join(curr_dump_dir,'log.txt'),'w')
        while True:
            try:
                self.top_param['lock'].acquire()
                self.side_param['lock'].acquire()
                with open(top_file,'rb') as fo:
                    top_data = pickle.load(fo)

                with open(side_file, 'rb') as fo:
                    side_data = pickle.load(fo)
            except Exception as e:
                continue
            finally:
                self.top_param['lock'].release()
                self.side_param['lock'].release()

            top = top_obj_det.detect_object(frame=top_data['frame'],
                                      show_mask=self.top_param['contour']['mask'],
                                      show_controur=True)
            side = side_obj_det.detect_object(frame=side_data['frame'],
                                      show_mask=self.side_param['contour']['mask'],
                                      show_controur=True)

            add_info(top_data, index)
            add_info(side_data, index)

            if top != {} or side != {}:
                index+=1
                msg = f'{index}[{datetime.now()}]. Top[{top_data["fps"]}fps]:{top}, Bottom[{side_data["fps"]}fps]:{side}\n'
                cv.imwrite(os.path.join(curr_dump_dir,'top',f'top_{index}.jpg'),top_data['frame'])
                cv.imwrite(os.path.join(curr_dump_dir,'side',f'side_{index}.jpg'), side_data['frame'])
                log_fo.write(msg)
                log_fo.flush()

            cv.imshow(self.top_param['name'],top_data['frame'])
            cv.imshow(self.side_param['name'],side_data['frame'])

            cv.waitKey(1)

if __name__ == "__main__":
    obj = OperatorTracker()

