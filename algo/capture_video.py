from algo.calibration import *
from datetime import datetime
import traceback
from algo.detection import *

class CaptureProcess:
    def __init__(self, param):
        self.param = param
        self.main_task(param)

    def main_task(self,param):
        video = cv.VideoCapture(param['cam_id'], cv.CAP_DSHOW)
        video.set(3, param['resolution'][0])
        video.set(4, param['resolution'][1])
        cal = Calibrate_v2(name=param['name'],rotate=param['rotate'],cal_type=param['cal_type'])
        fps = FPS()

        print(f'{param["name"]} Cam Starting')
        try:
            while True:
                try:
                    task = param['qu'].get(0)
                    if list(task.lower())[0] == 'c': # Only because of the way packet was being sent
                        cal.cal_state = CalState.SelectArea
                except Exception as e:
                    # print("capture_video.py, Calibration call failed ->", e)
                    pass
                ret, frame = video.read()

                if ret == True:
                    frame = cv.resize(frame,(720,300),interpolation=cv.INTER_CUBIC)
                    image = cal.keyHandler(frame)
                    data = {}
                    data['frame'] = image
                    data['dt'] = datetime.now()
                    data['fps'] = fps.updateFps()

                    param['lock'].acquire()
                    file = os.path.join(os.getcwd(), 'data', f"{param['name']}_data.bin")
                    fo = open(file,'wb')
                    pickle.dump(data,fo)
                    fo.close()
                    param['lock'].release()
                    # cv.imshow(param['name'],frame)
                else:
                    print(f"{param['name']}:No frame captured")

        except Exception as e:
            traceback.print_exc()
            print(f"{param['name']} capture: pickling or date time failure -> {e}")
