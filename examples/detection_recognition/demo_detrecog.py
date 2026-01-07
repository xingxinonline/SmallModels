#!/usr/bin/python3
from __future__ import print_function

import torch
import torch.backends.cudnn as cudnn
import cv2
import time
import math

from model.yufacedetectnet import YuFaceDetectNet
from model.model_modify import ShuffleFaceNet

import config
from utils import load_model
from utils import FaceDetect
from utils import FaceRecog
from utils import DrawImage
from utils import updateVector


if __name__ == '__main__':
    device = torch.device(config.device)
    torch.set_grad_enabled(False)
    cudnn.benchmark = True

    #detect model init
    detect_model = YuFaceDetectNet(phase='test', size=None )
    detect_model = load_model(detect_model, config.detect_model_path, True)
    detect_model.eval()
    detect_model = detect_model.to(device)

    #recognition model init
    recog_model = ShuffleFaceNet()
    recog_model = load_model(recog_model,config.recog_model_path,True)
    recog_model.eval()
    recog_model = recog_model.to(device)


    #init vector
    updateVector(detect_model,recog_model)
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    # count = 0
    lasttime = time.time()

    while 1:
        ret, img_raw = cap.read()

        img_raw = cv2.resize(img_raw, (320, 240))
        detect_result = FaceDetect(detect_model,img_raw)

        recog_result = FaceRecog(recog_model,img_raw,detect_result)
        imgresult = DrawImage(img_raw,recog_result,detect_result)

        cv2.imshow("PIMface", imgresult)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()