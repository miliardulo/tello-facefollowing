import sys
import time
from datetime import datetime

import cv2
import numpy as np
from djitellopy import Tello

facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
FRAMEWIDTH, FRAMEHEIGHT = (360, 240)


def findface(img, cascade):
    if facecascade.empty():
        print("Cascade XML failed to load.")
        sys.exit()
    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facecascade.detectMultiScale(imggray, 1.3)
    centers = []
    areas = []
    # hFrame, wFrame, _ = img.shape
    # cfx = wFrame // 2
    # cfy = hFrame // 2
    for (x, y, w, h) in faces:
        cx = x + (w // 2)
        cy = y + (h // 2)
        area = w * h
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        # cv2.arrowedLine(img, (cx, cy), (cfx, cfy), (0, 255, 0), 2)
        # cv2.circle(img, (cfx, cfy), 5, (255, 255, 0), cv2.FILLED)
        centers.append([cx, cy])
        areas.append(area)
    if len(areas) > 0:
        i = areas.index(max(areas))
        return img, [centers[i], areas[i]]
    else:
        return img, [[-1, -1], -1]


LOOPTIME = 0.029


# def stdPID(pidparams, input, previnput, iterm, setpoint):
#     kp, ki, kd = pidparams
#     err = setpoint - input
#     pterm = kp * err
#     iterm += ki * err * LOOPTIME
#     dterm = 1.0 * kd * (input - previnput) / LOOPTIME
#     previnput = input
#     output = pterm + iterm + dterm
#     print(setpoint, input)
#     return [output, previnput, iterm]


def mypd(params, input, preverror, setpoint):
    error = setpoint - input
    output = params[0] * error + params[1] * (error - preverror)
    return output, error


YAWPID = [0.2, 0.1, 0.05]
previnput = -1
iterm = 0
perr = 0


def calcSpeed(frame, face):
    global previnput, iterm, perr
    if face[1] < 0:
        return 0
    cx, cy = face[0]
    framecenterX, _, _ = frame.shape
    framecenterX = framecenterX // 2
    # if previnput < 0:
    #     yawspeed, previnput, iterm = stdPID(YAWPID, cx, cx, 0, framecenterX)
    # else:
    #     yawspeed, previnput, iterm = stdPID(YAWPID, cx, previnput, iterm, framecenterX)
    yawspeed, perr = mypd(cx, perr)
    yawspeed = int(np.clip(yawspeed, -100, 100))
    return yawspeed


def sendcmd(tello: Tello, fb, yaw):
    tello.send_rc_control(0, fb, 0, yaw)


def telloinit(te: Tello):
    te.connect()
    print("Battery: {}%".format(te.get_battery()))
    te.streamon()
    te.takeoff()
    te.send_rc_control(0, 0, 25, 0)
    time.sleep(2)
    return te


def telloshutdown(te: Tello):
    te.streamoff()
    te.land()
    te.end()


if __name__ == "__main__":
    # vidsrc = cv2.VideoCapture(0)
    te = Tello()
    telloinit(te)
    looptimequeue = []
    while True:
        start = time.time()
        # _, frame = vidsrc.read()
        frame = te.get_frame_read().frame
        frame = cv2.resize(frame, (FRAMEWIDTH, FRAMEHEIGHT))
        try:
            frame, mdata = findface(frame, facecascade)
        except TypeError:
            print("*", mdata)
        rot = calcSpeed(frame, mdata)
        cv2.putText(frame, str(rot), (40, 40), 2, 1.2, (0, 0, 0))
        cv2.imshow("Test", frame)
        if rot != 0:
            sendcmd(te, 0, rot)
        stop = time.time()
        looptimequeue.append(stop - start)
        if cv2.waitKey(1) == ord("q"):
            break
    # vidsrc.release()
    cv2.destroyAllWindows()
    telloshutdown(te)
    print(np.median(looptimequeue))
