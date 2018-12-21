# coding:utf-8


from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import sys
import win32api
import random


# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# help="path to facial landmark predictor	")
# ap.add_argument("-r", "--picamera", type=int, default=-1,
# help="whether or not the Raspberry Pi camera should be used")
# args = vars(ap.parse_args())
# # initialize dlib's face detector (HOG-based) and then create
# # the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
# print(args["shape_predictor"])
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# grab the indexes of the facial landmarks
(lebStart, lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rebStart, rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(jawStart, jawEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]


from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
# import cv2
from pymouse import PyMouse
from pykeyboard import PyKeyboard
from kivy.uix.label import Label


# def cmb(foreground,background,mask):
# 	# result = np.zeros_like(foreground)
# 	result = np.zeros(foreground.shape,dtype='uint8')
# 	result[mask] = foreground[mask]
# 	inv_mask = np.logical_not(mask)
# 	result[inv_mask] = background[inv_mask]
# 	return result
def cmb(fg,bg,a):
	a1 = cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
	# bg = cv2.cvtColor(bg,cv2.COLOR_BGR2RGB)
	# fg1 = cv2.cvtColor(fg,cv2.COLOR_BGR2GRAY)
	return  fg * a1  + bg * cv2.bitwise_not(a1)


class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)
        # print('test')

    def update(self, dt):
        # print('salam')

        ret, frame = self.capture.read()
        result = frame

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        state_left = win32api.GetKeyState(0x01)
        if state_left == -128 :
            print(state_left)
            global imgbackground
            img_name = "tst2.jpg"
            cv2.imwrite(img_name, frame)
            print("written!".format(img_name))
            imgbackground = cv2.imread('tst.jpg')
            state_left = 0
        try:
            # if not rects:
            #     # STOP CAMMERA IF FACE NOT DETECTED
            #     print('face not detected!')
            #     return None

            for rect in rects:
                # print(random.randint(1,101))

                try:
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)
                except Exception as e:
                    print(e)
                m = PyMouse()
                # if m.click(0,0,1):
                    # print('salaaaaaaaaaaaaaaam')
                # print(m.position())
                x, y = m.position()
                # print(x,y)
                # if m.click(x,y,1):
                #     print(x,y)


                faceline = shape[jawStart:lebEnd]
                # print(faceline)
                imgbackground = np.zeros(frame.shape,dtype='uint8')
                imgbackground = cv2.imread('tst2.jpg')
                facelineHull = cv2.convexHull(faceline)
                mask = np.zeros(frame.shape,dtype='uint8')
                mask = cv2.drawContours(mask, [facelineHull], -1, (255 , 255 , 255),thickness=cv2.FILLED)
                mask = cv2.bitwise_not(mask)
                img2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                foreground = cv2.bitwise_and(frame,frame,mask=mask)
                result=cmb(foreground,imgbackground,mask)
                result = cv2.bitwise_not(result)

                # cv2.drawContours(frame, [facelineHull], -1, (0, 255, 0), 1)
                # print('test')
                # print('test')

        except Exception as e:
            print(e)
        if ret:
            # convert it to texture
            buf1 = cv2.flip(result, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(
                size=(result.shape[1], result.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


class CamApp(App):
    def build(self):

        # initialize the video stream and allow the cammera sensor to warmup
        print("[INFO] camera sensor warming up...")

        # vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
        time.sleep(1.0)
        self.capture = cv2.VideoCapture(0)
        # background = cv2.VideoCapture('AT.mkv')
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (640,480))
        # imgbackground = cv2.imread('tst.jpg')
        # loop over the frames from the video stream
        _,frame = self.capture.read()
        imgbackground = np.zeros(frame.shape,dtype='uint8')

        self.my_camera = KivyCamera(capture=self.capture, fps=30)
        # while True:
        #     ret , frame = self.capture.read()
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord("q"):
        #         break
        #     if key == ord(" "):
        #         global imgbackground
        #         # SPACE pressed
        #         img_name = "tst.jpg"
        #         cv2.imwrite(img_name, frame)
        #         print("written!".format(img_name))
        #         imgbackground = cv2.imread('tst.jpg')
            # return frame.all()
        return self.my_camera
    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':

    CamApp().run()
