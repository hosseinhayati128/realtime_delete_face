
author:hossein hayati
last update : 13/9/2018

# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np

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


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor	")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# grab the indexes of the facial landmarks
(lebStart, lebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(rebStart, rebEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(jawStart, jawEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(1.0)
# background = cv2.VideoCapture('AT.mkv')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (640,480))
# imgbackground = cv2.imread('tst.jpg')
# loop over the frames from the video stream
frame = vs.read()
imgbackground = np.zeros(frame.shape,dtype='uint8')
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	# k = cv2.waitKey(1) & 0xFF
	key = cv2.waitKey(1) & 0xFF
	if key == ord(" "):
		global imgbackground
		# SPACE pressed
		img_name = "tst.jpg"
		cv2.imwrite(img_name, frame)
		print("written!".format(img_name))
		imgbackground = cv2.imread('tst.jpg')

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# frame1 = vs.read()
	# fr_background = background.read()[1]/255
	# height, width = fr_background.shape[:2]
	# fr_background = cv2.cvtColor(fr_background, cv2.COLOR_RGB2BGR)
	# fr_background = cv2.resize(fr_background, (640,480), interpolation = cv2.INTER_CUBIC)
	imgbackground = cv2.resize(imgbackground, (640,480), interpolation = cv2.INTER_CUBIC)
	# frame = imutils.resize(frame, width=400)
	# fr_background = imutils.resize(fr_background, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	# if not predictor(gray, rects):
	# 	cv2.imshow("Frame", frame)
	# 	continue

	try:
		if rects[0]:
			pass
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			try:
				shape = predictor(gray, rect)
				shape = face_utils.shape_to_np(shape)
				if shape.any():
					pass
			except:
				print('shape not found')
			# extract the face coordinates, then use the
			faceline = shape[jawStart:lebEnd]

			# compute the convex hull for face, then
			# visualize each of the face
			facelineHull = cv2.convexHull(faceline)

			mask = np.zeros(frame.shape,dtype='uint8')
			# mask = (255-mask)
			# mask = np.uint8(mask)
			# mask = mask.astype('int')
			# mask[mask == 0] = 255
			# mask[mask == 1] = 0
			mask = cv2.drawContours(mask, [facelineHull], -1, (255 , 255 , 255),thickness=cv2.FILLED)
			mask = cv2.bitwise_not(mask)
			img2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
			ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
			# frame = cv2.drawContours(frame1, [facelineHull], -1, (0, 0, 0),thickness=cv2.FILLED)
			# frame = cv2.drawContours(frame, [facelineHull], -1, (0, 255, 0))
			# mask = np.zeros(frame.shape , dtype="uint8")
			# cv2.drawContours(mask,[facelineHull],-1 , 255, -1)
			# result = cmb(fr_background,frame,mask)
			foreground = cv2.bitwise_and(frame,frame,mask=mask)
			# fr_background = fr_background.astype(foreground.dtype)
			result=cmb(foreground,imgbackground,mask)
			result = cv2.bitwise_not(result)
			# return fg * a + bg * (1-a)

			# out.write(result)


			# show the frame
			cv2.imshow("result", result)
			# cv2.imshow("Frame", frame)
			# try:
			# 	if not rects[0]:
			# 		print('nadarim')
			# except Exception as e :
			# 	cv2.destroyAllWindows()
			# 	print('nadarim')
			flag = 0

	except Exception as e:
		if flag == 0 :
			cv2.destroyAllWindows()
			flag = 1
		cv2.imshow("result", frame)

	# cv2.imshow('Frame1',frame)
	# cv2.imshow("Frame2", imgbackground)
	# key = cv2.waitKey(1) & 0xFF
	# if key == ord("p"):
	# 	# SPACE pressed
	# 	img_name = "tst.jpg"
	# 	cv2.imwrite(img_name, frame)
	# 	print("written!".format(img_name))
	# 	imgbackground = cv2.imread('tst.jpg')
	#
	# # if the `q` key was pressed, break from the loop
	# if key == ord("q"):
	# 	break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()



# def getContourStat(contour,image):
#   mask = np.zeros(image.shape,dtype="uint8")
#   cv2.drawContours(mask, [contour], -1, 255, -1,thickness=CV_FILLED)
#   mean,stddev = cv2.meanStdDev(image,mask=mask)
#   return mean, stddev
