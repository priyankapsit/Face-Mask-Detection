
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from cv2 import cv2
import threading
import os
import sys

#set Window Screen
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


# Initializing for video stream
def blackbox():
    time.sleep(1)

    thread = threading.Thread(target=VideoStream)
    thread.start()
    
    eli_count = 0
    while thread.is_alive():
        print('Loading', '.'*(eli_count+1), ' '*(2-eli_count), end='\r')
        eli_count = (eli_count + 1) % 3
        time.sleep(0.1)
    sys.stdout.flush()
    thread.join()
    print("Done               ")

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > args["confidence"]:
            #computer x,y coordianates
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		''' for faster inference we'll make batch predictions on *all*
		 faces at the same time rather than one-by-one predictions
		 in the above `for` loop'''
		preds = maskNet.predict(faces)

	return (locs, preds)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
# prototxtPath = os.path.sep.join([args["face"],"deploy.prototxt"])
prototxtPath = "deploy.prototxt"
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)
time.sleep(2)
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")

maskNet = load_model(args["model"])
blackbox()
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = ResizeWithAspectRatio(frame, width=1280)
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    print(locs,"  ",preds,end='   ')
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred
        print("pred ",pred)
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 150)
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # show the output frame
    cv2.imshow("Face Mask Detector Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        chose = input("Confirm to Exit [Y or N]")
        if chose == "Y" or chose == "y": break
cv2.destroyAllWindows()
vs.stop()