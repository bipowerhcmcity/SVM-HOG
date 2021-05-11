import cv2
import numpy as np
import HOG.HOG1
import os
import glob
from sklearn import svm
from imutils.video import VideoStream
import joblib
import time


# class_img= ["pos","neg"]
# y=[]
# features_arr = np.empty(shape=[0,3780],dtype=float)
# for i in range(len(class_img)):
#         img_dir = "persons/" + class_img[i]  # Enter Directory of all images
#         data_path = os.path.join(img_dir, '*')
#         files = glob.glob(data_path)
#         for j in files:
#             feature = HOG.HOG1.Hog1(cv2.imread(j))
#             features_arr = np.append(features_arr,[feature],axis=0)
#             if (class_img[i] == "pos"):
#                 y.append(1)
#             else:
#                 y.append(-1)
#
# print(y)
# print(features_arr.shape)
#
# clf = svm.SVC(kernel='linear')
# print(np.where(features_arr >= np.finfo(np.float64).max))
# clf.fit(features_arr,y)
#joblib.dump(clf,'model.pkl')


# load model :
clf = joblib.load("model.pkl")

# feature = HOG.HOG1.Hog1(cv2.imread("nomaask.jpg"))
# print("Result: ",clf.predict([feature]))

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
while(True):
    frame = vs.read()

    frame = cv2.resize(frame, (800, 400))

    feature = HOG.HOG1.Hog1(frame)

    print("Result: ",clf.predict([feature]))

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cv2.imwrite("Frame.png", frame)
        break