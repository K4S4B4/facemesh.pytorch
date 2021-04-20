import numpy as np
import torch
import cv2
import sys
import glob
import os

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazeface_landmark import BlazeFaceLandmark

from facemeshFeatParam import facemeshFeatParam

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

import onnxruntime

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

model = facemeshFeatParam()
model.eval()

########################
## FaceDetection
########################
back_detector = True
face_detector = BlazeFace(back_model=back_detector).to(gpu)
if back_detector:
    face_detector.load_weights("blazefaceback.pth")
    face_detector.load_anchors("anchors_face_back.npy")
else:
    face_detector.load_weights("blazeface.pth")
    face_detector.load_anchors("anchors_face.npy")

face_regressor = BlazeFaceLandmark().to(gpu)
face_regressor.load_weights("blazeface_landmark.pth")

########################
## FaceLandmark Onnx
########################
onnx_file_name = 'BlazeFaceFeaturemap_1_192_192_BGRxByte.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)
input_name = ort_session.get_inputs()[0].name

########################
## Input images
########################
WINDOW='test'
cv2.namedWindow(WINDOW)
if len(sys.argv) > 1:
    capture = cv2.VideoCapture(sys.argv[1])
    mirror_img = False
else:
    capture = cv2.VideoCapture(2)
    mirror_img = True

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

while hasFrame:
    frame_ct +=1

    frameRGB = np.ascontiguousarray(frame[:,:,::-1]) # BRG to RGB
    img1, img2, scale, pad = resize_pad(frameRGB)
    normalized_face_detections = face_detector.predict_on_image(img1)
    face_detections = denormalize_detections(normalized_face_detections, scale, pad)

    xc, yc, scale, theta = face_detector.detection2roi(face_detections.cpu())
    img, affine, box = face_regressor.extract_roi(frame, xc, yc, theta, scale)

    if img.shape[0] > 0:

        ### landmark detection
        img_in = img.to('cpu').detach().numpy().copy().astype(np.uint8)

        landmark, flag, params = model(img)
        landmark = landmark[0]
        params = params[0].to('cpu').detach().numpy().copy()
        print(params)

        if flag>.5:
            draw_landmarks(img_in[0], landmark[:,:2], FACE_CONNECTIONS, size=1)
    
        cv2.imshow(WINDOW, img_in[0])

        draw_roi(frame, box)
        draw_detections(frame, face_detections)

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)


cv2.destroyAllWindows()

