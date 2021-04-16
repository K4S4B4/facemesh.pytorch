import numpy as np
import torch
import cv2
import sys
import glob
import os

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazeface_landmark import BlazeFaceLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

import onnxruntime

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_grad_enabled(False)

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
#files =glob.glob("C:\\Users\\user\\Documents\\GitHub\\Pose2d\\Saved\\VideoCaptures\\*")
files =glob.glob("C:/Users/user/Documents/GitHub/Pose3d/Saved/VideoCaptures/*.jpg")

featureList = []

for fname in files:    #あとはForで1ファイルずつ実行されていく
    frame = cv2.imread(fname, cv2.IMREAD_COLOR)

    frameRGB = np.ascontiguousarray(frame[:,:,::-1]) # BRG to RGB
    img1, img2, scale, pad = resize_pad(frameRGB)
    normalized_face_detections = face_detector.predict_on_image(img1)
    face_detections = denormalize_detections(normalized_face_detections, scale, pad)

    xc, yc, scale, theta = face_detector.detection2roi(face_detections.cpu())
    img, affine, box = face_regressor.extract_roi(frame, xc, yc, theta, scale)


    ### landmark detection
    img_in = img.to('cpu').detach().numpy().copy().astype(np.uint8)

    #cv2.imwrite("C:\\temp\\img\\" + os.path.basename(fname), img_in[0])

    ort_inputs = {input_name: img_in}
    ort_outs = ort_session.run(None, ort_inputs)
    landmark, flag, features = ort_outs[0][0], ort_outs[1][0], ort_outs[2][0]

    featureList.append(features)

    #if flag>.5:
    #    draw_landmarks(img_in[0], landmark[:,:2], FACE_CONNECTIONS, size=1)
    
    #cv2.imshow("test", img_in[0])

    #draw_roi(frame, box)
    #draw_detections(frame, face_detections)

    #key = cv2.waitKey(1)

features_np = np.array(featureList)
print(features_np.shape)
np.save('C:/temp/features1.npy', features_np)

cv2.destroyAllWindows()

