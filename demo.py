import numpy as np
import torch
import cv2
import sys

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazeface_landmark import BlazeFaceLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

import onnxruntime

def draw_landmarks_shift(img, points, shift, connections=[], color=(0, 255, 0), size=2):
    points = points[:,:2]
    for point in points:
        x, y = point
        x = int(x) + shift[0]
        y = int(y) + shift[1]
        cv2.circle(img, (x, y), size, color, thickness=size)

def draw_landmarks_HV(img, X, Y, shift, connections=[], color=(0, 255, 0), size=2):
    for i in range(X.size):
        x = int(X[i]) + shift[0]
        y = int(Y[i]) + shift[1]
        cv2.circle(img, (x, y), size, color, thickness=size)

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
WINDOW='test'
cv2.namedWindow(WINDOW)
if len(sys.argv) > 1:
    capture = cv2.VideoCapture(sys.argv[1])
    mirror_img = False
else:
    capture = cv2.VideoCapture(0)
    mirror_img = True

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

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
        #flags, normalized_landmarks = face_regressor(img.to(gpu))
        img_in = img.to('cpu').detach().numpy().copy().astype(np.uint8)
        #img_in = np.expand_dims(img, axis=0).astype(np.uint8)
        ort_inputs = {input_name: img_in}
        ort_outs = ort_session.run(None, ort_inputs)
        landmark, flag, features = ort_outs[0][0], ort_outs[1][0], ort_outs[2][0]

        img_out = img_in[0]/255 * 0.1
        img_out = cv2.resize(img_out, (512,512))

        img_outXZ = img_out.copy()
        landmarkXZ = landmark[:,[0,2,1]].copy()

        img_outZY = img_out.copy()
        landmarkZY = landmark[:,[2,1,0]].copy()

        img_outHV = img_out.copy()


        if flag>.75:
            draw_landmarks(img_out, landmark[:,:2] *2.6666, FACE_CONNECTIONS, size=1)

        if flag>.75:
            #draw_landmarks_shift(img_outXZ, landmarkXZ[:,:2] *2.6666, (0, 96), FACE_CONNECTIONS, size=1)
            draw_landmarks_HV(img_outXZ, landmark[:,0]*2.6666, -landmark[:,2]*2.6666, (0, 294), FACE_CONNECTIONS, size=1)

        if flag>.75:
            #draw_landmarks_shift(img_outZY, landmarkZY[:,:2] *2.6666, (96, 0), FACE_CONNECTIONS, size=1)
            draw_landmarks_HV(img_outZY, -landmark[:,2]*2.6666, landmark[:,1]*2.6666, (294, 0), FACE_CONNECTIONS, size=1)
    
        cv2.imshow(WINDOW, img_out)
        cv2.imshow("XZ", img_outXZ)
        cv2.imshow("ZY", img_outZY)

        p = landmark
        Center = np.mean(p, 0)
        #H = (p[189]+p[245]+p[128])/3 - (p[113]+p[226]+p[31])/3 # right Eye
        #H = (p[342]+p[446]+p[261])/3 - (p[413]+p[465]+p[357])/3 # left Eye
        H = (p[386]+p[374]+p[385]+p[380])/4 - (p[159]+p[145]+p[158]+p[153])/4
        Hmag =  np.linalg.norm(H)
        H = H / Hmag
        V = (p[10]+p[338]+p[109])/3 - (p[152]+p[377]+p[148])/3
        Vmag = np.linalg.norm(V)
        V = V / Vmag
        D = np.cross(H, V)
        D = D / np.linalg.norm(D)
        V = np.cross(H, D)


        Hcoord = np.dot(p, H)
        Vcoord = np.dot(p, V)

        angle = -np.dot(D, [0,0,1])
        #print(angle)


        if flag>.75:
            draw_landmarks_HV(img_outHV, Hcoord*2.6666, Vcoord*2.6666, (0, 0), FACE_CONNECTIONS, size=1)
        cv2.imshow("HV", img_outHV)

        #print( (Vcoord[168]-Vcoord[1])/Vmag )

        #print(Hmag, Vmag)

        #laugh = np.dot( (p[164]+p[167]+p[393])/3 - (p[287]+p[432]+p[410] + p[57]+p[212]+p[186])/6, V) / Vmag
        #print(laugh)


        #laughR_B = np.dot( p[234] - p[185], H) / Hmag
        #laughR = np.dot( p[187] - p[185], H) / Hmag


        ## OK 角度による誤差大きい
        browLateral = (p[336]+p[285]+p[295])/6 - (p[107]+p[55]+p[65])/6
        #browLateral = (p[336]+p[285]+p[296]+p[295]+p[334]+p[282])/6 - (p[107]+p[55]+p[66]+p[65]+p[105]+p[52])/6
        eyeLateral = (p[386]+p[374]+p[385]+p[380]+p[387]+p[373])/6 - (p[159]+p[145]+p[158]+p[153]+p[160]+p[144])/6
        browLateral = np.linalg.norm(browLateral) / np.linalg.norm(eyeLateral)
        #print(browLateral)

        ## OK
        noseW = (p[279]+p[331]+p[294]+p[278])/4-(p[49]+p[102]+p[64]+p[48])/4
        noseW =  np.linalg.norm(noseW) / np.linalg.norm(eyeLateral)
        #print(noseW)

        ## OK
        mouthW = (p[409]+p[375]+p[291]+p[287])/4-(p[185]+p[146]+p[61]+p[57])/4
        mouthW =  np.linalg.norm(mouthW) / np.linalg.norm(eyeLateral)
        #print(mouthW)

        mouthV = (p[12]+p[13]+p[82]+p[312])/4-(p[15]+p[14]+p[87]+p[317])/4
        eyeToMouth = (p[409]+p[375]+p[291]+p[287]+p[185]+p[146]+p[61]+p[57])/8 - (p[386]+p[374]+p[385]+p[380]+p[387]+p[373] + p[159]+p[145]+p[158]+p[153]+p[160]+p[144])/12
        mouthV =  np.linalg.norm(mouthV) / np.linalg.norm(eyeToMouth)

        ## OK 角度による誤差大きい
        browVert = (p[336]+p[285]+p[296]+p[295]+p[334]+p[282] + p[107]+p[55]+p[66]+p[65]+p[105]+p[52])/12
        eyeVert = (p[386]+p[374]+p[385]+p[380]+p[387]+p[373] + p[159]+p[145]+p[158]+p[153]+p[160]+p[144])/12
        faceUpperRim = (p[10]+p[338]+p[109]+p[297]+p[67]+p[332] + p[103])/7
        browVert = np.linalg.norm(browVert - eyeVert) / np.linalg.norm(faceUpperRim - eyeVert)
        #print(browVert)

        print(angle, noseW, mouthW, mouthV, browVert, browLateral)


        #laughR_B = p[234]-p[185]
        #laughR = p[187]-p[185]
        #laughR = np.dot( laughR, H) / Hmag
        #laughR = np.linalg.norm(laughR)
        #laughR_B = np.linalg.norm(laughR_B)
        #print(laughR/laughR_B, laughR, laughR_B)

        #z = np.dot(p[0] - Center, D)
        #print(z)

        #for i in range(len(flags)):
        #    landmark, flag = landmarks[i], flags[i]
        #    if flag>.5:
        #        draw_landmarks(frame, landmark[:,:2], FACE_CONNECTIONS, size=1)


        draw_roi(frame, box)
        draw_detections(frame, face_detections)

        #cv2.imshow(WINDOW, frame[:,:,::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()

