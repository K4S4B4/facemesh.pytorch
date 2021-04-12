import numpy as np
import cv2
import onnxruntime

def resize_pad(img):
    """ resize and pad images to be input to the detectors

    The face and palm detector networks take 256x256 and 128x128 images
    as input. As such the input image is padded and resized to fit the
    size while maintaing the aspect ratio.

    Returns:
        img1: 256x256
        img2: 128x128
        scale: scale factor between original image and 256x256 image
        pad: pixels of padding in the original image
    """

    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = 512
        w1 = 512 * size0[1] // size0[0]
        padh = 0
        padw = 512 - w1
        scale = size0[1] / w1
    else:
        h1 = 512 * size0[0] // size0[1]
        w1 = 512
        padh = 512 - h1
        padw = 0
        scale = size0[0] / h1
    padh1 = padh//2
    padh2 = padh//2 + padh%2
    padw1 = padw//2
    padw2 = padw//2 + padw%2
    img1 = cv2.resize(img, (w1,h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)))
    pad = (int(padh1 * scale), int(padw1 * scale))
    img2 = cv2.resize(img1, (128,128))
    return img1, img2, scale, pad

# Vertex indices can be found in
# github.com/google/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualisation.png
# Found in github.com/google/mediapipe/python/solutions/face_mesh.py
FACE_CONNECTIONS = [
    # Lips.
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0),
    (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    # Left eye.
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380),
    (380, 381), (381, 382), (382, 362), (263, 466), (466, 388),
    (388, 387), (387, 386), (386, 385), (385, 384), (384, 398),
    (398, 362),
    # Left eyebrow.
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293),
    (293, 334), (334, 296), (296, 336),
    # Right eye.
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153),
    (153, 154), (154, 155), (155, 133), (33, 246), (246, 161),
    (161, 160), (160, 159), (159, 158), (158, 157), (157, 173),
    (173, 133),
    # Right eyebrow.
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105),
    (105, 66), (66, 107),
    # Face oval.
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251),
    (251, 389), (389, 356), (356, 454), (454, 323), (323, 361),
    (361, 288), (288, 397), (397, 365), (365, 379), (379, 378),
    (378, 400), (400, 377), (377, 152), (152, 148), (148, 176),
    (176, 149), (149, 150), (150, 136), (136, 172), (172, 58),
    (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109),
    (109, 10)
]

def draw_landmarks(img, points, connections=[], color=(0, 255, 0), size=2):
    points = points[:,:2]
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        cv2.circle(img, (x, y), size, color, thickness=size)
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        cv2.line(img, (x0, y0), (x1, y1), (0,0,0), size)

onnx_file_name = 'resource/MediaPipe/BlazeFace_B_192_192_BGRxByte.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#sess_options.enable_profiling = True
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)

input_name = ort_session.get_inputs()[0].name

WINDOW='test'
cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(2)
hasFrame, frame = capture.read()

while hasFrame:
    img1, img2, scale, pad = resize_pad(frame)
    img = cv2.resize(img1, (192,192))

    #img = cv2.imread("test_eye.jpg")
    #img = cv2.resize(img, (64, 64))
    img_in = np.expand_dims(img, axis=0).astype(np.uint8)
    ort_inputs = {input_name: img_in}

    ort_outs = ort_session.run(None, ort_inputs)

    landmark, flag = ort_outs[0][0], ort_outs[1][0]
    if flag>.5:
        draw_landmarks(img1, landmark[:,:2] * 2.6666, FACE_CONNECTIONS, size=1)

    # left eye visibi
    x1 = landmark[243, 0] - landmark[130, 0]
    y1 = landmark[243, 1] - landmark[130, 1]
    z1 = landmark[243, 2] - landmark[130, 2]


    cv2.imshow(WINDOW, img1)
    cv2.waitKey(1)

    hasFrame, frame = capture.read()

#torch.onnx.export(
#    net, 
#    (torch.randn(1,3,64,64, device=gpu), ), 
#    "irislandmarks.onnx",
#    input_names=("image", ),
#    output_names=("preds", "conf"),
#    opset_version=9
#)