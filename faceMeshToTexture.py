import torch
import torch.nn as nn
import numpy as np
import cv2
import onnxruntime
import time

class OffsetToTexture(nn.Module):
    def __init__(self):
        super(OffsetToTexture, self).__init__()

        self.numOfMasks = 468
        self.sizeOfMasks = 192
        masks = []

        for i in range(self.numOfMasks):
            mask = cv2.imread("resource/facialMasks/facialMask" + str(i) + ".png")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masks.append(mask)

        masks = np.array(masks)
        self.maskTensor = torch.from_numpy(masks).float().unsqueeze(3).expand(self.numOfMasks, self.sizeOfMasks, self.sizeOfMasks, 3) / 255 #(468, H, W, 3)
        #self.maskTensor = torch.from_numpy(masks).float().unsqueeze(3) / 255 #(468, H, W, 1)
        self.maskTensorNonZero = torch.nonzero(self.maskTensor, as_tuple=True)

    # input: (468, 3)
    def forward(self, input):
        input = input.unsqueeze(1).unsqueeze(1).expand(self.numOfMasks, self.sizeOfMasks, self.sizeOfMasks, 3) #(468, 1, 1, 3)
        multiplied = torch.zeros(self.numOfMasks, self.sizeOfMasks, self.sizeOfMasks, 3)
        multiplied[self.maskTensorNonZero] = self.maskTensor[self.maskTensorNonZero] * input[self.maskTensorNonZero]
        RGB = multiplied.sum(0) #(H, W, 3)
        A = torch.ones(self.sizeOfMasks, self.sizeOfMasks, 1)
        RGBA = torch.cat((RGB,A),2)
        return RGBA

def makeInputNumpy(numOfMasks):
    input = []
    for i in range(numOfMasks):
        x = np.random.rand()
        y = np.random.rand()
        z = np.random.rand()
        input.append([x,y,z])
    input = np.array(input)
    return input

def makeInput(numOfMasks):
    input = makeInputNumpy(numOfMasks)
    input = torch.from_numpy(input).float()
    return input

def test():
    model = OffsetToTexture()
    input = makeInput(model.numOfMasks)
    output = model(input)

    output_np = output.to('cpu').detach().numpy().copy()
    output_np = cv2.resize(output_np, (512,512))
    cv2.imshow("test", output_np)
    cv2.waitKey(0)

def createOnnx():
    model = OffsetToTexture()
    input = makeInput(model.numOfMasks)

    input_names = ['offsets'] #[468, 3]
    output_names = ['texture'] #[512, 512, 3]
    onnx_file_name = "OffsetToTexture.onnx"
    torch.onnx.export(model,
                    input,
                    onnx_file_name,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=input_names, 
                    output_names=output_names
                    )

def testOnnx():
    print(onnxruntime.get_device())
    print(onnxruntime.get_available_providers())
    onnx_file_name = "OffsetToTexture.onnx"
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)
    input_name = ort_session.get_inputs()[0].name

    input = makeInputNumpy(OffsetToTexture().numOfMasks).astype(np.float32)
    ort_inputs = {input_name: input}

    start = time.time()
    for i in range(10):
        ort_outs = ort_session.run(None, ort_inputs)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    cv2.imshow("test", ort_outs[0])
    cv2.waitKey(0)

test()
createOnnx()
#testOnnx()