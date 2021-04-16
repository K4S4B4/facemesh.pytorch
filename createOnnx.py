import numpy as np
import torch
import cv2
from facemesh import FaceMesh

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = FaceMesh().to(gpu)
net.load_weights("facemesh.pth")

##############################################################################
batch_size = 1
height = 192
width = 192
x = torch.randn((batch_size, height, width, 3), requires_grad=True).byte().to(gpu)
##############################################################################

input_names = ["input"] #[B,192,192,3],
output_names = ['landmark', 'confidence', 'features'] #[B,486,3], [B]


onnx_file_name = "BlazeFaceFeaturemap_{}_{}_{}_BGRxByte.onnx".format(batch_size, height, width)
dynamic_axes = {
    "input": {0: "batch_size"}, 
    "landmark": {0: "batch_size"}, 
    "confidence": {0: "batch_size"},
    "features": {0: "batch_size"}
    }

torch.onnx.export(net,
                x,
                onnx_file_name,
                export_params=True,
                opset_version=9,
                do_constant_folding=True,
                input_names=input_names, 
                output_names=output_names
                ,dynamic_axes=dynamic_axes
                )
