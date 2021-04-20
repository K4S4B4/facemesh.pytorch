import torch       # ���C�u�����uPyTorch�v��torch�p�b�P�[�W��C���|�[�g
import torch.nn as nn  # �u�j���[�����l�b�g���[�N�v���W���[���̕ʖ���`
from facemesh import FaceMesh
from FeatToParam import FeatToParam

# �utorch.nn.Module�N���X�̃T�u�N���X���v�ɂ�郂�f���̒�`
class facemeshFeatParam(nn.Module):
    def __init__(self):
        super(facemeshFeatParam, self).__init__()

        self.facemeshBase = FaceMesh()
        self.facemeshBase.load_weights("facemesh.pth")

        self.featureToParam = FeatToParam()
        self.featureToParam.load_weights("model3_b256_L0.001_Adam_E5.pth")

    def forward(self, input):
        r, c, features = self.facemeshBase(input)
        params = self.featureToParam(features)
        return r, c, params