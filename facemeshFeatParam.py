import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義
from facemesh import FaceMesh
from FeatToParam import FeatToParam

# 「torch.nn.Moduleクラスのサブクラス化」によるモデルの定義
class facemeshFeatParam(nn.Module):
    def __init__(self):
        super(facemeshFeatParam, self).__init__()

        self.facemeshBase = FaceMesh()
        self.facemeshBase.load_weights("facemesh.pth")

        self.featureToParam = FeatToParam()
        self.featureToParam.load_weights("model2.pth")

    def forward(self, input):
        r, c, features = self.facemeshBase(input)
        params = self.featureToParam(features)
        return r, c, params