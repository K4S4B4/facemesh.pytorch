import torch       # ライブラリ「PyTorch」のtorchパッケージをインポート
import torch.nn as nn  # 「ニューラルネットワーク」モジュールの別名定義


# 「torch.nn.Moduleクラスのサブクラス化」によるモデルの定義
class FeatToParam(nn.Module):
    def __init__(self):
        super(FeatToParam, self).__init__()
        # 層（layer：レイヤー）を定義
        self.coord_head2 = nn.Conv2d(32, 34, 3)

        #self.layer1 = nn.Linear(  # Linearは「全結合層」を指す
        #    INPUT_FEATURES,       # データ（特徴）の入力ユニット数
        #    OUTPUT_NEURONS)       # 出力結果への出力ユニット数

    def forward(self, input):
        # フォワードパスを定義
        output = self.coord_head2(input)
        output = output.reshape(-1, 34) 

        #output = activation(self.layer1(input))  # 活性化関数は変数として定義
        # 「出力＝活性化関数（第n層（入力））」の形式で記述する。
        # 層（layer）を重ねる場合は、同様の記述を続ければよい（第3回）。
        # 「出力（output）」は次の層（layer）への「入力（input）」に使う。
        # 慣例では入力も出力も「x」と同じ変数名で記述する（よって以下では「x」と書く）
        return output

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
