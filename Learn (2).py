import numpy as np
import cv2
import onnxruntime
import torch
import torch.optim as optim
import glob
from FeatToParam import FeatToParam


def train_step(train_X, train_y):
    # 訓練モードに設定
    model.train()

    # フォワードプロパゲーションで出力結果を取得
    #train_X                # 入力データ
    pred_y = model(train_X) # 出力結果
    #train_y                # 正解ラベル

    # 出力結果と正解ラベルから損失を計算し、勾配を求める
    optimizer.zero_grad()   # 勾配を0で初期化（※累積してしまうため要注意）
    loss = criterion(pred_y, train_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得
    loss.backward()   # 逆伝播の処理として勾配を計算（自動微分）

    # 勾配を使ってパラメーター（重みとバイアス）を更新
    optimizer.step()  # 指定されたデータ分の最適化を実施

    ## 正解数を算出
    #with torch.no_grad(): # 勾配は計算しないモードにする
    #    discr_y = discretize(pred_y)         # 確率値から「-1」／「1」に変換
    #    acc = (discr_y == train_y).sum()     # 正解数を計算する

    # 損失と正解数をタプルで返す
    return (loss.item())  # ※item()=Pythonの数値
    #return (loss.item(), acc.item())  # ※item()=Pythonの数値

onnx_file_name = 'BlazeFaceFeaturemap_1_192_192_BGRxByte.onnx'
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session = onnxruntime.InferenceSession(onnx_file_name, sess_options)
input_name = ort_session.get_inputs()[0].name


model = FeatToParam()

# 定数（学習方法設計時に必要となるもの）
LEARNING_RATE = 0.03   # 学習率： 0.03
REGULARIZATION = 0.03  # 正則化率： 0.03

# オプティマイザを作成（パラメーターと学習率も指定）
optimizer = optim.SGD(           # 最適化アルゴリズムに「SGD」を選択
    model.parameters(),          # 最適化で更新対象のパラメーター（重みやバイアス）
    lr=LEARNING_RATE,            # 更新時の学習率
    weight_decay=REGULARIZATION) # L2正則化（※不要な場合は0か省略）

# 変数（学習方法設計時に必要となるもの）
criterion = torch.nn.MSELoss()  # 損失関数：平均二乗誤差

########################
## Input images
########################
files =glob.glob("C:/temp/img/*.jpg")
data = np.loadtxt('C:/temp/data2.txt', delimiter=',')

batch_img = []

for fname in files:    #あとはForで1ファイルずつ実行されていく
    frame = cv2.imread(fname, cv2.IMREAD_COLOR)
    batch_img.append(frame)

batch_img_np = np.array(batch_img)

#img_in = np.expand_dims(batch_img_np, axis=0).astype(np.uint8)
img_in = batch_img_np
ort_inputs = {input_name: img_in}
ort_outs = ort_session.run(None, ort_inputs)
landmark, flag, features = ort_outs[0], ort_outs[1], ort_outs[2]

features_np = features.to('cpu').detach().numpy().copy()

np.save('C:/temp/features.npy', features_np)

#params = featToParam(features)



