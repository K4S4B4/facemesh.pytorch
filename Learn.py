import numpy as np
import cv2
import onnxruntime
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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

    # 損失を返す
    return loss.item()  # ※item()=Pythonの数値

def valid_step(valid_X, valid_y):
    # 評価モードに設定（※dropoutなどの挙動が評価用になる）
    model.eval()
    
    # フォワードプロパゲーションで出力結果を取得
    #valid_X                # 入力データ
    pred_y = model(valid_X) # 出力結果

    # 出力結果と正解ラベルから損失を計算
    loss = criterion(pred_y, valid_y)     # 誤差（出力結果と正解ラベルの差）から損失を取得

    # 損失を返す
    return loss.item()  # ※item()=Pythonの数値

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = FeatToParam().to(gpu)
# パラメーター（重みやバイアス）の初期化を行う
#torch.nn.init.xavier_uniform(model.coord_head2.weight)
state_dict = torch.load("model2.pth")
model.load_state_dict(state_dict)

# 定数（学習方法設計時に必要となるもの）
LEARNING_RATE = 0.005   # 学習率： 0.03
REGULARIZATION = 0  # 正則化率： 0.03

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
# 定数（学習方法設計時に必要となるもの）
BATCH_SIZE = 128  # バッチサイズ： 15（Playgroundの選択肢は「1」～「30」）
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# NumPy多次元配列からテンソルに変換し、データ型はfloatに変換する
features = np.load('C:/temp/features1.npy')
params = np.loadtxt('C:/temp/data1.txt', delimiter=',')
t_X_train = torch.from_numpy(features).float().to(gpu)
t_y_train = torch.from_numpy(params).float().to(gpu)
X_valid = np.load('C:/temp/features0.npy')
y_valid = np.loadtxt('C:/temp/data0.txt', delimiter=',')
t_X_valid = torch.from_numpy(X_valid).float().to(gpu)
t_y_valid = torch.from_numpy(y_valid).float().to(gpu)

#t_X_train = t_X_train[0:7610,:,:,:]
#t_y_train = t_y_train[0:7610,:]

# 「データ（X）」と「教師ラベル（y）」を、1つの「データセット（dataset）」にまとめる
dataset_train = TensorDataset(t_X_train, t_y_train)  # 訓練用
dataset_valid = TensorDataset(t_X_valid, t_y_valid)  # 精度検証用

# ミニバッチを扱うための「データローダー（loader）」（訓練用と精度検証用）を作成
loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
loader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE)



# 定数（学習／評価時に必要となるもの）
EPOCHS = 100             # エポック数： 100

# 変数（学習／評価時に必要となるもの）
avg_loss = 0.0           # 「訓練」用の平均「損失値」
avg_acc = 0.0            # 「訓練」用の平均「正解率」
avg_val_loss = 0.0       # 「評価」用の平均「損失値」
avg_val_acc = 0.0        # 「評価」用の平均「正解率」

# 損失の履歴を保存するための変数
train_history = []
valid_history = []

for epoch in range(EPOCHS):
    # forループ内で使う変数と、エポックごとの値リセット
    total_loss = 0.0     # 「訓練」時における累計「損失値」
    total_acc = 0.0      # 「訓練」時における累計「正解数」
    total_val_loss = 0.0 # 「評価」時における累計「損失値」
    total_val_acc = 0.0  # 「評価」時における累計「正解数」
    total_train = 0      # 「訓練」時における累計「データ数」
    total_valid = 0      # 「評価」時における累計「データ数」

    for train_X, train_y in loader_train:
        # 【重要】1ミニバッチ分の「訓練」を実行
        loss = train_step(train_X, train_y)

        # 取得した損失値と正解率を累計値側に足していく
        total_loss += loss          # 訓練用の累計損失値
        total_train += len(train_y) # 訓練データの累計数
            
    for valid_X, valid_y in loader_valid:
        # 【重要】1ミニバッチ分の「評価（精度検証）」を実行
        val_loss = valid_step(valid_X, valid_y)

        # 取得した損失値と正解率を累計値側に足していく
        total_val_loss += val_loss  # 評価用の累計損失値
        total_valid += len(valid_y) # 訓練データの累計数

    # ミニバッチ単位で累計してきた損失値や正解率の平均を取る
    n = epoch + 1                             # 処理済みのエポック数
    avg_loss = total_loss #/ n                 # 訓練用の平均損失値
    avg_val_loss = total_val_loss #/ n         # 訓練用の平均損失値

    # グラフ描画のために損失の履歴を保存する
    train_history.append(avg_loss)
    valid_history.append(avg_val_loss)

    # 損失や正解率などの情報を表示
    print(f'[Epoch {epoch+1:3d}/{EPOCHS:3d}]' \
          f' loss: {avg_loss:.5f}, ' \
          f' val_loss: {avg_val_loss:.5f}')
          #f' loss: {avg_loss:.5f}, acc: {avg_acc:.5f}') \
          #f' val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')

print('Finished Training')
print(model.state_dict())  # 学習後のパラメーターの情報を表示

torch.save(model.state_dict(), "model3.pth")