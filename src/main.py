import torch
import torch.optim as optim
import torch.nn as nn
import utils
import datetime
import sys
from config import DEVICE, BATCH_SIZE, LR, EPOCHS, IMAGE_DIRS, TRANSFORM, display_config
from dataset import MyDataset
from model import MyCNN

# 乱数シードの初期化
utils.set_seed()

# 設定情報の表示
display_config()

# 学習のデータローダーの作成
train_dataset = MyDataset(IMAGE_DIRS["train"],TRANSFORM)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# テストのデータローダーの取得
test_dataset = MyDataset(IMAGE_DIRS["test"],TRANSFORM)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# モデルの初期化
num_classes = len(IMAGE_DIRS["train"])
model = MyCNN(num_classes).to(DEVICE)

# 損失関数の設定
criterion = nn.CrossEntropyLoss().to(DEVICE)

# 最適化関数の設定
optimizer = optim.Adam(model.parameters(), lr=LR)

# 結果を保存するディレクトリの作成
result_dir = utils.create_result_directory()

# 設定ファイルの保存
utils.save_config(result_dir)

# 結果を保存するためのCSVファイルの初期化
csv_columns = ['Epoch', 'Loss', 'Accuracy', 'Test_Accuracy', 'time']
csv_file = utils.save_results_to_csv(result_dir, csv_columns)

# 学習ループ開始
print("Training Started!")

min_running_loss = sys.float_info.max

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
     
    ''' 学習ステップ '''

    # 学習モードに変更
    model.train()

    # エポック開始時の時刻を記録
    epoch_start_time = datetime.datetime.now()

    # 学習
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # エポック終了時の時刻を記録
    epoch_end_time = datetime.datetime.now()    


    ''' 評価ステップ '''

    # 評価モードに変更
    model.eval()

    # 学習データでの精度計算
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total

    # テストデータでの精度計算
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_accuracy = 100 * correct / total

    # エポックごとの結果を表示
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")

    # エポックごとの結果をCSVに保存
    utils.save_epoch_results(csv_file, epoch, running_loss, train_loader, train_accuracy, test_accuracy, epoch_start_time, epoch_end_time)

    # 最良モデルを保存
    min_running_loss = utils.save_best_model(model, running_loss, min_running_loss, result_dir)


print("Training Finished!")
