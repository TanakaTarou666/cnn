import torch
import os
import shutil
import random
import csv
import datetime  # 追加
from config import SEED, BASE_RESULT_DIR, RESULT_DIR

# 乱数シードの初期化
def set_seed():
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # cudnn の再現性を確保するための設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 結果を保存するためのディレクトリを作成
def create_result_directory():
    result_dir = BASE_RESULT_DIR + RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

# 設定ファイルのコピー
def save_config(result_dir):
    shutil.copy('./config.py', result_dir)

# 結果を保存するためのCSVファイルを初期化
def save_results_to_csv(result_dir, csv_columns):
    csv_file = result_dir + 'result.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_columns)
    return csv_file

# エポックごとの結果をCSVファイルに書き込む
def save_epoch_results(csv_file, epoch, running_loss, train_loader, train_accuracy, test_accuracy, start_time, end_time):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        # 時刻を文字列に変換（フォーマット例：YYYY-MM-DD HH:MM:SS）
        time_taken = end_time - start_time
        writer.writerow([epoch+1, running_loss/len(train_loader), train_accuracy, test_accuracy, time_taken])

# モデルを保存
def save_best_model(model, loss, best_loss, result_dir):
    if loss < best_loss: 
        best_loss = loss
        model_save_path = os.path.join(result_dir, f'best_model.pth')
        torch.save(model.state_dict(), model_save_path)
    return best_loss
