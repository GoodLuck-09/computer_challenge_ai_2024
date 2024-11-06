#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import sys
import pandas as pd
from other_code_files.deal_data import deal_test_data
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# 以上为依赖包引入部分, 请根据实际情况引入
# 引入的包需要安装的, 请在requirements.txt里列明, 最好请列明版本

# 以下为逻辑函数, main函数的入参和最终的结果输出不可修改
def main(to_pred_dir, result_save_path):

    run_py = os.path.abspath(__file__)
    model_dir = os.path.dirname(run_py)

    to_pred_dir = os.path.abspath(to_pred_dir)
    testa_csv_path = os.path.join(to_pred_dir, "testa_x", "testa_x.csv")
    testa_html_dir = os.path.join(to_pred_dir, "testa_x", "html")
    testa_image_dir = os.path.join(to_pred_dir, "testa_x", "image")

    deal_test_data(testa_csv_path, testa_html_dir, testa_image_dir, model_dir)

    # 1. 加载数据集
    train_df = pd.read_csv(model_dir + '/other_code_files/dealed_train_data.csv')
    test_df = pd.read_csv(model_dir + '/other_code_files/dealed_test_data.csv')

    # 2. 数据预处理
    # 编码标签
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['label'])
    train_labels = torch.tensor(train_labels).long()

    # 文本预处理：构建词汇表
    word_counts = Counter(" ".join(train_df['text']).split())
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: i + 1 for i, word in enumerate(vocab)}  # 从1开始，0留给填充

    # 将文本转换为整数序列
    def text_to_seq(text):
        return [vocab_to_int[word] for word in text.split() if word in vocab_to_int]

    # 文本转换为整数序列
    train_text_ints = [text_to_seq(text) for text in train_df['text']]
    test_text_ints = [text_to_seq(text) for text in test_df['text']]

    # 填充序列以匹配最长序列长度
    max_seq_length = max(len(seq) for seq in train_text_ints)
    train_text_ints = [seq + [0] * (max_seq_length - len(seq)) for seq in train_text_ints]
    test_text_ints = [seq + [0] * (max_seq_length - len(seq)) for seq in test_text_ints]

    # 划分数据集为训练集和验证集
    train_text_ints, val_text_ints, train_labels, val_labels = train_test_split(train_text_ints, train_labels,
                                                                                test_size=0.2, random_state=42)

    # 创建自定义数据集
    class NewsDataset(Dataset):
        def __init__(self, text_ints, labels=None):
            self.text_ints = torch.tensor(text_ints).long()
            self.labels = labels if labels is not None else None

        def __len__(self):
            return len(self.text_ints)

        def __getitem__(self, idx):
            return self.text_ints[idx] if self.labels is None else (self.text_ints[idx], self.labels[idx])

    # 创建训练集、验证集和测试集
    train_dataset = NewsDataset(train_text_ints, train_labels)
    val_dataset = NewsDataset(val_text_ints, val_labels)
    test_dataset = NewsDataset(test_text_ints)

    # 创建DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 3. 创建LSTM模型
    class LSTMModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout):
            super(LSTMModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, text):
            embedded = self.dropout(self.embedding(text))
            lstm_out, (hidden, cell) = self.lstm(embedded)
            hidden = self.dropout(hidden[-1])
            out = self.fc(hidden)
            return out

    # 模型参数
    vocab_size = len(vocab_to_int) + 1  # +1 for padding
    embedding_dim = 100
    hidden_dim = 128
    output_dim = 1
    num_layers = 2
    dropout = 0.5

    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)

    # 4. 训练模型
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        for texts, labels in train_loader:
            outputs = model(texts).squeeze(1)
            loss = criterion(outputs, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for texts, labels in val_loader:
                outputs = model(texts).squeeze(1)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                val_loss = val_loss / len(val_loader)

        # print(f'Epoch {epoch + 1}/{num_epochs} - Training Loss: {loss.item()}, Validation Loss: {val_loss}')

    # 5. 保存模型
    torch.save(model.state_dict(), model_dir +'/other_code_files/lstm_model.pth')   

    # 6. 加载模型
    loaded_model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers, dropout)
    loaded_model.load_state_dict(torch.load(model_dir + '/other_code_files/lstm_model.pth'))
    loaded_model.eval()

    # 7. 对测试集进行预测
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    predicted_labels_list = []
    with torch.no_grad():
        for texts in test_loader:
            outputs = loaded_model(texts).squeeze(1)
            probs = torch.sigmoid(outputs)
            # 将概率四舍五入到最接近的整数，得到预测的标签
            predicted_labels = torch.round(probs)
            predicted_labels_list.extend(predicted_labels)
    res_labels = [int(i) for i in predicted_labels_list]
    # 将预测结果转换为原始标签

    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
    # 以下区域为预测逻辑代码, 下面的仅为示例
    # 请选手根据实际模型预测情况修改

    #=-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==-=-=-=-==
    # pred_result_dict = {'id': test_df['id'], 'label': res_labels}
    # res_df = pd.DataFrame(pred_result_dict)
    # res_df.to_csv(result_save_path, mode='w', index=False, header=True)
    # 将预测结果保存到DataFrame
    testa = pd.read_csv(testa_csv_path)
    testa_length = len(testa)
    res_dict = {}
    res_dict['id'] = [i for i in range(testa_length)]
    res_dict['label'] = res_labels
    res_df = pd.DataFrame(res_dict)
    res_df.to_csv(result_save_path, mode='w', index=False, header=True)

if __name__ == "__main__":
    # 以下代码请勿修改, 若因此造成的提交失败由选手自负
    to_pred_dir = sys.argv[1]  # 所需预测的文件夹路径
    result_save_path = sys.argv[2]  # 预测结果保存文件路径，已指定格式为csv
    main(to_pred_dir, result_save_path)
    # to_pred_dir = '../test_'  # 所需预测的文件夹路径
    # result_save_path = '../result'   # 预测结果保存文件路径，已指定格式为csv
    # main(to_pred_dir, result_save_path)