import torch.nn as nn
from transformers import AdamW
from torch.utils.data import Dataset
import pandas as pd
import torch
import os
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader
#a = np.array(L1)list转换为ndarray
#B = A.detach().numpy()  # tensor转换为ndarray
#C = torch.from_numpy(B) # ndarray转换为tensor

# 超参数
hidden_dropout_prob = 0.3
num_labels = 20
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())


class mymodel (torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(20,2)

    def forward(self,input):


        out = self.fc(input)
        # out = out.softmax(dim = 1)
        return out

model2 = mymodel()
model2.to(device)
#数据集类,加载数据和特征1
class  HCdataset(Dataset):

    def __init__(self,human_file_path,chatgpt_file_path):
        self.human_file_path = human_file_path
        self.chatgpt_file_path = chatgpt_file_path
        self.total_text = []
        self.total_label1 = []

        with open(self.human_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.total_text.append(line)
                self.total_label1.append(0)
        f.close()
        with open(self.chatgpt_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.total_text.append(line)
                self.total_label1.append(1)
        f.close()
    def __getitem__(self, index):

        text = self.total_text[index]
        label1 = self.total_label1[index]
        sample = {"text": text, "label": label1}

        return sample

    def __len__(self):
        return len(self.total_text)

# dataset= HCdataset()
train_ag_human_path = "../data/AGdata/train/human.txt"
train_ag_chatgpt_path ="../data/AGdata/train/chatgpt.txt"
train_hc_human_path = "../data/HCdata/train/human.txt"
train_hc_chatgpt_path = "../data/HCdata/train/chatgpt.txt"

test_ag_human_path = "../data/AGdata/test/human.txt"
test_ag_chatgpt_path = "../data/AGdata/test/chatgpt.txt"
test_hc_human_path = "../data/HCdata/test/human.txt"
test_hc_chatgpt_path = "../data/HCdata/test/chatgpt.txt"

train_ag_dataset = HCdataset(train_ag_human_path,train_ag_chatgpt_path)
test_ag_dataset = HCdataset(test_ag_human_path,test_ag_chatgpt_path)
train_hc_dataset = HCdataset(train_hc_human_path,train_hc_chatgpt_path)
test_hc_dataset = HCdataset(test_hc_human_path,test_hc_chatgpt_path)
train_dataset = train_ag_dataset+train_hc_dataset
test_dataset = test_ag_dataset+test_hc_dataset


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
# 加载验证集

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)




# 定义 tokenizer，传入词汇表
tokenizer = BertTokenizer.from_pretrained(
    pretrained_model_name_or_path="bert-base-uncased",
    cache_dir=None,
    force_download=False
)


# 加载模型
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
model.to(device)

model.load_state_dict(torch.load("pretrained_model.pth"))


# 定义优化器和损失函数
# Prepare optimizer and schedule (linear warmup and decay)
# 设置 bias 和 LayerNorm.weight 不使用 weight_decay
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

#optimizer = AdamW(model.parameters(), lr=learning_rate)
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = nn.CrossEntropyLoss()
criterion.to(device)

# tokenized_text = tokenizer(text, max_length=300, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")

# 定义训练的函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(dataloader):
        # 标签形状为 (batch_size, 1)

        label = batch["label"]
        text = batch["text"]
        # tokenized_text 包括 input_ids， token_type_ids， attention_mask
        tokenized_text = tokenizer(text, max_length=300, add_special_tokens=True, truncation=True, padding=True, return_tensors="pt")
        tokenized_text = tokenized_text.to(device)
        label = label.to(device)
        # 梯度清零
        optimizer.zero_grad()
        #output: (loss), logits, (hidden_states), (attentions)
        output = model(**tokenized_text, labels=label)
        # y_pred_prob = logits : [batch_size, num_labels]
        y_pred_prob = output[1]
        y_pred_prob = model2(y_pred_prob)
        y_pred_label = y_pred_prob.argmax(dim=1)
        # 计算loss
        # 这个 loss 和 output[0] 是一样的
        loss = criterion(y_pred_prob.view(-1, 2), label.view(-1))
        # 计算acc
        acc = ((y_pred_label == label.view(-1)).sum()).item()
        # 反向传播
        loss.backward()
        optimizer.step()
        # epoch 中的 loss 和 acc 累加
        # loss 每次是一个 batch 的平均 loss
        epoch_loss += loss.item()
        # acc 是一个 batch 的 acc 总和
        epoch_acc += acc
        if i % 200 == 0:
            print(i,"个batch数据： current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(label)))

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(dataloader), epoch_acc / dataloader.dataset.__len__()

def evaluate(model, iterator, device):
    model.eval()
    model2.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            label = batch["label"]
            text = batch["text"]
            tokenized_text = tokenizer(text, max_length=300, add_special_tokens=True, truncation=True, padding=True,
                                       return_tensors="pt")
            tokenized_text = tokenized_text.to(device)
            label = label.to(device)
            output = model(**tokenized_text, labels=label)
            y_pred_prob = output[1]
            y_pred_prob = model2(y_pred_prob)
            y_pred_label = y_pred_prob.argmax(dim=1)
            loss = output[0]
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            # epoch 中的 loss 和 acc 累加
            # loss 每次是一个 batch 的平均 loss
            epoch_loss += loss.item()
            # acc 是一个 batch 的 acc 总和
            epoch_acc += acc

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(iterator), epoch_acc / iterator.dataset.__len__()



validacc_list = []

# 开始训练和验证
for i in range(epochs):
    print("---------------------------------第",i+1,"次训练开始----------------------------------")
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    print(i+1,"train loss: ", train_loss, "\t", "train acc:", train_acc)
    print("---------------------------------第", i + 1, "次训练结束----------------------------------")
    print("---------------------------------第", i + 1, "次测试开始----------------------------------")
    valid_loss, valid_acc = evaluate(model, test_loader, device)
    validacc_list.append(valid_acc)
    print(i+1,"valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
    print("---------------------------------第",i+1,"次测试结束----------------------------------")

    # test_sentence(sentence_list,label_list)
print(validacc_list)


path = "../model/first_stage/mm_AG4000_HC5000.pth"
#保存
torch.save(model, path)
# #读取
# model = torch.load(path)