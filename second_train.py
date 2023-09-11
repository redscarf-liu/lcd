import numpy as np
from transformers import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer, BertModel

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

path = "../model/mmAGhc40005000.pth"
model = torch.load(path)
# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',Truncation=True)

model.to(device)

cla_r = 1.5

def  text_to_chat(text):

    # text = "This is a sample text for BERT feature extraction."

    # 使用tokenizer将文本转换为token ID和attention mask
    tokens = tokenizer.encode_plus(text, max_length=300, add_special_tokens=True, return_tensors='pt')
    tokens.to(device)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # 使用BERT模型处理输入，得到特征向量
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        feature_vector = output[0].cpu().numpy()[0].tolist()
    feature = []
    for i in feature_vector:
        feature.append(i*1)
    # print(feature)
    return feature


test_text = "If I were asked to send one thing representing my country to an international exhibition, I would choose the Statue of Liberty. The Statue of Liberty is a symbol of freedom and democracy, representing the values that my country was founded upon."


#求列表间的欧式距离
def ou(a,b):
    a = np.array(a)
    b = np.array(b)

    dist = np.linalg.norm(a-b)
    return  dist



#用于对矩阵转置
def trans(l):
    hang = len(l)
    lie = len(l[0])
    re = []
    for y in range(lie):
        tem = []
        for x in l:
            tem.append(x[y])
        re.append(tem)
    return  re


train_ag_human_path = "../data/AGdata/train/human.txt"
train_ag_chatgpt_path ="../data/AGdata/train/chatgpt.txt"
train_hc_human_path = "../data/HCdata/train/human.txt"
train_hc_chatgpt_path = "../data/HCdata/train/chatgpt.txt"

test_ag_human_path = "../data/AGdata/test/human.txt"
test_ag_chatgpt_path = "../data/AGdata/test/chatgpt.txt"
test_hc_human_path = "../data/HCdata/test/human.txt"
test_hc_chatgpt_path = "../data/HCdata/test/chatgpt.txt"
test_ft_human_path = "../data/full_text/human.txt"
test_ft_chatgpt_path = "../data/full_text/realchatgpt.txt"

test_cs_human_path = "../data/cs224ndata/human.txt"
test_cs_chatgpt_path = "../data/cs224ndata/chatgpt.txt"


def getm ():

    x_gpt_train = []
    y_gpt_train = []

    with open(train_ag_chatgpt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            x_gpt_train.append(text_to_chat(i))
            y_gpt_train.append(1)

    with open(train_hc_chatgpt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            x_gpt_train.append(text_to_chat(i))
            y_gpt_train.append(1)

    tran_x_train =trans(x_gpt_train)
    m = []
    for i in tran_x_train:
        i.sort()
        m.append(i[len(i)//2])
    print("---------------------------------------",m)

    return  m

m = getm()


def getr():
    x_gpt_train = []
    y_gpt_train = []

    with open(train_ag_chatgpt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            x_gpt_train.append(text_to_chat(i))
            y_gpt_train.append(1)

    with open(train_hc_chatgpt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            x_gpt_train.append(text_to_chat(i))
            y_gpt_train.append(1)

    m = getm()
    print("初始化r时c为",m)

    all_distlist = []
    for i in x_gpt_train:
        all_distlist.append(ou(m,i))
    all_distlist.sort()
    leng = len(all_distlist)
    r = all_distlist[int(0.95*leng)]
    print("r初始化为",r)
    return r


cla_r = getr()


#数据集类,加载数据和特征1
class  HCdataset2(Dataset):

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


use_traindataset = HCdataset2(train_ag_human_path,train_ag_chatgpt_path)+HCdataset2(train_hc_human_path,train_hc_chatgpt_path)
train_loader = DataLoader(use_traindataset, batch_size=32, shuffle=True, num_workers=0)




no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

#optimizer = AdamW(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=0.001)


class CustomLoss(torch.nn.Module):
    def __init__(self,c,r):
        super(CustomLoss, self).__init__()
        self.c = torch.tensor(c).to(device)
        self.r = torch.tensor(r).to(device)
        self.distance = torch.nn.PairwiseDistance(p=2).to(device)  # 使用欧氏距离

    def forward(self, predictions, targets):
        distances = self.distance(predictions, self.c)  # 计算模型输出与c的距离
        loss = torch.mean((targets * torch.max(torch.zeros_like(distances),(distances - self.r))  ) ** 2 + ((1 - targets) * torch.max(torch.zeros_like(distances), self.r - distances)) ** 2)
        return loss



def train(train_loader):
    # model.train()
    c = getm()
    r = cla_r
    loss_fn = CustomLoss(c,r).to(device)
    print("本次训练分类中心为", loss_fn.c)
    print("本次训练分类半径为", loss_fn.r)
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(train_loader):
        # 标签形状为 (batch_size, 1)

        label = batch["label"]
        text = batch["text"]
        # tokenized_text 包括 input_ids， token_type_ids， attention_mask
        tokenized_text = tokenizer(text, max_length=300, add_special_tokens=True, truncation=True, padding=True,
                                   return_tensors="pt")
        tokenized_text = tokenized_text.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(**tokenized_text, labels=label)
        # y_pred_prob = logits : [batch_size, num_labels]

        y_pred_prob = output[1]
        loss = loss_fn(y_pred_prob, label)

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()



    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(train_loader)





def test(test_chatgpt_path,test_human_path,max):
    m = getm()
    x_test = []
    y_test = []
    with open(test_chatgpt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            x_test.append(text_to_chat(i))
            y_test.append(1)
    with open(test_human_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            x_test.append(text_to_chat(i))
            y_test.append(0)
    # x_test = pd.DataFrame(scaler.transform(x_test)).values.tolist()
    correct = 0
    print("测试时分类中心为：",m)
    print("测试时分类半径为",max)
    tp = 1
    fp = 1
    tn = 1
    fn = 1
    all = 4
    for i,x in enumerate(x_test):
        if y_test[i] == 1:
            if ou(m,x)>=max:
                fn+=1
                all+=1
            else:
                tp+=1
                all+=1
        else:
            if ou(m,x)>=max:
                tn+=1
                all+=1
            else:
                fp+=1
                all+=1

    accuracy = (tp + tn) / all
    return  [accuracy,tp,fp,tn,fn,all]



epoch = 16
for i in range(epoch):

    print("第",i,"个epoch开始训练")
    print("loss = ",train(train_loader))
    #调整分类半径
    re1 = test(test_ag_chatgpt_path, test_ag_human_path,cla_r)
    re2 = test(test_hc_chatgpt_path, test_hc_human_path,cla_r)
    tp1 = re1[1]
    fp1 = re1[2]
    tn1 = re1[3]
    fn1 = re1[4]
    all1 = re1[5]

    tp2 = re2[1]
    fp2 = re2[2]
    tn2 = re2[3]
    fn2 = re2[4]
    all2 = re1[5]

    tp =tp1+tp2
    fp =fp1+fp2
    tn =tn1+tn2
    fn =fn1+fn2
    all =all1 + all2

    wc = fn/(tp+fn)
    wh = fp/(tn+fp)
    ba = wc/(wc+wh)
#ba代表bf(Boundary fairness)即边界公平性

    if  ba>0.5:
        print("----------------------------------------------------------------------")
        print("本次调整r为扩大")
        cla_r = cla_r + 0.0001
    else:
        print("----------------------------------------------------------------------")
        print("本次调整r为缩小")
        cla_r = cla_r - 0.0001

    print("ag测试精度", test(test_ag_chatgpt_path, test_ag_human_path, cla_r ))
    print("hc测试精度", test(test_hc_chatgpt_path, test_hc_human_path, cla_r ))
    print("ft测试精度", test(test_ft_chatgpt_path, test_ft_human_path, cla_r ))
    print("cs测试精度", test(test_cs_chatgpt_path, test_cs_human_path, cla_r ))




# model_path = "/home/ncubigdata1/Documents/liurundong/HC/bias/myidea/model/mymodel_epoch16.pth"
#
# #保存
# torch.save(model, model_path)
