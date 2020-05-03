import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_idx = {}
tag_to_idx = {}

for context, tag in training_data:
    for word in context:
        if word.lower() not in word_to_idx:
            word_to_idx[word.lower()] = len(word_to_idx)
    for label in tag:
        if label.lower() not in tag_to_idx:
            tag_to_idx[label.lower()] = len(tag_to_idx)

alphabet = 'abcdefghijklmnopqrstuvwxyz'

char_to_idx = {}
for i in range(len(alphabet)):
    char_to_idx[alphabet[i]] = i

def make_sequence(x, dic):
    list=[]
    for i in x:
        list.append(dic[i.lower()])
    return torch.LongTensor(list)

class CharLSTM(nn.Module):
    def __init__(self, n_char, char_dim, char_hidden):
        super(CharLSTM, self).__init__()
        self.char_embedding = nn.Embedding(n_char, char_dim).cuda()
        self.char_lstm = nn.LSTM(char_dim, char_hidden).cuda()

    def forward(self, x):
        # char num, 1
        # print('input x : {:s}'.format(x.size().__str__()))
        # char num, 1, char_dim
        x = self.char_embedding(x)
        # print('embedding x : {:s}'.format(x.size().__str__()))
        # char num, 1, char_hidden
        out, _ = self.char_lstm(x)
        # print('char lstm forwar out : {:s}'.format(out.size().__str__()))
        # [last seq], 1, char_hidden
        out = out[-1, :, :]
        # print('trim char lstm forwar out : {:s}'.format(out.size().__str__()))
        return out

class TagLSTM(nn.Module):
    def __init__(self, n_word, n_char, char_dim, n_dim, char_hidden, n_hidden, n_tag):
        super(TagLSTM, self).__init__()
        self.word_embedding = nn.Embedding(n_word, n_dim).cuda()
        self.char_lstm = CharLSTM(n_char, char_dim, char_hidden)
        self.lstm = nn.LSTM(n_dim + char_hidden, n_hidden).cuda()
        self.linear1 = nn.Linear(n_hidden, n_tag).cuda()

    def forward(self, x, word_data):
        words = [i for i in word_data]
        # print('words size : {:d}'.format(len(words)))
        char = []
        # for every word
        for word in words:
            char_list = make_sequence(word, char_to_idx)
            # print('char list size : {:s}'.format(char_list.size().__str__()))
            # add batch_size for lstm : char num, batch
            char_list = char_list.unsqueeze(1)
            # print('lstm char list size : {:s}'.format(char_list.size().__str__()))
            # compute feature from char lstm
            char_list = Variable(char_list).cuda()
            # batch, char_hidden
            out = self.char_lstm(char_list)
            # print('lstm char output : {:s}'.format(out.size().__str__()))
            # remove batch dimension : hidden
            out = out.squeeze(0)
            # print('trim char output : {:s}'.format(out.size().__str__()))
            char.append(out)
        # word num, hidden
        char = torch.stack(char, dim = 0)
        # word num, batch, hidden
        char = char.unsqueeze(1)
        # print('char.size() : {:s}'.format(char.size().__str__()))
        # x ：batch, word_num
        # output x : batch, word_num, word_hidden
        x = self.word_embedding(x)
        # print('embedding x size : {:s}'.format(x.size().__str__()))
        # word_num, batch, word_hidden
        x = x.permute(1, 0, 2)
        # word feature + char feature
        # word num, batch, word embedding dim (n_dim) + char hidden
        x = torch.cat((x, char), dim = 2)
        # print('concat x : {:s}'.format(x.size().__str__()))
        # word num, batch, word hidden : 5, 1, 128
        x, _ = self.lstm(x)
        # print('lstm output : {:s}'.format(x.size().__str__()))
        s, b, h = x.shape
        # word num, word hidden
        x = x.view(-1, h)
        # print('x : {:s}'.format(x.size().__str__()))
        # word num, tag num
        x = self.linear1(x)
        # print('x : {:s}'.format(x.size().__str__()))
        # word num, tag num
        y = F.log_softmax(x, dim = 0)
        # print('y : {:s}'.format(y.size().__str__()))
        return y

# n_word, n_char, char_dim, n_dim, char_hidden, n_hidden, n_tag
net = TagLSTM(len(word_to_idx), len(char_to_idx), 10, 100, 50, 128, len(tag_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

for e in range(300):
    train_loss = 0
    for word, tag in training_data:
        # print(tag)
        word_list = make_sequence(word, word_to_idx).unsqueeze(0) # 添加第一维 batch
        tag = make_sequence(tag, tag_to_idx)
        word_list = Variable(word_list).cuda()
        tag = Variable(tag).cuda()
        # print(tag)
        # 前向传播
        out = net(word_list, word)
        loss = criterion(out, tag)
        train_loss += loss.item()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 50 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, train_loss / len(training_data)))

# convert model to inference module
net.eval()
test_sent = 'Everybody ate the apple'.split()
test = make_sequence(test_sent, word_to_idx).unsqueeze(0)
out = net(Variable(test).cuda(), test_sent)
print(out)
print(tag_to_idx)