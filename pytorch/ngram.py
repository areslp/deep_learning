import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F

CONTEXT_SIZE = 2 # 依据的单词数
EMBEDDING_DIM = 100 # 词向量的维度
# 我们使用莎士比亚的诗
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2]) for i in range(len(test_sentence) - 2)]

vocab = set(test_sentence)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

class NgramModel(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocab_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out)
        return log_prob

learning_rate = 1e-2
num_epochs = 100

vocab_size = len(vocab)
Ngram = NgramModel(vocab_size, 2, 100)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(Ngram.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    total_loss = 0.0
    total_correct = 0
    for word, label in trigram:
        # print(word)
        # print(label)
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word])) # 将两个词作为输入
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # print(word.size())
        out = Ngram(word)
        loss = criterion(out, label)
        print_loss = loss.item()
        _, pred = torch.max(out, 1)
        correct = pred == label
        total_loss += print_loss
        total_correct += correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # print(total_loss)
    # print(total_correct)
    print('loss : {:.4f}, acc : {:.4f}'.format(total_loss, total_correct / len(trigram)))

word, label = trigram[3]
word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
out = Ngram(word)
_, pred = torch.max(out, 1)
pred_word = idx_to_word[pred.item()]
print('real word is {}, pred word is {}'.format(label, pred_word))