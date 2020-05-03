import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
# import torchvision.transforms as tf


class RNN(nn.Module):
    def __init__(self, in_dim = 28, hidden_dim = 100, n_layer = 2, n_class = 10):
        super(RNN, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

batch_size = 64
learning_rate = 1e-2
num_epochs = 20

data_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = RNN()
if torch.cuda.is_available():
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    correct = 0
    num = 0
    # get input
    for data in train_loader:
        img, label = data
        img = Variable(img).cuda()
        label = Variable(label).cuda()
        img = img.squeeze()
        # print(img.size())
        out = model(img)
        # print(label)
        # print(out)
        # print(out.size())
        loss = criterion(out, label)
        print_loss = loss.item()
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        correct += num_correct
        # print('num_correct: {:d}'.format(num_correct))
        # print('img size: {:d}'.format(img.size(0)))
        num += img.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print('correct: {:.4f}'.format(correct))
        print('num : {:.4f}'.format(num))
        acc = float(correct) / num
        print('*'*10)
        print('epoch {}'.format(epoch + 1))
        print('loss is {:.4f}'.format(print_loss))
        print('acc is {:.4f}'.format(acc))

model.eval()
eval_loss = 0
eval_acc = 0

for data in test_loader:
    img, label = data
    # img = img.view(img.size(0), -1)
    with torch.no_grad():
        img = Variable(img).cuda()
        label = Variable(label).cuda()
    img = img.squeeze()
    out = model(img)
    loss = criterion(out, label)
    eval_loss += loss.item() * label.size(0)
    _, pred = torch.max(out, 1)
    num_correct = (pred == label).sum()
    eval_acc += num_correct.item()

print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_dataset)), eval_acc / (len(test_dataset))))