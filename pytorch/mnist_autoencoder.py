import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

batch_size = 128
learning_rate = 1e-2
num_epochs = 100

data_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root='./data/', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoderConv(nn.Module):
    def __init__(self):
        super(AutoEncoderConv, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride = 3, padding = 1), # (W - F + 2 P) / S + 1 = (28 - 3 + 2 * 1) / 3 + 1 = 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride = 2), # (W - F) / S + 1 = (10 - 2) / 2 + 1 = 5
            nn.Conv2d(16, 8, 3, stride = 2, padding = 1), # (5 - 3 + 2) / 2 + 1 = 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride = 1) # (3 - 2) / 1 + 1 = 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride = 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride = 3, padding = 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride = 2, padding = 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# model = AutoEncoder().cuda()
model = AutoEncoderConv().cuda()
# criterion = nn.MSELoss(size_average=False) # 效果很烂，毛乎乎的
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), learning_rate)

for epoch in range(num_epochs):
    # get input
    for data in train_loader:
        img, _ = data
        # img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        out = model(img)
        # print(label)
        # print(out)
        # print(out.size())
        loss = criterion(out, img)
        print_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 1 == 0:
        print('*'*10)
        print('epoch {}'.format(epoch + 1))
        print('loss is {:.4f}'.format(print_loss))

def to_img(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    # print(x.shape[0]) # batch size
    x = x.view(x.shape[0], 1, 28, 28)
    return x

idx = 0

for data in test_loader:
    img, _ = data
    # img = img.view(img.size(0), -1)
    img = Variable(img).cuda()
    out = model(img)
    img = to_img(out.cpu().data)
    save_image(img, 'image_{:04d}.png'.format(idx))
    idx = idx + 1


