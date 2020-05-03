import os

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image

import torch.nn.functional as F

batch_size = 128
learning_rate = 1e-2
num_epochs = 100

data_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = datasets.MNIST(root='./data/', transform=data_tf, download=True)
# train_dataset = datasets.MNIST(root='./data/', train=True, transform=data_tf, download=True)
# test_dataset = datasets.MNIST(root='./data/', train=False, transform=data_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.tanh(self.fc4(h3))
    
    def rep(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.rep(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

model = VAE().cuda()
rf = nn.MSELoss(size_average=False)
optimizer = optim.Adam(model.parameters(), learning_rate)

def loss_function(rx, x, mu, logvar):
    bce = rf(rx, x)
    klds = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    kld = torch.sum(klds).mul_(-0.5)
    return bce + kld

# reconstruction_function = nn.MSELoss(size_average=False)

# def loss_function(recon_x, x, mu, logvar):
#     """
#     recon_x: generating images
#     x: origin images
#     mu: latent mean
#     logvar: latent log variance
#     """
#     MSE = reconstruction_function(recon_x, x)
#     # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
#     KLD = torch.sum(KLD_element).mul_(-0.5)
#     # KL divergence
#     return MSE + KLD

def to_img(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    # print(x.shape[0]) # batch size
    x = x.view(x.shape[0], 1, 28, 28)
    return x

for epoch in range(num_epochs):
    # get input
    for data in train_loader:
        img, _ = data
        img = img.view(img.shape[0], -1)
        # print('img size(0) : {:s}'.format(img.size(0).__str__()))
        # print('img shape[0] : {:s}'.format(img.shape[0].__str__()))
        img = Variable(img).cuda()
        out, mu, logvar = model(img)
        loss = loss_function(out, img, mu, logvar) / img.shape[0]
        print_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 20 == 0:
        print('*'*10)
        print('epoch {}'.format(epoch + 1))
        print('loss is {:.4f}'.format(print_loss))
        save = to_img(out.cpu().data)
        if not os.path.exists('./test_vae_img'):
            os.mkdir('./test_vae_img')
        save_image(save, './test_vae_img/image_{}.png'.format(epoch + 1))

# idx = 0
# model.eval()

# for data in test_loader:
# for data in test_loader:
#     img, _ = data
#     img = img.view(img.size(0), -1)
#     img = Variable(img).cuda()
#     out, _, _ = model(img)
#     img = to_img(out.cpu().data)
#     save_image(img, 'image_{:04d}.png'.format(idx))
#     idx = idx + 1


