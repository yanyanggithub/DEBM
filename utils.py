import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import numpy as np
from modules.diffusion import Diffusion
from modules.flow_matching import Flow
import os


def plot(X, img_shape, filename):
    X = X.to("cpu")
    X = X.detach().numpy()
    fig = plt.figure(figsize=(5, 5))
    for i in range(25): 
        sub = fig.add_subplot(5, 5, i+1)
        img = X[i, :].reshape(img_shape)
        if len(img_shape) == 3:
            img = np.moveaxis(img, 0, -1)
        sub.imshow(img, cmap=plt.cm.gray)
    
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(filename)


def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])  
        train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                        transform=transform)
        img_shape = (28, 28)
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), 
                                 (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, 
                                        transform=transform)  
        img_shape = (3, 32, 32)
    
    return train_dataset, img_shape

class Trainer:
    def __init__(self, model, dataset_name='mnist',
                  checkpt="./output/chk_rbm.pt", n_epochs=10, 
                  lr=0.01, batch_size=64, device="cpu"): 
        self.model = model
        self.dataset_name = dataset_name
        self.device = device
        self.checkpt = checkpt
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.checkpt_epoch = 0
        self.setup()
    
    def setup(self):
        if torch.cuda.is_available():
            self.device = "cuda"
            torch.cuda.set_device("cuda:0")
        if os.path.isfile(self.checkpt):
            checkpoint = torch.load(self.checkpt, weights_only=True)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.checkpt_epoch = checkpoint['epoch']
            self.lr = checkpoint['lr']
        self.model.to(self.device)
        self.model.train()

    def save_chekpoint(self, epoch_):
        checkpoint = {
            'epoch': epoch_,
            'state_dict': self.model.state_dict(),
            'lr': self.lr
        }
        torch.save(checkpoint, self.checkpt)

    def train_rbm(self, train_loader):
        for epoch in range(self.n_epochs):
            loss_ = []
            if epoch and epoch % 10 == 0:
                # learning rate decay
                lr = lr * 0.9
            for _, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                input = data.view(-1, self.model.n_visible)
                loss = self.model.fit(input, self.lr, self.batch_size)
                loss_.append(loss.item())

            epoch_ = epoch + self.checkpt_epoch + 1
            self.save_chekpoint(epoch_) 

            print('Epoch %d Loss=%.4f' % (epoch_, np.mean(loss_)))
        return self.model

    def train_diffusion(self, train_loader):    
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        diffusion = Diffusion(timesteps=1000, device=self.device)

        for epoch in range(self.n_epochs):
            loss_ = []
            for _, (data, _) in enumerate(tqdm(train_loader)):
                data = data.to(self.device)
                if self.dataset_name == 'mnist':
                    # [batch_size, 1, 28, 28]
                    data.unsqueeze(3)
                    data = data * 2 - 1
                t = torch.randint(0, diffusion.timesteps, (data.shape[0],))
                t = t.to(self.device)
                noise = torch.randn_like(data).to(self.device)
                noisy_data = diffusion.add_noise(data, noise, t)
                optimizer.zero_grad()
                pred_noise = self.model(noisy_data, t)
                loss = loss_fn(pred_noise, noise)
                loss_.append(loss.item())
                loss.backward()
                optimizer.step()

            epoch_ = epoch + self.checkpt_epoch + 1
            self.save_chekpoint(epoch_) 
            print('Epoch %d Loss=%.4f' % (epoch_, np.mean(loss_)))
        return self.model
    

    # flow matching
    def train_fm(self, train_loader):    
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        flow = Flow(device=self.device)

        for epoch in range(self.n_epochs):
            loss_ = []
            for _, (data, _) in enumerate(tqdm(train_loader)):
                data = data.to(self.device)
                if self.dataset_name == 'mnist':
                    # [batch_size, 1, 28, 28]
                    data.unsqueeze(3)
                    data = data * 2 - 1
                x0 = torch.randn_like(data).to(self.device)
                t = torch.rand(data.shape[0]).to(self.device)
                xt, target = flow.sample_xt(data, x0, t)
                optimizer.zero_grad()
                pred = self.model(xt, t)
                loss = loss_fn(pred, target)
                loss_.append(loss.item())
                loss.backward()
                optimizer.step()

            epoch_ = epoch + self.checkpt_epoch + 1
            self.save_chekpoint(epoch_) 
            print('Epoch %d Loss=%.4f' % (epoch_, np.mean(loss_)))
        return self.model