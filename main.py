import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from agents.rbm import RBM, RMBConfig

def plot(data_mat):
    result = np.transpose(data_mat.numpy(), (1, 2, 0))
    plt.imshow(result, cmap='gray')
    plt.show()


def train_rmb(model, config, train_loader, optimizer, n_epochs=20, lr=0.01):
    model.train()

    for epoch in range(n_epochs):
        for _, (data, _) in enumerate(train_loader):
            v, v_reconstructed = model(data.view(-1, config.n_visible))
            loss = model.energy(v) - model.energy(v_reconstructed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch %d\t Loss=%.4f' % (epoch, loss))

    return model


def main():
    batch_size = 128 
    n_epochs = 40 
    learning_rate = 5e-4 
    data_type = "MNIST"  # "CIFAR10"
    # data_type = "CIFAR10"  # "CIFAR10"

    config = RMBConfig()

    dataset_cifar10 = datasets.CIFAR10('./data', train=True, 
                                       download=True, 
                                       transform=transforms.Compose([transforms.ToTensor()]))
    
    dataset_mnist = datasets.MNIST('./data', train=True, 
                                    download=True, 
                                    transform=transforms.Compose([transforms.ToTensor()]))
    
    if data_type == "MNIST":
        train_loader = torch.utils.data.DataLoader(dataset_mnist, batch_size=batch_size)
    elif data_type == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(dataset_cifar10, batch_size=batch_size)
        config.n_visible = 3072
        config.n_hidden  = 1024

    model = RBM(n_visible=config.n_visible, n_hidden=config.n_hidden, k=config.k)
    optimizer = optim.Adam(model.parameters(), learning_rate)

    model = train_rmb(model, config, train_loader, optimizer, n_epochs=n_epochs, lr=learning_rate)
    W = model.weight

    if data_type == "MNIST":
        plot(make_grid(W.view(batch_size, 1, 28, 28).data))
    elif data_type == "CIFAR10":
        plot(make_grid(W.view(batch_size, 3, 32, 32).data))
    

if __name__ == "__main__":
    main()


