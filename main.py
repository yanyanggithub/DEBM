import os.path
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from agents.rbm import RBM, RBMConfig
from torch.nn import functional as F


def plot(X, img_shape, filename):
    X = X.detach().numpy()
    fig = plt.figure(figsize=(5, 5))
    for i in range(25): 
        sub = fig.add_subplot(5, 5, i+1)
        sub.imshow(X[i, :].reshape(img_shape), cmap=plt.cm.gray)
    
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(filename)


def train_rmb(model, config, train_loader, optimizer, 
              checkpt="output/checkpoint.pt", n_epochs=20, lr=0.01):
    checkpt_epoch = 0
    if os.path.isfile(checkpt):
        checkpoint = torch.load(checkpt, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        checkpt_epoch = checkpoint['epoch']
    model.train()

    for epoch in range(n_epochs):
        for _, (data, _) in enumerate(train_loader):
            v, v_reconstructed = model(data.view(-1, config.n_visible))
            loss = model.energy(v) - model.energy(v_reconstructed)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_ = epoch + checkpt_epoch
        checkpoint = {
            'epoch': epoch_,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, checkpt)
        print('Epoch %d\t Loss=%.4f' % (epoch_, loss))

    return model


def main():
    batch_size = 128 
    n_epochs = 10 
    learning_rate = 5e-4 
    data_type = "MNIST"  # "CIFAR10"
    # data_type = "CIFAR10"  # "CIFAR10"

    config = RBMConfig()

    if data_type == "MNIST":
        img_shape = (28, 28)
        config.n_visible = 784
        config.n_hidden  = 128
        train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                       transform=ToTensor())
    elif data_type == "CIFAR10":
        img_shape = (32, 32, 3)
        config.n_visible = 3072
        config.n_hidden  = 1024    
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, 
                                         transform=ToTensor())
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    model = RBM(n_visible=config.n_visible, 
                n_hidden=config.n_hidden, k=config.k)
    optimizer = optim.Adam(model.parameters(), learning_rate)

    model = train_rmb(model, config, train_loader, optimizer, 
                      n_epochs=n_epochs, lr=learning_rate)
    W = model.weight

    # test the generated image
    images = next(iter(train_loader))[0]
    _, v_gen = model(images.view(-1, config.n_visible))

    # plot the results
    plot(W, img_shape, 'output/filters.png')
    plot(v_gen, img_shape, 'output/gen_img.png')


if __name__ == "__main__":
    main()


