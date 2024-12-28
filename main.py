import os.path
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms
from modules.stacked_rbm import StackedRBM
from modules.diffusion import Diffusion
from tqdm import tqdm
import numpy as np


def plot(X, img_shape, filename):
    X = X.detach().numpy()
    fig = plt.figure(figsize=(5, 5))
    for i in range(25): 
        sub = fig.add_subplot(5, 5, i+1)
        sub.imshow(X[i, :].reshape(img_shape), cmap=plt.cm.gray)
    
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig(filename)


def train_rmb(model, train_loader, 
              checkpt="output/chk_rbm.pt", n_epochs=10, 
              lr=0.01, batch_size=64):
    checkpt_epoch = 0
    if os.path.isfile(checkpt):
        checkpoint = torch.load(checkpt, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        checkpt_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
    model.train()

    for epoch in range(n_epochs):
        loss_ = []
        if epoch and epoch % 10 == 0:
            # learning rate decay
            lr = lr * 0.9
        for _, (data, _) in enumerate(train_loader):
            input = data.view(-1, model.n_visible)
            input = ((input/255.0) * 2.0) - 1.0
            loss = model.fit(input, lr=lr, batch_size=batch_size)
            loss_.append(loss.item())

        epoch_ = epoch + checkpt_epoch
        checkpoint = {
            'epoch': epoch_,
            'state_dict': model.state_dict(),
            'lr': lr
        }

        torch.save(checkpoint, checkpt)
        print('Epoch %d Loss=%.4f' % (epoch_, np.mean(loss_)))

    return model


def train_diffusion(model, train_loader, 
                   checkpt="output/chk_diffusion.pt", n_epochs=10, 
                   lr=0.01):
    checkpt_epoch = 0
    if os.path.isfile(checkpt):
        checkpoint = torch.load(checkpt, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        checkpt_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        loss_ = []
        for _, (data, _) in enumerate(tqdm(train_loader)):
            input = ((data/255.0) * 2.0) - 1.0
            input = input.moveaxis(3, 1)

            optimizer.zero_grad()
            loss = model.fit(data)
            loss_.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_ = epoch + checkpt_epoch
        checkpoint = {
            'epoch': epoch_,
            'state_dict': model.state_dict(),
            'lr': lr
        }

        torch.save(checkpoint, checkpt)
        print('Epoch %d Loss=%.4f' % (epoch_, np.mean(loss_)))

    return model


def stack_samples(gen_samples, stack_dim):
    gen_samples = list(torch.split(gen_samples, 1, dim=1))
    for i in range(len(gen_samples)):
        gen_samples[i] = gen_samples[i].squeeze(1)
    return torch.cat(gen_samples, dim=stack_dim)


def main_diffusion():
    batch_size = 1024 
    n_epochs = 1
    learning_rate = 0.01
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                    transform=transforms.ToTensor())
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    model = Diffusion(784, 1)
    model = train_diffusion(model, train_loader, 
                            n_epochs=n_epochs, 
                            lr=learning_rate)

    sample_batch_size = 25

    # image gen from noise
    x = torch.randn((sample_batch_size, 1, 28, 28))
    sample_steps = torch.arange(model.timesteps-1, 0, -1)
    for t in sample_steps:
        x = model.denoise(x, t)
    x = (x.clamp(-1, 1) + 1) / 2

    plot(x, [28, 28], 'output/diffusion.png')


def main_rbm():
    batch_size = 128 
    n_epochs = 100
    learning_rate = 0.01
    k = 1
    img_shape = (28, 28)
    n_nodes=[784, 256, 128]
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                    transform=transforms.ToTensor())
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    model = StackedRBM(n_nodes, k)

    model = train_rmb(model, train_loader, 
                      n_epochs=n_epochs, 
                      lr=learning_rate, batch_size=batch_size)
    w0 = model.rbm_modules[0].weight
    w1 = model.rbm_modules[1].weight

    # test the generated image
    images = next(iter(train_loader))[0]
    v_gen, _ = model(images.view(-1, model.n_visible))

    # plot the results
    plot(w0, img_shape, 'output/filters0.png')
    plot(w1, [16, 16], 'output/filters1.png')

    plot(v_gen, img_shape, 'output/gen_img.png')


if __name__ == "__main__":
    main_rbm()
    # main_diffusion()


