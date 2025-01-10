import os.path
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms
from modules.stacked_rbm import StackedRBM
from modules.diffusion import Diffusion
from modules.unet import Unet
from tqdm import tqdm
import numpy as np
from ast import literal_eval
import sys

dataset_name = 'mnist'  # 'mnist' or 'cifar10'
model_name = 'rbm' # 'rbm' or 'diffusion'
batch_size = 128
n_epochs = 100
learning_rate = 1e-2
data_dir = './data'
output_dir = './output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.set_device("cuda:0")

transform = transforms.Compose([transforms.ToTensor()])
if dataset_name == 'mnist':     
    train_dataset = datasets.MNIST('./data', train=True, download=True, 
                                    transform=transform)
    img_shape = (28, 28)
elif dataset_name == 'cifar10':
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, 
                                     transform=transform)  
    img_shape = (3, 32, 32)


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
            data = data.to(device)
            input = data.view(-1, model.n_visible)
            loss = model.fit(input, lr=lr, batch_size=batch_size)
            loss_.append(loss.item())

        epoch_ = epoch + checkpt_epoch + 1
        checkpoint = {
            'epoch': epoch_,
            'state_dict': model.state_dict(),
            'lr': lr
        }

        torch.save(checkpoint, checkpt)
        print('Epoch %d Loss=%.4f' % (epoch_, np.mean(loss_)))

    return model


def train_diffusion(model, train_loader, 
                   checkpt="./output/chk_diffusion.pt", n_epochs=10, 
                   lr=0.01):
    checkpt_epoch = 0
    if os.path.isfile(checkpt):
        checkpoint = torch.load(checkpt, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        checkpt_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    diffusion = Diffusion(timesteps=1000, device=device)

    for epoch in range(n_epochs):
        loss_ = []
        for _, (data, _) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            if dataset_name == 'mnist':
                # [batch_size, 1, 28, 28]
                data.unsqueeze(3)
            elif dataset_name == 'cifar10':
                data = data / 255
            data = data * 2 - 1
            t = torch.randint(0, diffusion.timesteps, (data.shape[0],))
            t = t.to(device)
            noise = torch.randn_like(data).to(device)
            noisy_data = diffusion.add_noise(data, noise, t)
            optimizer.zero_grad()
            pred_noise = model(noisy_data, t)
            loss = loss_fn(pred_noise, noise)
            loss_.append(loss.item())
            loss.backward()
            optimizer.step()

        epoch_ = epoch + checkpt_epoch + 1
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
    if dataset_name == 'mnist':
        n_channels = 1
    elif dataset_name == 'cifar10':
        n_channels = 3

    model = Unet(n_channels, t_emb_dim=128, device=device)
    model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    model = train_diffusion(model, train_loader, 
                            n_epochs=n_epochs, 
                            lr=learning_rate)

    sample_batch_size = 25

    # image gen from noise
    if dataset_name == 'mnist':
        x = torch.randn((sample_batch_size, 1, 28, 28))
    elif dataset_name == 'cifar10':
        x = torch.randn((sample_batch_size, 3, 32, 32))
    x = x.to(device)
    diffusion = Diffusion(timesteps=1000, device=device)
    with torch.no_grad():
        for t in reversed(range(diffusion.timesteps)):
            noise_pred = model(x, torch.as_tensor(t).unsqueeze(0).to(device))
            x, _ = diffusion.denoise(x, noise_pred, torch.as_tensor(t).to(device))
    plot(x, img_shape, './output/diffusion.png')


def main_rbm():
    k = 1

    if dataset_name == 'mnist':
        filter1_shape = (16, 16)
        n_nodes = [784, 256, 128]
    elif dataset_name == 'cifar10':
        filter1_shape = (32, 32)
        n_nodes = [3072, 1024, 784]
        
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True)

    model = StackedRBM(n_nodes, k)
    model.to(device)

    model = train_rmb(model, train_loader, 
                      n_epochs=n_epochs, 
                      lr=learning_rate, batch_size=batch_size)
    w0 = model.rbm_modules[0].weight
    w1 = model.rbm_modules[1].weight

    # test the generated image
    images = next(iter(train_loader))[0]
    images = images.to(device)
    v_gen, _ = model(images.view(-1, model.n_visible))

    # plot the results
    plot(w0, img_shape, './output/rbm_filters0.png')
    plot(w1, filter1_shape, './output/rbm_filters1.png')

    plot(v_gen, img_shape, 'output/rbm_reconstructed.png')


def parse_args():
    for arg in sys.argv[1:]:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")


if __name__ == "__main__":
    parse_args()
    if model_name == 'rbm':
        main_rbm()
    elif model_name == 'diffusion':
        main_diffusion()


