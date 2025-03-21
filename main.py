"""
This training script is designed for a poor person with only a laptop to train 
RBM and Diffusion models on MNIST or CIFAR10 datasets.

You can specify in command line:
python3 main.py --model_name=rbm --dataset_name=mnist --n_epochs=100

"""

import os.path
import torch
from modules.stacked_rbm import StackedRBM
from modules.diffusion import Diffusion
from modules.unet import Unet
from ast import literal_eval
import sys
import mlflow

from utils import plot, load_dataset, Trainer, setup_logging

dataset_name = 'mnist'  # 'mnist' or 'cifar10'
model_name = 'rbm' # 'rbm' or 'diffusion' or 'fm'
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


# flow matching
def main_fm(train_dataset, checkpt_file, img_shape):
    if dataset_name == 'mnist':
        n_channels = 1
    elif dataset_name == 'cifar10':
        n_channels = 3

    model = Unet(n_channels, t_emb_dim=128, device=device)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    trainer = Trainer(model, dataset_name, checkpt=checkpt_file, n_epochs=n_epochs, 
                      lr=learning_rate, batch_size=batch_size, device=device)
    model = trainer.train_fm(train_loader)
    sample_batch_size = 25

    # image gen from noise
    if dataset_name == 'mnist':
        x = torch.randn((sample_batch_size, 1, 28, 28))
    elif dataset_name == 'cifar10':
        x = torch.randn((sample_batch_size, 3, 32, 32))
    x = x.to(device)
    with torch.no_grad():
        flow = model(x, torch.as_tensor(0).unsqueeze(0).to(device))
        x = x - flow
    plot(x, img_shape, './output/fm_gen.png')


# denoising diffusion
def main_diffusion(train_dataset, checkpt_file, img_shape):
    if dataset_name == 'mnist':
        n_channels = 1
    elif dataset_name == 'cifar10':
        n_channels = 3

    model = Unet(n_channels, t_emb_dim=128, device=device)

    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    trainer = Trainer(model, dataset_name, checkpt=checkpt_file, n_epochs=n_epochs, 
                      lr=learning_rate, batch_size=batch_size, device=device)
    model = trainer.train_diffusion(train_loader)
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


# restricted boltzmann machine
def main_rbm(train_dataset, checkpt_file, img_shape):
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

    trainer = Trainer(model, dataset_name, checkpt=checkpt_file, n_epochs=n_epochs, 
                      lr=learning_rate, batch_size=batch_size, device=device)
    
    model = trainer.train_rbm(train_loader)

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
    experiment_name = model_name + '_' + dataset_name
    checkpt = 'chk_' + experiment_name + '.pt'
    checkpt_file = os.path.join(output_dir, checkpt)    
    # Initialize logging
    setup_logging(output_dir)    
    train_dataset, img_shape = load_dataset(dataset_name)

    # Set up MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    if model_name == 'rbm':
        main_rbm(train_dataset, checkpt_file, img_shape)
    elif model_name == 'diffusion':
        main_diffusion(train_dataset, checkpt_file, img_shape)
    elif model_name == 'fm':
        main_fm(train_dataset, checkpt_file, img_shape)


