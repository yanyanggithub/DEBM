# Deep Energy-Based Models (DEBM)
Here is a collection of simplistic energy based models that took inspiration from ideas in thermodynamics for image generation, including Boltzmann machines and denoising diffusion. This repository is designed with the aim of enabling individuals with modest laptop GPUs (or no GPU) to train stacked Restricted Boltzmann machines (RBM) and Diffusion models on MNIST or CIFAR10 datasets.

You can specify in command line:
```
python3 main.py --model_name=rbm --dataset_name=mnist --n_epochs=100
```

RBM often is useful to learn image features. For 100 epochs of default model at default learning rate, you will see something like this.

![image](assets/rbm_filter.png)

RBM learns to reconstructed an image from hidden -> visible layers, that is not like generating an image from a text prompt. You will see a blurry reconstruction from the original. The reconstruction is learned by **unsupervised learning** i.e. contrastive divergence (CD). In order to achieve better results, you want to employ stacked RBMs with more hidden layers and multiple CD processes.

![image](assets/rbm_resconstructed.png)

Diffusion is a denoising probabilistic model. It is more powerful but is a scored-based model (somewhat different to just energy). Training diffusion is slow, you can give a try in command line:
```
python3 main.py --model_name=diffusion --dataset_name=mnist --n_epochs=100 --learning_rate=0.001
```
Here is an example of a generated image from denosing diffusion probabilistic models after 100 epochs.

![image](assets/diffusion.png)


