import os.path
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from torchvision import transforms
from agents.stacked_rbm import StackedRBM


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
              checkpt="output/checkpoint.pt", n_epochs=10, 
              lr=0.01, batch_size=64):
    checkpt_epoch = 0
    if os.path.isfile(checkpt):
        checkpoint = torch.load(checkpt, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'])
        checkpt_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
    model.train()

    for epoch in range(n_epochs):
        if epoch and epoch % 10 == 0:
            # learning rate decay
            lr = lr * 0.9
        for _, (data, _) in enumerate(train_loader):
            input = data.view(-1, model.n_visible)
            loss = model.fit(input, lr=lr, batch_size=batch_size)

        epoch_ = epoch + checkpt_epoch
        checkpoint = {
            'epoch': epoch_,
            'state_dict': model.state_dict(),
            'lr': lr
        }

        torch.save(checkpoint, checkpt)
        print('Epoch %d\t Loss=%.4f' % (epoch_, loss))

    return model


def main():
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
    _, v_gen = model(images.view(-1, model.n_visible))

    # plot the results
    plot(w0, img_shape, 'output/filters0.png')
    plot(w1, [16, 16], 'output/filters1.png')

    plot(v_gen, img_shape, 'output/gen_img.png')


if __name__ == "__main__":
    main()


