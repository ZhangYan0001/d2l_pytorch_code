import torch
from IPython import display
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.utils.data


def use_svg_display():
  display.set_matplotlib_formats("svg")


def sgd(params, lr, batch_size):
  for param in params:
    param.data -= lr * param.grad / batch_size


def show_fashion_mnist(images, labels):
  display.set_matplotlib_formats("svg")
  _, figs = plt.subplots(1, len(images), figsize=(12, 12))
  for f, img, lbl in zip(figs, images, labels):
    f.imshow(img.view((28, 28)).numpy())
    f.set_title(lbl)
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)

  plt.show()


# 将数值标签转换成相对应的文本标签
def get_fashion_mnist_labels(labels):
  text_labels = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
  ]
  return [text_labels[int(i)] for i in labels]


def load_data_fashion_mnist(batch_size: int):
  mnist_train = torchvision.datasets.FashionMNIST(
    root="./Datasets/", train=True, download=True, transform=transforms.ToTensor()
  )
  mnist_test = torchvision.datasets.FashionMNIST(
    root="./Datasets/", train=False, download=True, transform=transforms.ToTensor()
  )
  num_workers = 4
  train_iter = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
  )
  test_iter = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
  )
  return train_iter, test_iter


def evaluate_accuracy(data_iter, net):
  acc_sum, n = 0.0, 0
  for X, y in data_iter:
    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
    n += y.shape[0]

  return acc_sum / n


def train_ch3(
  net,
  train_iter,
  test_iter,
  loss,
  num_epochs,
  batch_size,
  params=None,
  lr=None,
  optimizer=None,
):
  for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
      y_hat = net(X)
      l1 = loss(y_hat, y).sum()

      if optimizer is not None:
        optimizer.zero_grad()
      elif params is not None and params[0].grad is not None:
        for param in params:
          param.grad.data.zero_()

      l1.backward()

      if optimizer is None:
        sgd(params, lr, batch_size)
      else:
        optimizer.step()
      train_l_sum += l1.item()
      train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
      n += y.shape[0]

    test_acc = evaluate_accuracy(test_iter,net)
    print(
      "epoch %d , loss %.4f , train acc %.3f, test acc %.3f"
      % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc)
    )
