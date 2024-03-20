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
