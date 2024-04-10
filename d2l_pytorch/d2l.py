import torch
import time
import random
from torch import nn
from IPython import display
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.utils.data


class FlattenLayer(nn.Module):
  def __init(self):
    super(FlattenLayer, self).__init__()

  def forward(self, x):
    return x.view(x.shape[0], -1)


def set_figsize(figsize):
  use_svg_display()
  plt.rcParams["figure.figsize"] = figsize


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
    root="../Datasets/", train=True, download=True, transform=transforms.ToTensor()
  )
  mnist_test = torchvision.datasets.FashionMNIST(
    root="../Datasets/", train=False, download=True, transform=transforms.ToTensor()
  )
  num_workers = 4
  train_iter = torch.utils.data.DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
  )
  test_iter = torch.utils.data.DataLoader(
    mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers
  )
  return train_iter, test_iter


def evaluate_accuracy(data_iter, net, device=None):
  if device is None and isinstance(net, torch.nn.Module):
    device = list(net.parameters())[0].device

  acc_sum, n = 0.0, 0
  with torch.no_grad():
    for X, y in data_iter:
      if isinstance(net, torch.nn.Module):
        net.eval()
        acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
        net.train()
      else:
        if "is_training" in net.__code__.co_varnames:
          acc_sum += (net(X, is_trainning=False).argmax(dim=1) == y).float().sum().item()
        else:
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

    test_acc = evaluate_accuracy(test_iter, net)
    print(
      "epoch %d , loss %.4f , train acc %.3f, test acc %.3f"
      % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc)
    )


# 卷积运算
def corr2d(X, K):
  h, w = K.shape
  Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))

  for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
      Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
  return Y


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
  net = net.to(device)
  print("training on ", device)
  loss = torch.nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()

    for X, y in train_iter:
      X = X.to(device)
      y = y.to(device)
      y_hat = net(X)
      l = loss(y_hat, y)
      optimizer.zero_grad()
      l.backward()
      optimizer.step()
      train_l_sum += l.cpu().item()
      train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
      n += y.shape[0]
      batch_count += 1

    test_acc = evaluate_accuracy(test_iter, net)
    print(
      "epoch %d , loss %.4f, train acc %.3f, test acc %.3f time %.1f sec"
      % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start)
    )


def load_data_fashion_mnist_ch5(batch_size, resize=None, root: str = ""):
  trans = []

  if resize:
    trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transforms = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
      root=root, train=True, download=True, transform=transforms
    )
    mnist_test = torchvision.datasets.FashionMNIST(
      root=root, train=False, download=True, transform=transforms
    )

    train_iter = torch.utils.data.DataLoader(
      mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_iter = torch.utils.data.DataLoader(
      mnist_test, batch_size=batch_size, shuffle=False, num_workers=4
    )

  return train_iter, test_iter


def load_data_jay_lyrics(file_path: str):
  with open(file_path, "r", encoding="utf-8") as f:
    corpus_chars = f.read()

  corpus_chars.replace("\n", " ").replace("\r", " ")
  corpus_chars = corpus_chars[:10000]

  idx_to_char = list(set(corpus_chars))
  char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
  vocab_size = len(char_to_idx)
  corpus_indices = [char_to_idx[char] for char in corpus_chars]
  return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_random(corpus_indices, batch_size, num_steps, device=None):
  num_examples = (len(corpus_indices) - 1) // num_steps
  epoch_size = num_examples

  example_indices = list(range(num_examples))
  random.shuffle(example_indices)

  def _data(pos):
    return corpus_indices[pos : pos + num_steps]

  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  for i in range(epoch_size):
    i = i * batch_size
    batch_indices = example_indices[i : i + batch_size]
    X = [_data(j * num_steps) for j in batch_indices]
    Y = [_data(j * num_steps + 1) for j in batch_indices]
    yield (
      torch.tensor(X, dtype=torch.float32, device=device),
      torch.tensor(Y, dtype=torch.float32, device=device),
    )
