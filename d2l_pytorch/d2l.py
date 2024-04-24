import torch
import time
import random
import math
from torch import nn
from IPython import display
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import torch.nn.functional as F
import sys
import os
import collections
import torchtext.vocab as Vocab
from tqdm import tqdm


class FlattenLayer(nn.Module):
  def __init(self):
    super(FlattenLayer, self).__init__()

  def forward(self, x):
    return x.view(x.shape[0], -1)


class RNNModel(nn.Module):
  def __init__(self, rnn_layer, vocab_size):
    super(RNNModel, self).__init__()
    self.rnn = rnn_layer
    self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
    self.vocab_size = vocab_size
    self.dense = nn.Linear(self.hidden_size, vocab_size)
    self.state = None

  def forward(self, inputs, state):
    X = to_onehot(inputs, self.vocab_size)
    Y, self.state = self.rnn(torch.stack(X), state)
    output = self.dense(Y.view(-1, Y.shape[-1]))
    return output, self.state


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
      l1 = loss(y_hat, y)
      optimizer.zero_grad()
      l1.backward()
      optimizer.step()
      train_l_sum += l1.cpu().item()
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
  epoch_size = num_examples // batch_size

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


def data_iter_consecutive(corpus_indices, batch_size, num_steps, device=None):
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  corpus_indices = torch.tensor(corpus_indices, dtype=torch.float32, device=device)
  data_len = len(corpus_indices)
  batch_len = data_len // batch_size
  indices = corpus_indices[0 : batch_size * batch_len].view(batch_size, batch_len)
  epoch_size = (batch_len - 1) // num_steps

  for i in range(epoch_size):
    i = i * num_steps
    X = indices[:, i : i + num_steps]
    Y = indices[:, i + 1 : i + num_steps + 1]
    yield X, Y


def one_hot(x, n_class, dtype=torch.float32):
  x = x.long()
  res = torch.zeros(x.shape[0], n_class, dtype=dtype, device=x.device)
  res.scatter_(1, x.view(-1, 1), 1)
  return res


def to_onehot(X, n_class):
  return [one_hot(X[:, i], n_class) for i in range(X.shape[1])]


def grad_clipping(params, theta, device):
  norm = torch.tensor([0.0], device=device)
  for param in params:
    norm += (param.grad.data**2).sum()

  norm = norm.sqrt().item()
  if norm > theta:
    for param in params:
      param.grad.data *= theta / norm


def predict_rnn(
  prefix,
  num_chars,
  rnn,
  params,
  init_rnn_state,
  num_hiddens,
  vocab_size,
  device,
  idx_to_char,
  char_to_idx,
):
  state = init_rnn_state(1, num_hiddens, device)
  output = [char_to_idx[prefix[0]]]

  for t in range(num_chars + len(prefix) - 1):
    X = to_onehot(torch.tensor([[output[-1]]], device=device), vocab_size)
    (Y, state) = rnn(X, state, params)

    if t < len(prefix) - 1:
      output.append(char_to_idx[prefix[t + 1]])
    else:
      output.append(int(Y[0].argmax(dim=1).item()))

  return "".join([idx_to_char[i] for i in output])


def train_and_predict_rnn(
  rnn,
  get_params,
  init_rnn_state,
  num_hiddens,
  vocab_size,
  device,
  corpus_indices,
  idx_to_char,
  char_to_idx,
  is_random_iter,
  num_epochs,
  num_steps,
  lr,
  clipping_theta,
  batch_size,
  pred_period,
  pred_len,
  prefixes,
):
  if is_random_iter:
    data_iter_fn = data_iter_random
  else:
    data_iter_fn = data_iter_consecutive

  params = get_params()
  loss = nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
    if not is_random_iter:
      state = init_rnn_state(batch_size, num_hiddens, device)

    l_sum, n, start = 0.0, 0, time.time()
    data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, device)
    for X, Y in data_iter:
      if is_random_iter:
        state = init_rnn_state(batch_size, num_hiddens, device)
      else:
        for s in state:
          s.detach_()
      inputs = to_onehot(X, vocab_size)
      (outputs, state) = rnn(inputs, state, params)
      outputs = torch.cat(outputs, dim=0)
      y = torch.transpose(Y, 0, 1).contiguous().view(-1)

      l1 = loss(outputs, y.long())

      if params[0].grad is not None:
        for param in params:
          param.grad.data.zero_()
      l1.backward()
      grad_clipping(params, clipping_theta, device)
      sgd(params, lr, 1)
      l_sum += l1.item() * y.shape[0]
      n += y.shape[0]

    if (epoch + 1) % pred_period == 0:
      print(
        "epoch %d, perplexity %f, time %.2f sec"
        % (epoch + 1, math.exp(l_sum / n), time.time() - start)
      )
      for prefix in prefixes:
        print(
          " -",
          predict_rnn(
            prefix,
            pred_len,
            rnn,
            params,
            init_rnn_state,
            num_hiddens,
            vocab_size,
            device,
            idx_to_char,
            char_to_idx,
          ),
        )


def predict_rnn_pytorch(prefix, num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
  state = None
  output = [char_to_idx[prefix[0]]]
  for t in range(num_chars + len(prefix) - 1):
    X = torch.tensor([output[-1]], device=device).view(1, 1)
    if state is not None:
      if isinstance(state, tuple):
        state = (state[0].to(device), state[1].to(device))
      else:
        state = state.to(device)
    (Y, state) = model(X, state)
    if t < len(prefix) - 1:
      output.append(char_to_idx[prefix[t + 1]])
    else:
      output.append(int(Y.argmax(dim=1).item()))

  return "".join([idx_to_char[i] for i in output])


def train_and_predict_rnn_pytorch(
  model,
  num_hiddens,
  vocab_size,
  device,
  corpus_indices,
  idx_to_char,
  char_to_idx,
  num_epochs,
  num_steps,
  lr,
  clipping_theta,
  batch_size,
  pred_period,
  pred_len,
  prefixes,
):
  loss = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  model.to(device)
  state = None
  for epoch in range(num_epochs):
    l_sum, n, start = 0.0, 0, time.time()
    data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)
    for X, Y in data_iter:
      if state is not None:
        if isinstance(state, tuple):
          state = (state[0].detach(), state[1].detach())
        else:
          state = state.detach()

      (output, state) = model(X, state)
      y = torch.transpose(Y, 0, 1).contiguous().view(-1)
      l1 = loss(output, y.long())
      optimizer.zero_grad()
      l1.backward()
      grad_clipping(model.parameters(), clipping_theta, device)
      optimizer.step()
      l_sum += l1.item() * y.shape[0]
      n += y.shape[0]

    try:
      perplexity = math.exp(l_sum / n)
    except OverflowError:
      perplexity = float("inf")
    if (epoch + 1) % pred_period == 0:
      print(
        " epoch %d, perplexity %f , time %.2f sec " % (epoch + 1, perplexity, time.time() - start)
      )
      for prefix in prefixes:
        print(
          " -",
          predict_rnn_pytorch(
            prefix, pred_len, model, vocab_size, device, idx_to_char, char_to_idx
          ),
        )


def read_imdb(folder="train", data_root: str = ""):
  data = []
  for label in ["pos", "neg"]:
    folder_name = os.path.join(data_root, folder, label)

    for file in tqdm(os.listdir(folder_name)):
      with open(os.path.join(folder_name, file), "rb") as f:
        review = f.read().decode("utf-8").replace("\n", "").lower()
        data.append([review, 1 if label == "pos" else 0])

  random.shuffle(data)
  return data


# 预处理数据
# 分词
def get_tokenized_imdb(data):
  """
  data: list of [string, label]
  """

  def tokenizer(text):
    return [tok.lower() for tok in text.split(" ")]

  return [tokenizer(review) for review, _ in data]


# 创建字典
def get_vocab_imdb(data):
  tokenized_data = get_tokenized_imdb(data)
  counter = collections.Counter([tk for st in tokenized_data for tk in st])
  return Vocab.Vocab(counter, min_freq=5)


def preprocess_imdb(data, vocab):
  max_l = 500

  def pad(x):
    return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

  tokenized_data = get_tokenized_imdb(data)
  features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
  labels = torch.tensor([score for _, score in data])
  return features, labels


def load_pretrained_embedding(words, pretrained_vocab):
  """从预训练好的vocab中提取出words对应的词向量"""
  embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0])  # 初始化为0
  oov_count = 0  # out of vocabulary
  for i, word in enumerate(words):
    try:
      idx = pretrained_vocab.stoi[word]
      embed[i, :] = pretrained_vocab.vectors[idx]
    except KeyError:
      oov_count += 1

  if oov_count > 0:
    print("There are %d oov words." % oov_count)
  return embed


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
  net = net.to(device)
  print("training on ", device)
  batch_count = 0
  for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()

    for X, y in train_iter:
      X = X.to(device)
      y = y.to(device)
      y_hat = net(X)
      l1 = loss(y_hat, y)
      optimizer.zero_grad()
      l1.backward()
      optimizer.step()
      train_l_sum += l1.cpu().item()
      train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
      n += y.shape[0]
      batch_count += 1

    test_acc = evaluate_accuracy(test_iter, net)
    print(
      "epoch %d, loss %.4f, train acc %.3f, test acc %.3f , time %.1f sec"
      % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start)
    )


def predict_sentiment(net, vocab, sentence):
  """sentence是词语的列表"""
  device = list(net.parameters())[0].device
  sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
  label = torch.argmax(net(sentence.view((1, -1))), dim=1)
  return "positive" if label.item() == 1 else "negative"
