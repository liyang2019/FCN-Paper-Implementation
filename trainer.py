from torch import optim
import torch
from dataset import ADE20KDataSet
from torch.utils.data import DataLoader
from fcn8s import FCN8s
import torch.nn.functional as F
from torch.autograd import Variable


class Trainer:
  def __init__(self, cuda, model, train_loader, val_loader, n_epochs, n_save, learning_rate):
    """
    Initialization for Trainer of FCN model for image segmentation.
    :param cuda: True is cuda available.
    :param model: The end-to-end FCN model.
    :param train_loader: The training dataset loader.
    :param val_loader: The validation dataset loader.
    :param n_epochs: The max number for epochs.
    :param n_save; Save every n_save iterations.
    :param learning_rate: The learning rate.
    """
    self.cuda = cuda
    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.n_epochs = n_epochs
    self.n_save = n_save
    self.optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    self.n_iter = 0

  def train_epoch(self):
    for img, seg, _ in self.train_loader:
      self.n_iter += 1
      self.optimizer.zero_grad()
      if self.cuda:
        img, seg = img.cuda(), seg.cuda()
      img, seg = Variable(img), Variable(seg)
      output = self.model(img)
      loss = self.loss(output, seg)
      loss /= len(img)
      loss.backward()
      self.optimizer.step()
      print(self.n_iter, float(loss.data))

  def train(self):
    for epoch in range(self.n_epochs):
      print('epoch', epoch)
      self.train_epoch()

  @staticmethod
  def loss(input, target, size_average=True):
    """
    The cross entropy loss function.
    :param size_average: True if loss averaged over minibatch.
    :param input: The output of the model, (n, c, h, w)
    :param target: The target, (n, h, w)
    :return: The loss.
    """
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) > 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target > 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, size_average=False)
    if size_average:
      loss /= mask.data.sum()
    return loss


def main():
  cuda = torch.cuda.is_available()

  num_classes = 21
  pretrained = True
  image_size = 32
  batch_size = 5
  n_epochs = 2
  n_save = 10
  learning_rate = 1e-2

  fcn = FCN8s(num_classes=num_classes, pretrained=pretrained)
  fcn = fcn.cuda() if cuda else fcn
  train_set = ADE20KDataSet('data', 'data/top020_train.txt', 'data/top020_class.txt', image_size, train=True)
  val_set = ADE20KDataSet('data', 'data/top020_val.txt', 'data/top020_class.txt', image_size, train=False)
  train_loader = DataLoader(train_set, batch_size, shuffle=True, drop_last=True)
  val_loader = DataLoader(val_set, len(val_set))
  trainer = Trainer(cuda, fcn, train_loader, val_loader, n_epochs, n_save, learning_rate)
  trainer.train()


if __name__ == '__main__':
  main()
