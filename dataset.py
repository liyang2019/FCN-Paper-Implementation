import os
from torch.utils.data import Dataset
from torchvision import transforms
import random
from scipy.misc import imread, imresize, imshow
import numpy as np
import torch


class ADE20KDataSet(Dataset):

  def __init__(self, file, root, size, max_sample=-1, train=True):
    """
    Initialization.
    :param file: The filename of the image samples txt file.
    :param root: The folder root of the image samples.
    :param size: The image and segmentation size after scale and crop for training.
    :param max_sample: The max number of samples.
    :param train: True if is training.
    """
    self.root = root
    self.size = size
    self.train = train

    # mean and std using ImageNet mean and std.
    self.img_transform = transforms.Compose([
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])])

    self.list_sample = [x.strip('\n') for x in open(file, 'r').readlines()]

    if self.train:
      random.shuffle(self.list_sample)

    if max_sample > 0:
      self.list_sample = self.list_sample[0: max_sample]

    num_sample = len(self.list_sample)

    assert num_sample > 0

  @staticmethod
  def _scale_and_crop(img, seg, crop_size, train):
    """
    scale crop the image to make every image of the same square size, H = W = crop_size
    :param img: The image.
    :param seg: The segmentation of the image.
    :param crop_size: The crop size.
    :param train: True if is training.
    :return: The cropped image and segmentation.
    """
    h, w = img.shape[0], img.shape[1]

    if train:
      # random scale
      scale = random.random() + 0.5  # 0.5-1.5
      scale = max(scale, 1. * crop_size / (min(h, w) - 1))  # ??
    else:
      # scale to crop size
      scale = 1. * crop_size / (min(h, w) - 1)

    img_scale = imresize(img, scale, interp='bilinear')
    seg_scale = imresize(seg, scale, interp='nearest')

    h_s, w_s = img_scale.shape[0], img_scale.shape[1]
    if train:
      # random crop
      x1 = random.randint(0, w_s - crop_size)
      y1 = random.randint(0, h_s - crop_size)
    else:
      # center crop
      x1 = (w_s - crop_size) // 2
      y1 = (h_s - crop_size) // 2

    img_crop = img_scale[y1: y1 + crop_size, x1: x1 + crop_size, :]
    seg_crop = seg_scale[y1: y1 + crop_size, x1: x1 + crop_size]
    return img_crop, seg_crop

  @staticmethod
  def _flip(img, seg):
    img_flip = img[:, ::-1, :]
    seg_flip = seg[:, ::-1]
    return img_flip, seg_flip

  def __getitem__(self, index):
    """
    Get image from file.
    :param index: The index of the image.
    :return: The image, segmentation, and the image base name.
    """
    img_basename = self.list_sample[index]
    path_img = os.path.join(self.root, img_basename + '.jpg')
    path_seg = os.path.join(self.root, img_basename + '_seg.png')
    assert os.path.exists(path_img), '[{}] does not exist'.format(path_img)
    assert os.path.exists(path_seg), '[{}] does not exist'.format(path_seg)
    # load image and label
    try:
      img = imread(path_img, mode='RGB')
      seg = imread(path_seg, mode='RGB')
      assert (img.ndim == 3)
      assert (seg.ndim == 3)
      assert (img.shape[0] == seg.shape[0])
      assert (img.shape[1] == seg.shape[1])

      # random scale, crop, flip
      if self.size > 0:
        img, seg = self._scale_and_crop(img, seg, self.size, self.train)
        if random.choice([-1, 1]) > 0:
          img, seg = self._flip(img, seg)

      # image to float
      img = img.astype(np.float32) / 255.
      img = img.transpose((2, 0, 1))

      # segmentation to integer encoding according to
      # the loadAde20K.m file in http://groups.csail.mit.edu/vision/datasets/ADE20K/
      seg = np.round(seg[:, :, 0] / 10.) * 256 + seg[:, :, 1]

      # label to int from -1 to 149
      seg = seg.astype(np.int) - 1

      # to torch tensor
      image = torch.from_numpy(img)
      segmentation = torch.from_numpy(seg)

    except Exception as e:
      print('Failed loading image/segmentation [{}]: {}'.format(path_img, e))
      # dummy data
      image = torch.zeros(3, self.size, self.size)
      segmentation = -1 * torch.ones(self.size, self.size).long()
      return image, segmentation, img_basename

      # substracted by mean and divided by std
    image = self.img_transform(image)

    return image, segmentation, img_basename

  def __len__(self):
    """
    Get the length of the dataset.
    :return: The length of the dataset.
    """
    return len(self.list_sample)


def main():
  ade20k = ADE20KDataSet('data/filenames020.txt', 'data', 128)
  image, segmentation, img_basename = ade20k.__getitem__(1)
  print(image)
  print(segmentation)
  print(img_basename)


if __name__ == '__main__':
  main()
