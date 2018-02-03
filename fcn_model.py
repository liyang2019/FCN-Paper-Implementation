from torch import nn
from torchvision import models


# This is implemented in full accordance with the original one (https://github.com/shelhamer/fcn.berkeleyvision.org)
class FCN8s(nn.Module):
  def __init__(self, num_classes, caffe=False):
    super(FCN8s, self).__init__()
    vgg = models.vgg16()
    features, classifier = list(vgg.features.children()), list(vgg.classifier.children())
    '''
    100 padding for 2 reasons:
        1) support very small input size
        2) allow cropping in order to match size of different layers' feature maps
    Note that the cropped part corresponds to a part of the 100 padding
    Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
    '''
    features[0].padding = (100, 100)

    for f in features:
        if 'MaxPool' in f.__class__.__name__:
            f.ceil_mode = True
        elif 'ReLU' in f.__class__.__name__:
            f.inplace = True

    self.features3 = nn.Sequential(*features[: 17])
    self.features4 = nn.Sequential(*features[17: 24])
    self.features5 = nn.Sequential(*features[24:])

    self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
    self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
    self.score_pool3.weight.data.zero_()
    self.score_pool3.bias.data.zero_()
    self.score_pool4.weight.data.zero_()
    self.score_pool4.bias.data.zero_()

    fc6 = nn.Conv2d(512, 4096, kernel_size=7)
    fc6.weight.data.copy_(classifier[0].weight.data.view(4096, 512, 7, 7))
    fc6.bias.data.copy_(classifier[0].bias.data)
    fc7 = nn.Conv2d(4096, 4096, kernel_size=1)
    fc7.weight.data.copy_(classifier[3].weight.data.view(4096, 4096, 1, 1))
    fc7.bias.data.copy_(classifier[3].bias.data)
    score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)
    score_fr.weight.data.zero_()
    score_fr.bias.data.zero_()
    self.score_fr = nn.Sequential(
        fc6, nn.ReLU(inplace=True), nn.Dropout(), fc7, nn.ReLU(inplace=True), nn.Dropout(), score_fr
    )

    self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
    self.upscore_pool4 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, bias=False)
    self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, bias=False)
    self.upscore2.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
    self.upscore_pool4.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 4))
    self.upscore8.weight.data.copy_(get_upsampling_weight(num_classes, num_classes, 16))

