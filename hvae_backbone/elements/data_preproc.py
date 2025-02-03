import torchvision.transforms as transforms

from .. import params


def default_transform():
    return lambda x: transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(params.data_params.shape),
        #Normalize(),
        #MinMax(),
    ])(x)


class Normalize(object):
    def __call__(self, img):
        """
        Normalize image to be between [0, 1]

        :param img: PIL): Image
        :return: Normalized image
        """
        return img


class MinMax(object):
    def __call__(self, img):
        """
        Normalize image to be between [-1, 1]

        :param img: PIL): Image
        :return: Tensor
        """
        return img