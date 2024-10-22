import torch

def normalize_image(fname, image, vrange=[None, None]):
    assert len(image.shape) > 3, 'Wrong image shape'
    return image / image.max()

    