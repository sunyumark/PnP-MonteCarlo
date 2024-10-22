import matplotlib.pyplot as plt
import torch

def save_image(fname, image, colormap='gray', vrange=[None, None]):
    if torch.is_tensor(image):
        image = image.detach().cpu()
        if torch.is_complex(image):
            image = image.abs()
    
    plt.figure()

    # grayscale image
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[-1] == 1):
        plt.imshow(image, cmap=colormap, vmin=vrange[0], vmax=vrange[1])
        plt.colorbar()

    # RGB image
    elif len(image.shape) == 3 and image.shape[-1] == 3:
        # convert to [0, 1]
        image = (image+1)/2
        # clip to [0, 1]
        image = torch.clamp(image, 0, 1)
        plt.imshow(image)
        
    # wrong image shape
    else:
        raise RuntimeError('Wrong image shape')
    
    plt.savefig(fname)
    plt.close()