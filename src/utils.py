import matplotlib
import matplotlib.cm
import numpy as np

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#TODO: remover comentarios após identificar como printar a imagem
def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # print("v-.-.-.-.-v")
    # print("Colorize_value_shape:",value.shape)
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    # print("Colorize_normalizado_value_shape:",value.shape)
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)
    # print("Colorize_cmapper_value_shape:",value.shape)

    img = value[:,:,:3]
    # print("Colorize_image_shape:",value.shape)
    img = img.transpose((2,0,1))
    # print("Colorize_transposto_value_shape:",value.shape)
    # print("^-.-.-.-.-^")

    return img