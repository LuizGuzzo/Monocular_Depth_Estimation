import matplotlib
import matplotlib.cm
import numpy as np
from PIL import Image

def DepthNorm(depth, maxDepth=1000.0): 
    # return maxDepth / depth
    return (depth - depth.min())/(depth.max()-depth.min())

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


# https://github.com/aliyun/NeWCRFs/blob/a6b6ab0abc3766809380da80850f1553b05755a3/newcrfs/utils.py
# apenas utilizam a mascara GT aparentemente pro calculo : https://github.com/aliyun/NeWCRFs/blob/a6b6ab0abc3766809380da80850f1553b05755a3/newcrfs/eval.py
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = np.mean(thresh < 1.25) # como tirar a media de um lista de booleanos em pytorch
    d2 = np.mean(thresh < 1.25 ** 2)
    d3 = np.mean(thresh < 1.25 ** 3)

    rms = (gt - pred) ** 2
    rms = np.sqrt(np.mean(rms))

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(np.mean(log_rms))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]

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

def colorizeCPU(value, cmap='plasma'):

    image = value

    image = image.astype(np.float32) # convert to float
    image -= image.min() # ensure the minimal value is 0.0
    image /= image.max() # maximum value in image is now 1.0

    cm = matplotlib.cm.get_cmap(cmap)
    image = Image.fromarray(np.uint8(cm(image)*255))

    return image