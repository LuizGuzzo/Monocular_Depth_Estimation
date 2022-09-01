import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from PIL import Image
from io import BytesIO
import random

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'depth': depth}

class RandomChannelSwap(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if not _is_pil_image(image): raise TypeError('img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth): raise TypeError('img should be PIL Image. Got {}'.format(type(depth)))
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])
        return {'image': image, 'depth': depth}

def loadZipToMem(zip_file):
    # carrega o arquivo zip para a memoria PRO TREINAMENTO
    print('Loading dataset zip file...', end='')
    from zipfile import ZipFile
    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train_raw = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    # nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    from sklearn.utils import shuffle
    nyu2_train_raw = shuffle(nyu2_train_raw, random_state=0) #treinamento
    # nyu2_test = shuffle(nyu2_test, random_state=0) #validaçao

    split = int(np.floor(.2*len(nyu2_train_raw))) #20% do dataset
    
    # nyu2_train = nyu2_train_raw[split:]
    # nyu2_test = nyu2_train_raw[:split]

    nyu2_train = nyu2_test = nyu2_train_raw # treinando errado para verificar se vale a pena..

    # if True: # modo de teste
    #     nyu2_train = nyu2_train[:100]
    #     nyu2_test = nyu2_test[:100]

    # data é o dicionario {Nome: imagem}
    # nyu2_train é a lista de nomes para treinamento
    print('Loaded ({0}) to train and ({1}) to validate.'.format(len(nyu2_train),len(nyu2_test)))
    return data, nyu2_train, nyu2_test


class depthDatasetMemory(Dataset):
    # All datasets are subclasses of torch.utils.data.Dataset i.e, they have __getitem__ 
    # and __len__ methods implemented. Hence, they can all be passed to a torch.utils.data.DataLoader
    # which can load multiple samples in parallel using torch.multiprocessing workers.
    # https://pytorch.org/vision/stable/datasets.html?highlight=torch%20utils%20dataset
    
    def __init__(self, data, nyu2_train, transform=None):
        # carrega o data set na classe
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __getitem__(self, idx):
        # retorna um sample transformado
        sample = self.nyu_dataset[idx]
        image = Image.open( BytesIO(self.data[sample[0]]) )
        depth = Image.open( BytesIO(self.data[sample[1]]) )
        sample = {'image': image, 'depth': depth}
        if self.transform: sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.nyu_dataset)

class ToTensor(object):
    # Convert a PIL Image or numpy.ndarray to tensor.
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html?highlight=totensor#torchvision.transforms.ToTensor
    def __init__(self,is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        image = self.to_tensor(image)

        depth = depth.resize((320, 240))

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:            
            depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if not(_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def getNoTransform(is_test=False):
    return transforms.Compose([
        ToTensor(is_test=is_test)
    ])

def getDefaultTrainTransform(): 
    #concatenação de transformações
    return transforms.Compose([
        RandomHorizontalFlip(),
        RandomChannelSwap(0.5),
        ToTensor()
    ])

def getTrainingTestingData(batch_size):
    data, nyu2_train, nyu2_test = loadZipToMem('CSVdata.zip')

    # Um conjunto de dados que serao utilizados para estimação dos parametros (treinamento) 2/3
    # Um para ajuste de parametros (validacao)  1/6
    # Um para teste 1/6 (intocado usado apenas para a avaliacao final)

    # Roda o set de treinamento todo, e calcula os RMSE pela validação,
    #  verifica se é melhor do que o anterior se sim atualiza a rede
    # após rodar todos os epochs calcula o RMSE usando o conjunto de teste separado

    # cria uma classe que ira ler da as imagens e realizar a transformação necessaria.
    transformed_training = depthDatasetMemory(data, nyu2_train, transform=getDefaultTrainTransform())
    transformed_testing = depthDatasetMemory(data, nyu2_test, transform=getNoTransform())

    # https://pytorch.org/vision/stable/datasets.html?highlight=torch%20utils%20dataset 
    dataloaders = {
        'train' : DataLoader(transformed_training, batch_size, shuffle=True),
        'val' : DataLoader(transformed_testing, batch_size, shuffle=False)
    }
    return dataloaders
