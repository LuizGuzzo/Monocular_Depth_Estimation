import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter
import os

from matplotlib import pyplot as plt
from model import PTModel
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize


# TODO:
#  > DONE corrigir bug q o epoch nao esta sendo salvo com o modelo de checkpoint
#  > DONE corrigir bug de log
#  > fazer um programa que testa o modelo - HARD
#  > verificar como que ta sendo usado o batch size
#  > alterar o metodo de avaliação que ta usando 654 testes para 50688 treinos

# Duvidas:
# gradiente é necessariamente oque? a curva? a tecnica? quanto menor gradiente significa menor erro?
# porque o loss aumenta quando termina o epoch e depois volta ao normal?

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=7, type=int, help='batch size')
    parser.add_argument('--cp', default=0, type=int, help='1 to enable usage of the last checkpoint')
    args = parser.parse_args()
    
    # Create model
    model = PTModel().cuda()

    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs
    prefix = 'MobileNet_' + str(batch_size)

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Loss
    # Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y.
    l1_criterion = nn.L1Loss()

    epoch_interation = 0
    if args.cp == 1 :
        print("starting training from the last checkpoint")
        checkpoint = torch.load("./checkpoints/global_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_interation = checkpoint['epoch']
        loss = checkpoint['loss']
        print("epoch started:",epoch_interation)
        print("loss started:",loss.item())



    # Start training...
    for epoch in range(epoch_interation,args.epochs,1):
        batch_time = AverageMeter() # medidores
        losses = AverageMeter()
        N = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad() 

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm( depth )

            # Predict
            output = model(image)

            # Compute the loss
            l_depth = l1_criterion(output, depth_n)
            l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)

            loss = (1.0 * l_ssim) + (0.1 * l_depth) # Se loss é uma variavel.. como que ele tem .backward()?

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward() # gradiants computed
            optimizer.step() # method, that updates the parameters.

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        
            # Log progress
            niter = epoch*N+i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if i % 300 == 0: # a cada 300 imagens processadas
                print("Recording epoch`s intermediate results. %300")
                LogProgress(model, writer, test_loader, niter)
                
            if i == 100: break # testing

        # Record epoch's intermediate results
        print("Recording epoch`s intermediate results")
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

        # Record checkpoint
        print("saving a checkpoint.")
        print("epoch:",epoch)
        print("loss:",loss.item())
        CHECK_PATH = f'checkpoints/global_checkpoint_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, CHECK_PATH) 
        
        print("checkpoint saved at:",CHECK_PATH)
        
        

# TODO: como que eu consigo printar uma imagem após o processamento?
# não consegui extrair a imagem pura para dar um img.show()
# TODO: verificar se as imagens selecionadas no test_loader são as mesmas sempre..
def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    # print("image_shape:",image.shape) # torch.Size([7, 3, 480, 640])
    # print("image_data_shape:",image.data.shape) # torch.Size([7, 3, 480, 640])
    # registrando no log, a Imagem original + a GT

    image_grid = vutils.make_grid(image.data, nrow=6, normalize=True)
    depth_colorized_grid = colorize(vutils.make_grid(depth.data, nrow=6, normalize=False))
    # print("image_grid:", image_grid.shape) # torch.Size([3, 966, 3854])
    # print("depth_colorized_grid:", depth_colorized_grid.shape) # (3, 486, 1934)
    if epoch == 0: writer.add_image('Train.1.Image', image_grid , epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', depth_colorized_grid, epoch)
    
    output = DepthNorm( model(image) )
    # print("output_shape:",output.shape) # torch.Size([7,1,240,320])
    # print("output_data_shape:",output.data.shape) # torch.Size([7,1,240,320])
    # registrando no log o mapa estimado e a diferença com o GT

    output_colorized_grid = colorize(vutils.make_grid(output.data, nrow=6, normalize=False))
    diff_colorized_grid = colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False))
    # print("res_output_colorized:", output_colorized_grid.shape) # (3, 486, 1934)
    # print("diff_colorized_grid:", diff_colorized_grid.shape) # (3, 486, 1934)
    writer.add_image('Train.3.Ours', output_colorized_grid, epoch)
    writer.add_image('Train.3.Diff', diff_colorized_grid, epoch)

    # plt.imshow(output_colorized_grid)#,interpolation='nearest')
    del image
    del depth
    del output

if __name__ == '__main__':
    main()
