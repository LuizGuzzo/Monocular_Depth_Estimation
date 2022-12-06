from cmath import nan
import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter
import math

from matplotlib import pyplot as plt
from model_mobileV3_Unet_interpolado_small_newCRF import PTModel
from loss import Silog_loss_variance, SSIM
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize


# TODO:
#  > calcular o ETA par treinamento e outro ETA para a finalizacao do teste.
#  > calcular a acuracia após o treinamento ao rodar todo o banco de teste.

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=4, type=int, help='batch size') #12
    parser.add_argument('--cp', default=0, type=int, help='1 to enable usage of the last checkpoint')
    args = parser.parse_args()
    
    # Create model
    model = PTModel().cuda()

    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs

    # Load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    prefix = 'MobileNetV3_large'
    writer = SummaryWriter(comment='{}-e{}-bs{}-lr{}'.format(prefix, args.epochs, args.bs, args.lr), flush_secs=30)

    # Loss
    # Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y.
    l1_criterion = nn.L1Loss()
    SIlog = Silog_loss_variance()
    ssim = SSIM()


    epoch_interation = 0
    if args.cp == 1 :
        print("starting training from the last checkpoint")
        checkpoint = torch.load("./checkpoints/global_checkpoint.pth")
        # checkpoint = torch.load("./checkpoints/freeze-ep10-loss0.0427.pth")
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
        size_loader = len(train_loader)

        # Switch to train mode
        model.train()

        end = time.time()

        for loader_pos, sample_batched in enumerate(train_loader):

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
            # Normalize depth
            depth_n = DepthNorm( depth )

            prediction = model(image)

            # Compute the loss
            l_depth = l1_criterion(prediction, depth_n) # criterion that measures the mean absolute error (MAE) between each element in the input x and target y
            # print("l_depth.item():",l_depth.item())
            l_ssim = ssim(prediction, depth_n)
            # print("l_ssim.item():",l_ssim.item())
            l_SIlog = SIlog(prediction, depth_n)          

            loss = (1.0 * l_ssim) + (0.1 * l_depth) #+ (0.1 * l_SIlog)

            # if loss.data.item() != loss.data.item():
            #     import torchvision.transforms as vtransforms
                
            #     vtransforms.ToPILImage()(depth[0,:]).show(title="gt cuda")
            #     vtransforms.ToPILImage()(depth_n[0,:]).show(title="gt cuda")
            #     vtransforms.ToPILImage()(prediction[0,:]).show(title="pred gt cuda")
            #     print("error")

            # Update step
            optimizer.zero_grad() # The gradients are then set to zero before each update
            losses.update(loss.data.item(), image.size(0))
            loss.backward() # gradiants computed
            optimizer.step() # method, that updates the parameters.

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val*(size_loader - loader_pos))))
        
            # Log progress
            niter = epoch*size_loader+loader_pos
            if loader_pos % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                'ETA {eta}\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})'
                .format(epoch, loader_pos, size_loader, batch_time=batch_time, loss=losses, eta=eta))

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)

            if loader_pos % 300 == 0: # registra a imagem como ela esta sendo processada pela rede
                print("Recording epoch`s intermediate results. %300")
                LogProgress(model, writer, test_loader, niter)

        # Record epoch's intermediate results
        print("Recording epoch`s intermediate results")
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

        # Record checkpoint
        print("saving a checkpoint.")
        print("epoch:",epoch)
        print("loss:",loss.item())
        CHECK_PATH = f'checkpoints/global_checkpoint.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }, CHECK_PATH) 
        
        print("checkpoint saved at:",CHECK_PATH)
    
        


def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential)) # pega apenas uma amostra do batch, o tamanho é normalmente correspondente ao BatchSize
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
    
    prediction = DepthNorm( model(image) )
    # print("output_shape:",prediction.shape) # torch.Size([7,1,240,320])
    # print("output_data_shape:",prediction.data.shape) # torch.Size([7,1,240,320])
    # registrando no log o mapa estimado e a diferença com o GT

    output_colorized_grid = colorize(vutils.make_grid(prediction.data, nrow=6, normalize=False))
    diff_colorized_grid = colorize(vutils.make_grid(torch.abs(prediction-depth).data, nrow=6, normalize=False))
    # print("res_output_colorized:", output_colorized_grid.shape) # (3, 486, 1934)
    # print("diff_colorized_grid:", diff_colorized_grid.shape) # (3, 486, 1934)
    writer.add_image('Train.3.Ours', output_colorized_grid, epoch)
    writer.add_image('Train.3.Diff', diff_colorized_grid, epoch)
   
    writer.add_image('Train.3.Ours', output_colorized_grid, epoch)
    writer.add_image('Train.3.Diff', diff_colorized_grid, epoch)

    del image
    del depth
    del prediction

if __name__ == '__main__':
    main()
