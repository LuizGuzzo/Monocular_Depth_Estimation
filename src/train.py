import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter
import math
import copy

from matplotlib import pyplot as plt
from model_mobileV3 import PTModel
from loss import ssim
from data import getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorize


# TODO:
#  > Fazer um avaliador da rede
#  > calcular o ETA para treinamento e outro ETA para validacão a cada epoch
#  > verificar acertividade do modelo in training apenas para registrar no log (uma pequena amostra) %300, seria calcular o loss?
#  >  


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=7, type=int, help='batch size')
    parser.add_argument('--cp', default=1, type=int, help='1 to enable usage of the last checkpoint')
    args = parser.parse_args()
    
    # Create model
    model = PTModel().cuda()

    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam( model.parameters(), args.lr )
    batch_size = args.bs
    

    # Load data
    dataloaders = getTrainingTestingData(batch_size=batch_size)

    # Logging
    prefix = 'MobileNetV3_large'
    writer = SummaryWriter(comment='{}-e{}-bs{}-lr{}'.format(prefix, args.epochs, args.bs, args.lr), flush_secs=30)

    # Loss
    # Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y.
    l1_criterion = nn.L1Loss()


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

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                print(">> training mode started <<")
                model.train()  # Set model to training mode
            else:
                print("<< validation mode started >>")
                model.eval()   # Set model to evaluate mode


            batch_time = AverageMeter() # medidores
            losses = AverageMeter()
            size_loader = len(dataloaders[phase])

            # Switch to train mode
            model.train()

            end = time.time()

            for loader_pos, sample_batched in enumerate(dataloaders[phase]):

                # Prepare sample and target
                image = torch.autograd.Variable(sample_batched['image'].cuda())
                depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

                # Normalize depth
                depth_n = DepthNorm( depth )

                # Predict
                output = model(image)

                optimizer.zero_grad() # The gradients are then set to zero before each update

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    if phase == 'train':
                            
                        # Compute the loss
                        l_depth = l1_criterion(output, depth_n) # criterion that measures the mean absolute error (MAE) between each element in the input x and target y
                        # print("l_depth.item():",l_depth.item())
                        l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
                        # print("l_ssim.item():",l_ssim.item())

                        loss = (1.0 * l_ssim) + (0.1 * l_depth) # l_depth tem peso 0.1 e l_ssim tem peso 1?

                        # Update step
                        
                        losses.update(loss.data.item(), image.size(0))
                        loss.backward() # gradiants computed
                        optimizer.step() # method, that updates the parameters.

                    else:
                        # validation part
                        continue

                    # Measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
                    eta = str(datetime.timedelta(seconds=int(batch_time.val*(size_loader - loader_pos))))
                
                    # Log progress
                    niter = epoch*size_loader+loader_pos
                    if loader_pos % 5 == 0:
                        # Print to console
                        print(
                            'MODE: {mode}\t'
                            'Epoch: [{0}][{1}/{2}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                            'ETA {eta}\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})'
                            .format(epoch, loader_pos, size_loader, batch_time=batch_time, loss=losses, eta=eta, mode=phase))

                    if phase == 'train':
                        if loader_pos % 5 == 0:
                            # Log to tensorboard
                            writer.add_scalar('{}}/Loss'.format(phase), losses.val, niter)
                        if loader_pos % 300 == 0:
                            print("Recording epoch`s intermediate results.")
                            LogProgress(model, writer, dataloaders['val'], niter)

                # calcular as estatisticas
                # na etapa de validacao? ai qnd terminar ela verifica se é melhor do q a ultima validacao?
                

                #deep copy the best model
                if 1 and phase == 'val': # o modelo é o melhor e esta em processo de validacao:
                    best_model = copy.deepcopy(model.state_dict())
                    LogProgress(model, writer, dataloaders['val'], niter) #modelo eleito é registrado o teste
                    writer.add_scalar('Best/Loss.avg', losses.avg, epoch) # grava o loss medio do treinamento


                if phase == 'train':
                    print("Recording epoch`s avg results")
                    LogProgress(model, writer, dataloaders['val'], niter) 
                    writer.add_scalar('{}/Loss.avg'.format(phase), losses.avg, epoch) # grava o loss medio do treinamento

                # Record checkpoint
                #TODO: atualizar para salvar o melhor modelo
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

                #TODO: printar treinamento completo e o valor de acuracia do melhor modelo (devo armazenar isso num log?)
                #TODO: carregar o peso do melhor modelo
                
            

    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # ensina a salvar a rede e usar a q tem melhor validacao
    # Creates a validation criterion
    # MSE_criterion = nn.MSELoss()
    # Old_RMSE = 0
    
    # # """
    # # validate the NN trained

    # # average relative error (rel)
    # # root mean squared error (rms) nn.MSELoss + root
    # # average (log10) error
    # # threshold accuracy (δi)

    # # calcular a avaliação da rede recem treinada pelo todo dataset de avaliaçao
    # # e compara-lo com a ultima avaliação feita, a melhor avaliação é mantida a rede
    # # se nao existe rede passada (é o epoch 0) apenas continue

    # # validation step
    # model.eval()
    # MSE = []
    # actual = []
    # pred = []
    # for i, sample_batched in enumerate(test_loader):
        
    #     image = torch.autograd.Variable(sample_batched['image'].cuda())
    #     depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

    #     depth_n = DepthNorm( depth ) # GT
    #     output = model(image) # predicted GT

    #     MSE.append(MSE_criterion(output, depth_n))

    #     del image
    #     del depth
    #     del output

    #     #somar todos MSE

    # print("MSE.size:",len(MSE))
    # RMSE = 10 # nem terminei pq ja ta dando estouro de memoria
    # print("RMSE:",RMSE)
    # print("Old_RMSE:",Old_RMSE)

    # if (RMSE < Old_RMSE) or (epoch == 0): # o antigo é pior que o atual OU é epoch zero
    #     # atualiza o antigo para ser o atual
    #     print("New_model wins. Overwriting the Old_model")
    #     Old_Model = model
    #     Old_optimzer = optimizer
    #     Old_MSE_criterion = MSE_criterion
    #     Old_loss = loss
    #     Old_RMSE = RMSE
    # else:
    #     # antigo é melhor que o atual, atual recebe o antigo
    #     print("Old_model wins. Overwriting the New_model")
    #     model = Old_Model
    #     optimizer = Old_optimzer
    #     MSE_criterion = Old_MSE_criterion
    #     loss = Old_loss
    #     RMSE = Old_RMSE


    # # """
        


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
   
    writer.add_image('Train.3.Ours', output_colorized_grid, epoch)
    writer.add_image('Train.3.Diff', diff_colorized_grid, epoch)

    del image
    del depth
    del output

if __name__ == '__main__':
    main()
