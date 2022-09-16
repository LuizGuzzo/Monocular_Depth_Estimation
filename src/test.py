import time
import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torchvision.utils as vutils    
from tensorboardX import SummaryWriter
import math
import numpy as np
import torchvision.transforms as vtransforms

from matplotlib import pyplot as plt
from model_mobileV3 import PTModel
from loss import ssim
from data import getTestingData, getTrainingTestingData
from utils import AverageMeter, DepthNorm, colorizeCPU, compute_errors


# TODO:
#  > Fazer um avaliador da rede 
#  > calcular o ETA par treinamento e outro ETA para a finalizacao do teste.
#  > calcular a acuracia após o treinamento ao rodar todo o banco de teste.
#  > mexer no data.py para ter a opção de carregar APENAS o banco de teste, sem precisar carregar o de treinamento junto

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation via Transfer Learning')
    parser.add_argument('--bs', '--batch-size', default=10, type=int, help='batch size')
    parser.add_argument('--pt', '--path', default="./checkpoints/global_checkpoint.pth", type=str, help='path to the model') # mob3L-ep3-loss0.040_10kDS.pth
    
    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
    
    args = parser.parse_args()
    
    
    # Starting model
    PATH = args.pt
    print("getting the Model from: ",PATH)
    checkpoint = torch.load(PATH)
    model = PTModel().cuda()
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model started.')

    # Testing parameters
    optimizer = torch.optim.Adam( model.parameters(), 0.0001 )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

    # Load data
    batch_size = args.bs
    # test_loader = getTestingData(batch_size=batch_size)
    _, test_loader = getTrainingTestingData(batch_size=batch_size)

    # Logging
    prefix = 'TEST_MobileNetV3_large'
    writer = SummaryWriter(comment= prefix, flush_secs=30)

    # Loss
    # Creates a criterion that measures the mean absolute error (MAE) between each element in the input x and target y.
    l1_criterion = nn.L1Loss()
       
    epoch_interation = checkpoint['epoch']
    loss = checkpoint['loss']
    print("Model with: {} epoch".format(epoch_interation))
    print("loss:",loss.item())

    # Start testing...
    batch_time = AverageMeter() # medidores

    size_loader = len(test_loader)

    # Switch to eval mode
    model.eval()
    # MSE_criterion = nn.MSELoss()
    # MSE = []

    end = time.time()

    eval_measures = torch.zeros(10).cuda()

    for loader_pos, sample_batched in enumerate(test_loader):
        with torch.no_grad():    
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # print("image.shape",image.shape) # [2,3,480,640] [numero do batch,canais,largura,altura]
            # print("depth.shape",depth.shape) # [2,1,240,320]

            # Normalize depth
            gt_depth = DepthNorm( depth )

            # Predict
            pred_depth = model(image)
            # gt_depth_cuda = gt_depth
            gt_depth = gt_depth.cpu().numpy().squeeze() #shape (5, 240, 320)
            # pred_depth_cuda = pred_depth
            pred_depth = pred_depth.cpu().numpy().squeeze() #shape (5, 240, 320)


        # vtransforms.ToPILImage()(image[0,:]).show()
        # vtransforms.ToPILImage()(depth[0,:]).show()
        # # vtransforms.ToPILImage()(gt_depth_cuda[0,:]).show()
        # vtransforms.ToPILImage()(gt_depth[0,:]).show() # pq ta saindo td branco? pode ser o dataset... e ele deixa o resultado estranho na img
        # # vtransforms.ToPILImage()(pred_depth_cuda[0,:]).show()
        # vtransforms.ToPILImage()(pred_depth[0,:]).show()
        # # colorizeCPU(gt_depth[0,:]).show()
        # # colorizeCPU(pred_depth[0,:]).show()


        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        # print("valid_mask.shape:",valid_mask.shape)

        # gt_height, gt_width = gt_depth.shape[1],gt_depth.shape[2]
        eval_mask = np.zeros(valid_mask.shape)
        # print("eval_mask.shape:",eval_mask.shape)

        # eval_mask[:,int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
        eval_mask[:,int(45/2):int(471/2), int(41/2):int(601/2)] = 1

        valid_mask = np.logical_and(valid_mask, eval_mask)
        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        # from PIL import Image
        # # Image.fromarray(eval_mask[0,:,:]).show()
        # Image.fromarray(valid_mask[0,:,:]).show()
        # test = pred_depth[valid_mask]
        # Image.fromarray(test[0,:,:]).show()
        # Image.fromarray(gt_depth[valid_mask][0,:,:]).show()



        # measures = compute_errors(gt_depth, pred_depth)
        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1
        
        #como foi usado no trabalho do cara, da uma estudada:
        # def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False):
        #     N = len(rgb)

        #     bs = batch_size

        #     predictions = []
        #     testSetDepths = []
            
        #     for i in range(N//bs):    
        #         x = rgb[(i)*bs:(i+1)*bs,:,:,:]
                
        #         # Compute results
        #         true_y = depth[(i)*bs:(i+1)*bs,:,:]
        #         pred_y = scale_up(2, predict(model, x/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0
                
        #         # Test time augmentation: mirror image estimate
        #         pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        #         # Crop based on Eigen et al. crop
        #         true_y = true_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        #         pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        #         pred_y_flip = pred_y_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
                
        #         # Compute errors per image in batch
        #         for j in range(len(true_y)):
        #             predictions.append(   (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))   )
        #             testSetDepths.append(   true_y[j]   )

        #     predictions = np.stack(predictions, axis=0) # predictions é uma lista, e o np.stack junta tudo numa nova linha
        #     testSetDepths = np.stack(testSetDepths, axis=0)

        #     e = compute_errors(predictions, testSetDepths)


        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time.val*(size_loader - loader_pos))))
    
        # Log progress
        if loader_pos % 5 == 0:
            # Print to console
            print('Step: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
            'ETA {eta}'
            .format(loader_pos, size_loader, batch_time=batch_time, eta=eta))


    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)))
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'))
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))



if __name__ == '__main__':
    main()
