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
from model_mobileV3_Unet_interpolado import PTModel
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
    parser.add_argument('--bs', '--batch-size', default=32, type=int, help='batch size') # 16
    parser.add_argument('--pt', '--path', default="./checkpoints/withEigen_interpoled_new.pth", type=str, help='path to the model') # mob3L-ep3-loss0.040_10kDS.pth
    
    parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3) # 1e-3
    parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80) # 80
    
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
    # l1_criterion = nn.L1Loss()
       
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
            gt_depth_cuda = gt_depth
            gt_depth = gt_depth.cpu().numpy().squeeze() #shape (5, 240, 320)
            pred_depth_cuda = pred_depth
            pred_depth = pred_depth.cpu().numpy().squeeze() #shape (5, 240, 320)

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        # print("mask.shape:",mask.shape)

        gt_height, gt_width = gt_depth.shape[1],gt_depth.shape[2]
        crop_mask = np.zeros(mask.shape)
        # print("crop_mask.shape:",crop_mask.shape)

        # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results #https://github.com/mrharicot/monodepth/blob/master/utils/evaluate_kitti.py
        # For fair comparison with state-of-the-art single view depth prediction, we evaluate our results on the same cropped region of interest as [8]. Since the supervised methods are trained using the ground-truth depth that ranges between 1 and 50 meters whereas we can predict larger depths, we clamp the predicted depth values for our method between 1 and 50 for evaluation. i.e. setting the depths bigger than 50 metres to 50.
        crop = np.array([int(0.09375*gt_height),int(0.98125*gt_height), int(0.0640625*gt_width),int(0.9390625*gt_width)]).astype(np.int32)
        crop_mask[:,crop[0]:crop[1],crop[2]:crop[3]] = 1
        

        mask = np.logical_and(mask, crop_mask)
        measures = compute_errors(gt_depth[mask], pred_depth[mask])
        # measures = compute_errors(gt_depth, pred_depth) # para remover a mascara, precisa comentar as alteraçoes do pred_depth tbm

        if True: # enable view mode
            from PIL import Image
            colorizedGT = colorizeCPU(gt_depth[0,:])
            colorizedPred = colorizeCPU(pred_depth[0,:])

            vtransforms.ToPILImage()(image[0,:]).show(title="RGB")
            vtransforms.ToPILImage()(depth[0,:]).show(title="depth")

            # vtransforms.ToPILImage()(gt_depth[0,:]).show(title="gt") #printa tudo escuro (é o gray)       
            # vtransforms.ToPILImage()(pred_depth[0,:]).show(title="pred gt")

            vtransforms.ToPILImage()(gt_depth_cuda[0,:]).show(title="gt cuda")
            vtransforms.ToPILImage()(pred_depth_cuda[0,:]).show(title="pred gt cuda")

            colorizedGT.show(title="gt colorized")
            colorizedPred.show(title="pred gt colorized")

            # Image.fromarray(mask[0,:,:]).show(title="mask") #printa a mascara
            Image.composite(Image.fromarray(crop_mask[0,:,:]),colorizedGT,Image.fromarray(np.invert(mask[0,:,:]))).show(title="validation gt")
            Image.composite(Image.fromarray(crop_mask[0,:,:]),colorizedPred,Image.fromarray(np.invert(mask[0,:,:]))).show(title="validation pred gt")
    

        # measures = compute_errors(gt_depth, pred_depth)
        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1
        

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
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'silog', 'abs_rel', 'log10', 'rms','sq_rel', 'log_rms', 'd1', 'd2','d3'))
    #             0.123    0.053    0.465                       0.846 0.974 0.994
    for i in range(8):
        print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.4f}'.format(eval_measures_cpu[8]))



if __name__ == '__main__':
    main()
