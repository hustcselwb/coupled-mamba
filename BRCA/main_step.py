import torch
import numpy as np
from train_test_deq import train
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='DEQ Feature Fusion')
    parser.add_argument('-n', '--name', default='checkpoint', type=str, help='Name of the experiment')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], help='Train or test')
    parser.add_argument('-mrna', '--use_mrna', action='store_true', help='Use MRNA Modality')
    parser.add_argument('-dna', '--use_dna', action='store_true', help='Use DNA Modality')
    parser.add_argument('-mirna', '--use_mirna', action='store_true', help='Use MiRNA Modality')
    parser.add_argument('-lr', '--learning_rate', help='Set the learning rate', default=1e-4, type=float)
    parser.add_argument('-e', '--train_epoch', help='Set the training epoch', default=2500, type=int)
    parser.add_argument('--no_deq', action='store_true', help='Do not use DEQ for feature fusion')
    parser.add_argument('--jacobian_weight', default=100, type=float, help='Jacobian loss weight')
    parser.add_argument('--num_layers', default=1, type=int, help='Number of layer steps')
    parser.add_argument('--f_thres', default=55, type=int, help='Threshold for equilibrium solver')
    parser.add_argument('--b_thres', default=56, type=int, help='Threshold for gradient solver')
    parser.add_argument('--stop_mode', default='abs', choices=['abs', 'rel'], help='stop mode for solver')
    parser.add_argument('--cosine_scheduler', action='store_true', help='Use cosine scheduler for learning rate decay')
    parser.add_argument('--use_default_fuse', action='store_true', help='Use default concatenation for feature fusion')
    parser.add_argument('--no_print', action='store_true', help='Do not print results')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    data_folder = 'BRCA'
    testonly = False
    modelpath = './model/'
    loss_trace = []
    for i in range(1, 61):
#         torch.manual_seed(1)
        args.num_layers = i
        loss_curr_step = []
        for _ in range(10):
            l, _, _, _ = train(data_folder, modelpath, args)
            loss_curr_step.append(l)
            print(f"test loss at step {i}:", l)
        loss_trace.append(loss_curr_step)
        print(f"Avg test loss at step {i}:", np.nanmean(loss_curr_step))
    np.save('loss1.npy', loss_trace)