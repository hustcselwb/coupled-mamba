import torch
from train_test import train
import argparse
import numpy as np

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
    parser.add_argument('--jacobian_weight', default=20, type=float, help='Jacobian loss weight')
    parser.add_argument('--num_layers', default=1, type=int, help='Number of layer steps')
    parser.add_argument('--f_thres', default=55, type=int, help='Threshold for equilibrium solver')
    parser.add_argument('--b_thres', default=56, type=int, help='Threshold for gradient solver')
    parser.add_argument('--stop_mode', default='abs', choices=['abs', 'rel'], help='stop mode for solver')
    parser.add_argument('--cosine_scheduler', action='store_true', help='Use cosine scheduler for learning rate decay')
    parser.add_argument('--use_default_fuse', action='store_true', help='Use default concatenation for feature fusion')
    parser.add_argument('--no_print', action='store_true', help='Do not print results')
    parser.add_argument('--solver', default='anderson', choices=['anderson', 'broyden'], help='Fixed point solver')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    data_folder = '/root/mamba-main/BRCA/BRCA'
    testonly = args.mode == 'test'
    modelpath = './savemodel/'
    accs = []
    mf1s = []
    wf1s = []
    traces = []
    if testonly:
        trace = train(data_folder, modelpath, testonly)
    else:
        for _ in range(1):
            args.mode = 'train'
            _, acc, wf1, mf1 = train(data_folder, modelpath, testonly)
            accs.append(acc)
            wf1s.append(wf1)
            mf1s.append(mf1)
            print(acc, wf1, mf1)
            args.mode = 'test'
            trace = train(data_folder, modelpath, testonly)
    #         print(len(trace))
            traces.append(trace)
    #     np.save(f'trace_deq.npy', traces)
        np.save(f'trace_fuse_only.npy', traces)  # change this name accordingly for convergence plot
        print("Accuracy", accs)
        print("Weighted F1", wf1s)
        print("Macro F1", mf1s)