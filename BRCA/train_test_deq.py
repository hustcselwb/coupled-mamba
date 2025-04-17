"""
    Training and testing of the model with DEQ fusion
"""
import os
from re import I
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from mm_model import MMDynamic
import utils

cuda = True if torch.cuda.is_available() else False

import matplotlib.pyplot as plt

def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1,1), 1)    
    return y_onehot

def prepare_trte_data(data_folder, use_view=['mrna', 'dna', 'mirna']):
    num_view = len(use_view)
    assert num_view > 0, 'no modality included!'
    view_ids = {'mrna': 1, 'dna': 2, 'mirna': 3}
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in [view_ids[view] for view in use_view]:
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i)+"_te.csv"), delimiter=','))
    
    eps = 1e-10
    X_train_min = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    X_train_max = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1]) for i in range(len(data_tr_list))]
    data_te_list = [data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1]) for i in range(len(data_tr_list))]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr+num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                       data_tensor_list[i][idx_dict["te"]].clone()),0))
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels


def train_epoch(data_list, label, model, optimizer, lr_schedule_values, epoch):
    # Using cosine scheduler
    if lr_schedule_values is not None: #or wd_schedule_values is not None and data_iter_step % update_freq == 0:
        for i, param_group in enumerate(optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[epoch]
#             if wd_schedule_values is not None and param_group["weight_decay"] > 0:
#                 param_group["weight_decay"] = wd_schedule_values[it]
    
    model.train()
    optimizer.zero_grad()
    loss, _, loss_dict, trace = model(data_list, label)
    print('\r'+str([f'{k}: {v:.5f}' for k, v in loss_dict.items()]), end='')
    loss = torch.mean(loss)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 30)
#     total_norm = 0
#     for p in model.parameters():
#         param_norm = p.grad.detach().data.norm(2)
#         total_norm += param_norm.item() ** 2
#     total_norm = total_norm ** 0.5
#     print(total_norm)
    optimizer.step()
#     return trace


def test_epoch(data_list, model):
    model.eval()
    with torch.no_grad():
        logit, trace = model.infer(data_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob, trace['rel_trace'], logit

def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint, strict=False)


def train(data_folder, modelpath, args):
    test_inverval = 50
    testonly = args.mode == 'test'
    if 'BRCA' in data_folder:
        hidden_dim = [500]#[500]
        num_epoch = args.train_epoch
        lr = args.learning_rate
        step_size = 500
        num_class = 5
    elif 'ROSMAP' in data_folder:
        hidden_dim = [300]
        num_epoch = 1000
        lr = 1e-4
        step_size = 500
        num_class = 2
    if not args.no_print:
        print(f'hidden dim: {hidden_dim}\tnum of epoch: {num_epoch}\tlearning rate: {lr}\tsolver: {args.solver}\tJacobian loss weight: {args.jacobian_weight}')

    use_view = []
    if args.use_mrna:
        use_view.append('mrna')
    if args.use_dna:
        use_view.append('dna')
    if args.use_mirna:
        use_view.append('mirna')
    if not args.no_print:
        print(f'Using modalities: {use_view}')
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder, use_view=use_view)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_tr_tensor = labels_tr_tensor.cuda()
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()
    
    labels_te_tensor = torch.LongTensor(labels_trte[trte_idx["te"]])
    onehot_labels_te_tensor = one_hot_tensor(labels_te_tensor, num_class)
    labels_te_tensor = labels_te_tensor.cuda()
    onehot_labels_te_tensor = onehot_labels_te_tensor.cuda()
    
    dim_list = [x.shape[1] for x in data_tr_list]
    model = MMDynamic(dim_list, 
                      hidden_dim, 
                      num_class, 
                      dropout=0.5, 
                      jacobian_weight=args.jacobian_weight, 
                      f_thres=args.f_thres, 
                      b_thres=args.b_thres, 
                      stop_mode=args.stop_mode,
                      use_deq=not args.use_default_fuse,
                      deq=not args.no_deq,
                      num_layers=args.num_layers,
                      solver=args.solver)
#     print("Model Architecture:", model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    
    # Instantiate cosine scheduler
    lr_schedule_values = utils.cosine_scheduler(
        lr, 1e-8, num_epoch+1, 1,
        warmup_epochs=50, warmup_steps=-1,
    ) if args.cosine_scheduler else None
    
    if testonly:
        load_checkpoint(model, os.path.join(modelpath, data_folder, f'{args.name}_best.pt'))
        te_prob, trace, _ = test_epoch(data_test_list, model)
        if not args.no_print:
            if num_class == 2:
                print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
            else:
                print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                print("Test F1 weighted: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                print("Test F1 macro: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
                
        return trace
    else:    
        if not args.no_print:
            print("\nTraining...")
        best_acc = 0
        best_wf1 = 0
        best_mf1 = 0
        for epoch in range(num_epoch+1):
            train_epoch(data_tr_list, labels_tr_tensor, model, optimizer, lr_schedule_values, epoch)
            if not args.cosine_scheduler:
                scheduler.step()
            if epoch % test_inverval == 0:
                te_prob, trace, logit = test_epoch(data_test_list, model)
                if not args.no_print:
                    print("\nTest: Epoch {:d}".format(epoch))
                    if num_class == 2:
                        print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                        print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                        print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:,1])))
                    else:
                        print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                        print("Test F1 weighted: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                        print("Test F1 macro: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
                acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                wf1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
                mf1 = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
                if (acc + wf1 + mf1) > (best_acc + best_wf1 + best_mf1):
                    best_acc = acc
                    best_wf1 = wf1
                    best_mf1 = mf1
                    save_checkpoint(model.state_dict(), os.path.join(modelpath, data_folder), filename=f'{args.name}_best.pt')
#         save_checkpoint(model.state_dict(), os.path.join(modelpath, data_folder), filename=f'{args.name}.pt')
        
        return torch.nn.CrossEntropyLoss(reduction='mean')(logit, labels_te_tensor).cpu().item(), best_acc, best_wf1, best_mf1
