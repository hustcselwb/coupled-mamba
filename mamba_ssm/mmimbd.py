def train(
        model, train_dataloader, valid_dataloader, total_epochs, is_packed=False,
        early_stop=False, task="multilabel", optimtype=torch.optim.Adam, lr=0.001, weight_decay=0.0,
        objective=nn.BCEWithLogitsLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None, input_to_float=True, clip_val=8,
        track_complexity=True, use_bert=False,  gradient_accumulation_steps=1, freeze_txt_epoch=None, freeze_img_epoch=None, lower_lr_for_fusion=False,return_epoch_results=True):
    """
    Handle running a simple supervised training loop.
    
    :param encoders: list of modules, unimodal encoders for each input modality in the order of the modality input data.
    :param fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
    :param head: classification or prediction head, takes in output of fusion module and outputs the classification or prediction results that will be sent to the objective function for loss calculation
    :param total_epochs: maximum number of epochs to train
    :param additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
    :param is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
    :param early_stop: whether to stop early if valid performance does not improve over 7 epochs
    :param task: type of task, currently support "classification","regression","multilabel"
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
    :param auprc: whether to compute auprc score or not
    :param save: the name of the saved file for the model with current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
    :param input_to_float: whether to convert input to float type or not
    :param clip_val: grad clipping limit
    :param track_complexity: whether to track training complexity or not
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model().to(device)


    def _trainprocess():
        op = optimtype([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        # total_steps = int(len(train_dataloader) / train_dataloader.batch_size / gradient_accumulation_steps) * total_epochs
        # op = optimtype([p for p in model.parameters() if p.requires_grad] +
        #                additional_params, lr=lr, warmup=0.1, t_total=total_steps)
        if lower_lr_for_fusion:
            op = optimtype([
                    {'params': model.encoders.parameters()},
                    {'params': model.head.parameters()},
                    {'params': model.fuse.parameters(), 'lr': lr*0.1}
                ], lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
#         scheduler = torch.optim.lr_scheduler.StepLR(op, step_size=10, gamma=0.9)
        bestvalloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            total_jac_loss = 0.0
            totals = 0
            model.train()

            for step, j in enumerate(train_dataloader):

                model.train()
#                     try:
                j_new = torch.unsqueeze(j, dim=1)
                out = model([_processinput(i).to(device)
                                for i in j[:-1]])

                jac_loss = torch.tensor([0.])
                if isinstance(out, tuple):
                    jac_loss = out[1]
                    out = out[0]
                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model
                loss = deal_with_objective(
                    objective, out, j[-1], objective_args_dict)
                loss += 0.1 * jac_loss.mean()

                totalloss += loss * len(j[-1])
                totals += len(j[-1])

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                try:
                    loss.backward()
                except:
                    print('\nSingularity detected during backward, proceeding to next batch.')
                    continue


                if (step + 1) % gradient_accumulation_steps == 0:
                    op.step()
                    op.zero_grad()
            print('')
            print("Epoch "+str(epoch)+" train loss: "+str(totalloss/totals))
#             scheduler.step()
            validstarttime = time.time()
            if validtime:
                print("train total: "+str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    if is_packed:
                        model.eval()
                        out = model([[_processinput(i).to(device)
                                    for i in j[0]], j[1]])
                    elif use_bert:
                        model.eval()
                        try:
                            out = model([j[0].to(device), j[1].to(device)] + [_processinput(i).to(device) for i in j[2:-1]])
                        except:
                            print('\nSingularity detected during validation')
                            continue
                    else:
                        model.eval()
                        try:
                            out = model([_processinput(i).to(device)
                                        for i in j[:-1]])
                        except:
                            print('\nSingularity detected during validation')
                            continue
                    
                    jac_loss = torch.tensor([0.])
                    if isinstance(out, tuple):
                        jac_loss = out[1]
                        out = out[0]
                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False
                    loss = deal_with_objective(
                        objective, out, j[-1], objective_args_dict)
                    loss += 0.1 * jac_loss.mean()
                    totalloss += loss*len(j[-1])
                    
                    if task == "classification":
                        pred.append(torch.argmax(out, 1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out).round())
                    true.append(j[-1])
                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss/totals
            print('')
            if task == "classification":
                acc = accuracy(true, pred)
                if scheduler is not None:
                    scheduler.step(acc)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " acc: "+str(acc))
                if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                if scheduler is not None:
                    scheduler.step(f1_micro)
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss) +
                      " f1_micro: "+str(f1_micro)+" f1_macro: "+str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch "+str(epoch)+" valid loss: "+str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    torch.save(model, save)
                else:
                    patience += 1
            if early_stop and patience > 5:
                break
            if auprc:
                print("AUPRC: "+str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  "+str(validendtime-validstarttime))
                print("Valid total: "+str(totals))
    if track_complexity:
        all_in_one_train(_trainprocess, [model]+additional_optimizing_modules)
    else:
        _trainprocess()