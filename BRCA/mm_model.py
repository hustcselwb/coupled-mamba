""" 
    DEQ fusion module and the MM-Dynamics model
    Modified based on the DEQ repo (https://github.com/locuslab/deq)
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from solver import anderson, broyden, weight_tie
from jacobian import jac_loss_estimate, power_method
import sys
sys.path.append('/root/anaconda3/envs/retnet_2/lib/python3.10/site-packages/mamba_ssm')
#import pdb
from  mamba_ssm.modules.mamba_simple import MutiMamba,MutiMambabrca
# from mamba_ssm.modules.mamba_simple import MutiMambaimdb

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

def list2vec(z1_list):
    """Convert list of tensors to a vector"""
    bsz = z1_list[0].size(0)
    return torch.cat([elem.reshape(bsz, -1) for elem in z1_list], dim=1)

def vec2list(z1, cutoffs):
    """Convert a vector back to a list, via the cutoffs specified"""
    bsz = z1.shape[0]
    z1_list = []
    start_idx, end_idx = 0, cutoffs[0]
    for i in range(len(cutoffs)):
        z1_list.append(z1[:, start_idx:end_idx].view(bsz, cutoffs[i]))
        if i < len(cutoffs)-1:
            start_idx = end_idx
            end_idx += cutoffs[i + 1]
    return z1_list


class IdentityBlock(nn.Module):
    def __init__(self, out_dim, deq_expand=2, num_groups=2, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(IdentityBlock, self).__init__()
        
    def forward(self, x, injection_feature):
        return injection_feature

    
class SimpleResidualBlock(nn.Module):
    def __init__(self, out_dim, deq_expand=2, num_groups=2, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(SimpleResidualBlock, self).__init__()

        self.out_dim = out_dim

        self.conv1 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.conv2 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.gn1 = nn.GroupNorm(4, self.out_dim, affine=True)
        self.gn2 = nn.GroupNorm(4, self.out_dim, affine=True)
        self.gn3 = nn.GroupNorm(4, self.out_dim, affine=True)
        
    def forward(self, x, injection_feature):
        residual = x

        out = F.relu(self.gn1(self.conv1(x)))
        out = self.conv2(out) + injection_feature
        out = self.gn2(out)

        out += residual
        out = self.gn3(F.relu(out))
        
        return out


class DEQFusionBlock(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.2, wnorm=False):
        """
        Purified-then-combined fusion block.
        """
        super(DEQFusionBlock, self).__init__()

        self.out_dim = num_out_dims[-1]
        
        self.gate = nn.Linear(num_out_dims[0], self.out_dim)
        self.fuse = nn.Linear(self.out_dim, self.out_dim)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
            
        self.gn3 = nn.GroupNorm(4, self.out_dim, affine=True)
            
    def forward(self, x, injection_features, residual_feature):
        extracted_feats = []
        for i, inj_feat in enumerate(injection_features):
            # if not self.training:
            #     print(i, self.gate(inj_feat + x).abs().mean().item(), self.gate(inj_feat + x).abs().min().item(), self.gate(inj_feat + x).abs().max().item())
            extracted_feats.append(torch.mul(x, self.dropout1(self.gate(inj_feat + x))))
        
        out = self.dropout2(self.fuse(torch.stack(extracted_feats, dim=0).sum(dim=0))) + residual_feature
        out = self.gn3(F.relu(out))
        
        return out
    
class DEQIdentityFusionModule(nn.Module):
    def __init__(self, num_out_dims):
        super(DEQIdentityFusionModule, self).__init__()

        self.num_branches = len(num_out_dims)
        self.block = SimpleResidualBlock
        self.fusion_block = DEQFusionBlock
        self.branches = self._make_branches(self.num_branches, num_out_dims)

    def _make_one_branch(self, branch_index, num_out_dims):
        out_dim = num_out_dims[branch_index]
        return self.block(out_dim, deq_expand=1, num_groups=1, dropout=0)
    
    def _make_fusion_branch(self, branch_index, num_out_dims):
        #out_dim = num_out_dims[branch_index]
        return self.fusion_block(num_out_dims, deq_expand=2, dropout=0)

    def _make_branches(self, num_branch, num_out_dims):
        branch_layers = [self._make_one_branch(i, num_out_dims) for i in range(num_branch - 1)]
        branch_layers.append(self._make_fusion_branch(num_branch - 1, num_out_dims))
        return nn.ModuleList(branch_layers)

    def forward(self, x, injection):
        x_block_out = []
        for i in range(self.num_branches - 1):
            x_block_out.append(injection[i])
        x_block_out.append(self.branches[self.num_branches - 1](x[self.num_branches - 1], x_block_out, injection[-1]))
        
        return x_block_out

class DEQFusionLayer(nn.Module):
    def __init__(self, num_out_dims):
        super(DEQFusionLayer, self).__init__()

        self.num_branches = len(num_out_dims)
        self.block = SimpleResidualBlock
        self.fusion_block = DEQFusionBlock
        self.branches = self._make_branches(self.num_branches, num_out_dims)

    def _make_one_branch(self, branch_index, num_out_dims):
        out_dim = num_out_dims[branch_index]
        return self.block(out_dim, deq_expand=1, num_groups=1, dropout=0)
    
    def _make_fusion_branch(self, branch_index, num_out_dims):
        #out_dim = num_out_dims[branch_index]
        return self.fusion_block(num_out_dims, deq_expand=2, dropout=0)

    def _make_branches(self, num_branch, num_out_dims):
        branch_layers = [self._make_one_branch(i, num_out_dims) for i in range(num_branch - 1)]
        branch_layers.append(self._make_fusion_branch(num_branch - 1, num_out_dims))
        return nn.ModuleList(branch_layers)

    def forward(self, x, injection):
        x_block_out = []
        for i in range(self.num_branches - 1):
            out = self.branches[i](x[i], injection[i])
            x_block_out.append(out)
        x_block_out.append(self.branches[self.num_branches - 1](x[self.num_branches - 1], x_block_out, injection[-1]))
        
        return x_block_out
    

class DEQFusion(nn.Module):
    def __init__(self, channel_dim, num_modals, f_thres=55, b_thres=56, stop_mode="abs", deq=True, num_layers=1, solver='anderson'):
        super(DEQFusion, self).__init__()
        self.f_thres = f_thres
        self.b_thres = b_thres
        self.stop_mode = stop_mode
        self.func_ = DEQFusionLayer([channel_dim for _ in range(num_modals + 1)])
#         self.func_ = DEQIdentityFusionModule([channel_dim for _ in range(num_modals + 1)])
        self.f_solver = anderson if solver == 'anderson' else broyden
        self.b_solver = anderson if solver == 'anderson' else broyden
        self.deq = deq
        self.num_layers = num_layers
        
    def featureFusion(self, features, fusion_feature, compute_jac_loss=True):
        batch_size = features[0].shape[0]
        feature_dim = features[0].shape[1]

        x_list = [f for f in features] + [fusion_feature]
        out_dim_list = [f.shape[1] for f in features] + [fusion_feature.shape[1]]
        z_list = [torch.zeros(batch_size, dim_size).cuda() for dim_size in out_dim_list]
        cutoffs = [elem.size(1) for elem in z_list]
        z1 = list2vec(z_list)
        #self.func_._reset(z_list)
        func = lambda z: list2vec(self.func_(vec2list(z, cutoffs), x_list))

        deq = self.deq
        jac_loss = torch.tensor(0.0).to(fusion_feature)

        if not deq:
            result = {'rel_trace': 0}
            for layer_ind in range(self.num_layers):
                z1 = func(z1)
            new_z1 = z1
#             if self.training:
#                 if compute_jac_loss:
#                     z2 = z1.clone().detach().requires_grad_()
#                     new_z2 = func(z2)
#                     jac_loss = jac_loss_estimate(new_z2, z2)
        else:
            with torch.no_grad():
                result = self.f_solver(func, z1, threshold=self.f_thres, stop_mode=self.stop_mode)
                z1 = result['result']
#                 print(z1.min(),z1.max())
                print(result['nstep'], end=" ")
            new_z1 = z1
            if self.training:
                new_z1 = func(z1.requires_grad_())
                if compute_jac_loss:
                    jac_loss = jac_loss_estimate(new_z1, z1)

                def backward_hook(grad):
                    if self.hook is not None:
                        # To avoid infinite loop
                        self.hook.remove()
                        torch.cuda.synchronize()
                    new_grad = self.b_solver(lambda y: autograd.grad(new_z1, z1, y, retain_graph=True)[0] + grad, \
                                                torch.zeros_like(grad), threshold=self.b_thres)['result']
                    return new_grad
                self.hook = new_z1.register_hook(backward_hook)
        net = vec2list(new_z1, cutoffs)

        return net[-1], jac_loss.view(1,-1), result
    
    def forward(self, features, fusion_feature):
        """ 
            features: List[Tensor], features from different modalities
            fusion_feature: Tensor, initial fused feature computed from `features`
        """
        fused_feat, jacobian_loss, trace = self.featureFusion(features, fusion_feature)
        
        return fused_feat, jacobian_loss, trace


def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class MMDynamic(nn.Module):
    def __init__(self, 
                 in_dim, 
                 hidden_dim, 
                 num_class, 
                 dropout, 
                 jacobian_weight=100, 
                 f_thres=105, 
                 b_thres=106, 
                 stop_mode='abs', 
                 use_deq=False,
                 deq=False,
                 num_layers=2,
                 solver='anderson'):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout
        
        self.use_deq = use_deq
        self.deq = deq
        self.num_layers = num_layers

        self.FeatureInforEncoder = nn.ModuleList([LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim)-1):
            if self.use_deq:
                self.MMClasifier.append(LinearLayer(hidden_dim[0], hidden_dim[layer]))
            else:
                self.MMClasifier.append(LinearLayer(self.views*hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            if self.use_deq:
                self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
            else:
                self.MMClasifier.append(LinearLayer(self.views*hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)
        
        if self.use_deq:
            self.jacobian_weight = jacobian_weight
            self.deq_fusion = DEQFusion(hidden_dim[-1], self.views, f_thres, b_thres, stop_mode, self.deq, self.num_layers, solver)
#             self.branches = nn.Sequential()
#             for _ in range(self.views):
# #                 self.branches.append(nn.Sequential(nn.Linear(hidden_dim[-1],hidden_dim[-1]),))
#                 self.branches.append(DEQ(hidden_dim[-1], f_thres,b_thres,stop_mode,self.deq))
#                 self.branches.append(nn.Linear(hidden_dim[-1],hidden_dim[-1]))
#                 self.branches.append(nn.Identity())
#             self._deq = DEQ(hidden_dim[-1], f_thres,b_thres,stop_mode,self.deq)
#             self.deq_fusion = Fusion(hidden_dim[-1], self.views)

    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](data_list[view]))
            feature[view] = data_list[view] * FeatureInfo[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]
            # print(feature[view].shape)
        
        if self.use_deq:
            # Our DEQ Fusion
            fusion_feature = torch.stack([f for f in feature.values()], dim=0).sum(dim=0)
            MMfeature, jacobian_loss, trace = self.deq_fusion([f for f in feature.values()], fusion_feature)
            
#             MMfeature = torch.stack(MMfeature, dim=0).sum(dim=0)

#             MMfeature, jac, trace = zip(*[self.branches[i](f) for i,f in enumerate(feature.values())])
#             MMfeature = torch.stack(list(MMfeature), dim=0).sum(dim=0)
#             jacobian_loss = torch.stack(list(jac), dim=0).mean()
#             MMfeature = torch.stack([self.branches[i](f) for i,f in enumerate(feature.values())],dim=0).sum(dim=0)
#             print(MMfeature.shape)
#             fusion_feature = torch.stack([f for i,f in enumerate(feature.values())],dim=0).sum(dim=0)
#             MMfeature, jacobian_loss = self._deq(fusion_feature)
#             jacobian_loss = torch.tensor(0.)
        else:
            for i in range(self.views):
                feature[i]=torch.unsqueeze(feature[i],dim=1).cuda()
            multimamba=MutiMambabrca(512,1).cuda()
            MMfeature=multimamba(feature[0],feature[1],feature[2])
            MMfeature=torch.squeeze(MMfeature,dim=1)

            # MMfeature = torch.cat([i for i in feature.values()], dim=1)

            # import ipdb
            # ipdb.set_trace()
            # print(MMfeature.shape)
            trace = {}
        loss_dict = dict()
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit, trace
        MMLoss = torch.mean(criterion(MMlogit, label))
        loss_dict['MMLoss'] = MMLoss.item()
        for view in range(self.views):
            MMLoss = 1.5 * MMLoss + torch.mean(FeatureInfo[view])
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(F.mse_loss(TCPConfidence[view].view(-1), p_target)+criterion(TCPLogit[view], label))
            loss_dict[f'confidence_loss_view{view}'] = confidence_loss.item()
            MMLoss = MMLoss+confidence_loss
        if self.use_deq:
            loss_dict['jacobian'] = jacobian_loss.mean().item()
            MMLoss += self.jacobian_weight * jacobian_loss.mean()
        loss_dict['total'] = MMLoss.item()
        
        return MMLoss, MMlogit, loss_dict, trace
    
    def infer(self, data_list):
        MMlogit, trace = self.forward(data_list, infer=True)
        return MMlogit, trace

            


