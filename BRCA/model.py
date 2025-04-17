""" Components of the original MM-Dynamics model
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from solver import anderson, broyden
from jacobian import jac_loss_estimate, power_method


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


class SmallBlock_V3(nn.Module):
    def __init__(self, out_dim, deq_expand=2, num_groups=2, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(SmallBlock, self).__init__()

        self.out_dim = out_dim

        self.conv1 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.conv2 = torch.nn.Linear(self.out_dim, self.out_dim)
        #self.gn2 = torch.nn.BatchNorm1d(self.out_dim)#nn.GroupNorm(1, inplanes, affine=True)
        #self.relu2 = nn.GeLU(inplace=True)
        self.relu2 = nn.GELU()
        #self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, injection_feature):
        out = self.conv1(x) + injection_feature
        out = self.relu2(out)
        #out = self.relu2(self.conv2(self.gn2(out)))
        #out = self.relu2(self.conv2(out))
        return out

class SmallBlock_V2(nn.Module):
    def __init__(self, out_dim, deq_expand=2, num_groups=2, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(SmallBlock_V2, self).__init__()

        self.out_dim = out_dim

        self.conv1 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.conv2 = torch.nn.Linear(self.out_dim, self.out_dim)
        #self.gn2 = torch.nn.BatchNorm1d(self.out_dim)#nn.GroupNorm(1, inplanes, affine=True)
        #self.relu2 = nn.GeLU(inplace=True)
        self.relu2 = nn.GELU()
        #self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, injection_feature):
        out = self.conv1(x) + self.conv2(injection_feature)
        #out = self.relu2(out)
        #out = self.relu2(self.conv2(self.gn2(out)))
        out = self.relu2(out)
        return out
    
class SmallBlock(nn.Module):
    def __init__(self, out_dim, deq_expand=2, num_groups=2, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(SmallBlock, self).__init__()

        self.out_dim = out_dim

        self.conv1 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.conv2 = torch.nn.Linear(self.out_dim, self.out_dim)
        #self.gn2 = torch.nn.BatchNorm1d(self.out_dim)#nn.GroupNorm(1, inplanes, affine=True)
        #self.relu2 = nn.GeLU(inplace=True)
        self.relu2 = nn.GELU()
        #self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, injection_feature):
        out = self.conv1(x) + injection_feature
        #out = self.relu2(out)
        #out = self.relu2(self.conv2(self.gn2(out)))
        out = self.relu2(self.conv2(out))
        return out

class EQSimpleGELUFusionBlock(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(EQSimpleGELUFusionBlock, self).__init__()

        self.branch1_dim = num_out_dims[0]
        self.branch2_dim = num_out_dims[1]
        self.out_dim = num_out_dims[2]
        
        #self.conv1 = torch.nn.Linear(self.branch1_dim, self.out_dim)
        #self.conv2 = torch.nn.Linear(self.branch2_dim, self.out_dim)
        #self.conv_inject = torch.nn.Linear(self.branch1_dim + self.branch2_dim, self.out_dim)
        self.conv_inject = torch.nn.Linear(self.out_dim, self.out_dim)
        #self.gn3 = torch.nn.BatchNorm1d(self.out_dim) #nn.GroupNorm(1, inplanes, affine=True)
        self.relu3 = nn.GELU()
        self.conv3 = torch.nn.Linear(self.out_dim, self.out_dim)
            
    def forward(self, x, injection_feature_branch1, injection_feature_branch2, residual_feature_branch1, residual_feature_branch2):
        
        #extracted_fea1 = torch.mul(x, self.conv1(injection_feature_branch1))
        #extracted_fea2 = torch.mul(x, self.conv2(injection_feature_branch2))
        
        #injection_feature = torch.cat((residual_feature_branch1, residual_feature_branch2), dim=1)
        #injection_feature = residual_feature_branch1 + residual_feature_branch2
        injection_feature = self.conv_inject(x)
        out = injection_feature_branch1 + injection_feature_branch2 + injection_feature
        #out = self.relu3(self.conv3(self.gn3(out)))
        out = self.relu3(self.conv3(out))
        return out
    
class EQSimpleFusionBlock(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(EQSimpleFusionBlock, self).__init__()

        self.branch1_dim = num_out_dims[0]
        self.branch2_dim = num_out_dims[1]
        self.out_dim = num_out_dims[2]
        
        #self.conv1 = torch.nn.Linear(self.branch1_dim, self.out_dim)
        #self.conv2 = torch.nn.Linear(self.branch2_dim, self.out_dim)
        #self.conv_inject = torch.nn.Linear(self.branch1_dim + self.branch2_dim, self.out_dim)
        self.conv_inject = torch.nn.Linear(self.out_dim, self.out_dim)
        #self.gn3 = torch.nn.BatchNorm1d(self.out_dim) #nn.GroupNorm(1, inplanes, affine=True)
        #self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Linear(self.out_dim, self.out_dim)
            
    def forward(self, x, injection_feature_branch1, injection_feature_branch2, residual_feature_branch1, residual_feature_branch2):
        
        #extracted_fea1 = torch.mul(x, self.conv1(injection_feature_branch1))
        #extracted_fea2 = torch.mul(x, self.conv2(injection_feature_branch2))
        
        #injection_feature = torch.cat((residual_feature_branch1, residual_feature_branch2), dim=1)
        #injection_feature = residual_feature_branch1 + residual_feature_branch2
        injection_feature = self.conv_inject(x)
        out = injection_feature_branch1 + injection_feature_branch2 + injection_feature
        #out = self.relu3(self.conv3(self.gn3(out)))
        out = self.conv3(out)
        return out

class EQFusionV4Block(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(EQFusionV4Block, self).__init__()

        self.branch1_dim = num_out_dims[0]
        self.branch2_dim = num_out_dims[1]
        self.out_dim = num_out_dims[2]
        
        #self.conv1 = torch.nn.Linear(self.branch1_dim, self.out_dim)
        #self.conv2 = torch.nn.Linear(self.branch2_dim, self.out_dim)
        #self.conv_inject = torch.nn.Linear(self.branch1_dim + self.branch2_dim, self.out_dim)
        self.conv_att1 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.conv_att2 = torch.nn.Linear(self.out_dim, self.out_dim)
        self.act_att1 = nn.Tanh()
        self.act_att2 = nn.Tanh()
        self.gn3 = torch.nn.BatchNorm1d(self.out_dim) #nn.GroupNorm(1, inplanes, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Linear(self.out_dim, self.out_dim)
            
    def forward(self, x, injection_feature_branch1, injection_feature_branch2, residual_feature_branch1, residual_feature_branch2):
        
        #extracted_fea1 = torch.mul(x, self.conv1(injection_feature_branch1))
        #extracted_fea2 = torch.mul(x, self.conv2(injection_feature_branch2))
        
        #injection_feature = torch.cat((residual_feature_branch1, residual_feature_branch2), dim=1)
        #injection_feature = residual_feature_branch1 + residual_feature_branch2
        out = torch.mul(x, self.act_att1(self.conv_att1(injection_feature_branch1))) + torch.mul(x, self.act_att2(self.conv_att2(injection_feature_branch2)))
        #self.conv_inject = torch.nn.Linear(self.out_dim, self.out_dim)
        #injection_feature = self.conv_(x)
        #out = injection_feature_branch1 + injection_feature_branch2 + injection_feature
        #out = self.relu3(self.conv3(self.gn3(out)))
        #out = self.conv3(out + residual_feature_branch1 + residual_feature_branch2)
        out = self.conv3(self.gn3(out + residual_feature_branch1 + residual_feature_branch2))
        return out
    
class EQFusionV3Block(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(EQFusionV3Block, self).__init__()

        self.branch1_dim = num_out_dims[0]
        self.branch2_dim = num_out_dims[1]
        self.out_dim = num_out_dims[2]
        
        #self.conv1 = torch.nn.Linear(self.branch1_dim, self.out_dim)
        #self.conv2 = torch.nn.Linear(self.branch2_dim, self.out_dim)
        #self.conv_inject = torch.nn.Linear(self.branch1_dim + self.branch2_dim, self.out_dim)
        self.conv_inject = torch.nn.Linear(self.out_dim, self.out_dim)
        self.gn3 = torch.nn.BatchNorm1d(self.out_dim) #nn.GroupNorm(1, inplanes, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Linear(self.out_dim, self.out_dim)
            
    def forward(self, x, injection_feature_branch1, injection_feature_branch2, residual_feature_branch1, residual_feature_branch2):
        
        #extracted_fea1 = torch.mul(x, self.conv1(injection_feature_branch1))
        #extracted_fea2 = torch.mul(x, self.conv2(injection_feature_branch2))
        
        #injection_feature = torch.cat((residual_feature_branch1, residual_feature_branch2), dim=1)
        #injection_feature = residual_feature_branch1 + residual_feature_branch2
        injection_feature = self.conv_inject(x)
        out = injection_feature_branch1 + injection_feature_branch2 + injection_feature
        #out = self.relu3(self.conv3(self.gn3(out)))
        out = self.conv3(out) + residual_feature_branch1 + residual_feature_branch2
        return out

class EQFusionV21Block(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(EQFusionV21Block, self).__init__()

        self.branch1_dim = num_out_dims[0]
        self.branch2_dim = num_out_dims[1]
        self.out_dim = num_out_dims[2]
        
        self.conv1 = torch.nn.Linear(self.branch1_dim, self.out_dim)
        self.conv2 = torch.nn.Linear(self.branch2_dim, self.out_dim)
        #self.conv_inject = torch.nn.Linear(self.branch1_dim + self.branch2_dim, self.out_dim)
        #self.conv_inject = torch.nn.Linear(self.out_dim, self.out_dim)
        #self.gn3 = torch.nn.BatchNorm1d(self.out_dim) #nn.GroupNorm(1, inplanes, affine=True)
        #self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Linear(self.out_dim, self.out_dim)
            
    def forward(self, x, injection_feature_branch1, injection_feature_branch2, residual_feature_branch1, residual_feature_branch2):
        
        extracted_fea1 = torch.mul(x, self.conv1(injection_feature_branch1 + x))
        extracted_fea2 = torch.mul(x, self.conv2(injection_feature_branch2 + x))
        
        #injection_feature = torch.cat((residual_feature_branch1, residual_feature_branch2), dim=1)
        injection_feature = residual_feature_branch1 + residual_feature_branch2
        out = extracted_fea1 + extracted_fea2 + injection_feature
        #out = self.relu3(self.conv3(self.gn3(out)))
        out = self.conv3(out)
        return out
    
class EQFusionV2Block(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(EQFusionV2Block, self).__init__()

        self.branch1_dim = num_out_dims[0]
        self.branch2_dim = num_out_dims[1]
        self.out_dim = num_out_dims[2]
        
        self.conv1 = torch.nn.Linear(self.branch1_dim, self.out_dim)
        self.conv2 = torch.nn.Linear(self.branch2_dim, self.out_dim)
        #self.conv_inject = torch.nn.Linear(self.branch1_dim + self.branch2_dim, self.out_dim)
        #self.conv_inject = torch.nn.Linear(self.out_dim, self.out_dim)
        #self.gn3 = torch.nn.BatchNorm1d(self.out_dim) #nn.GroupNorm(1, inplanes, affine=True)
        #self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Linear(self.out_dim, self.out_dim)
            
    def forward(self, x, injection_feature_branch1, injection_feature_branch2, residual_feature_branch1, residual_feature_branch2):
        
        extracted_fea1 = torch.mul(x, self.conv1(injection_feature_branch1))
        extracted_fea2 = torch.mul(x, self.conv2(injection_feature_branch2))
        
        #injection_feature = torch.cat((residual_feature_branch1, residual_feature_branch2), dim=1)
        injection_feature = residual_feature_branch1 + residual_feature_branch2
        out = extracted_fea1 + extracted_fea2 + injection_feature
        #out = self.relu3(self.conv3(self.gn3(out)))
        out = self.conv3(out)
        return out
    
class EQFusionBlock(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(EQFusionBlock, self).__init__()

        self.branch1_dim = num_out_dims[0]
        self.branch2_dim = num_out_dims[1]
        self.out_dim = num_out_dims[2]
        
        self.conv1 = torch.nn.Linear(self.branch1_dim, self.out_dim)
        self.conv2 = torch.nn.Linear(self.branch2_dim, self.out_dim)
        #self.conv_inject = torch.nn.Linear(self.branch1_dim + self.branch2_dim, self.out_dim)
        self.conv_inject = torch.nn.Linear(self.out_dim, self.out_dim)
        #self.gn3 = torch.nn.BatchNorm1d(self.out_dim) #nn.GroupNorm(1, inplanes, affine=True)
        #self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Linear(self.out_dim, self.out_dim)
            
    def forward(self, x, injection_feature_branch1, injection_feature_branch2, residual_feature_branch1, residual_feature_branch2):
        
        extracted_fea1 = torch.mul(x, self.conv1(injection_feature_branch1))
        extracted_fea2 = torch.mul(x, self.conv2(injection_feature_branch2))
        
        #injection_feature = torch.cat((residual_feature_branch1, residual_feature_branch2), dim=1)
        injection_feature = residual_feature_branch1 + residual_feature_branch2
        injection_feature = self.conv_inject(injection_feature)
        out = extracted_fea1 + extracted_fea2 + injection_feature
        #out = self.relu3(self.conv3(self.gn3(out)))
        out = self.conv3(out)
        return out
    
class FusionBlock(nn.Module):
    def __init__(self, num_out_dims, deq_expand=2, num_groups=1, dropout=0.0, wnorm=False):
        """
        A canonical residual block with two 1x1 convolutions and an intermediate ReLU.
        """
        super(FusionBlock, self).__init__()

        self.branch1_dim = num_out_dims[0]
        self.branch2_dim = num_out_dims[1]
        self.out_dim = num_out_dims[2]
        
        self.conv1 = torch.nn.Conv1d(self.branch1_dim, self.out_dim, 1)
        self.conv2 = torch.nn.Conv1d(self.branch2_dim, self.out_dim, 1)
        self.conv_inject = torch.nn.Conv1d(self.out_dim, self.out_dim, 1)
        self.gn3 = torch.nn.BatchNorm1d(self.out_dim) #nn.GroupNorm(1, inplanes, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv1d(self.out_dim, self.out_dim, 1)
            
    def forward(self, x, injection_feature, residual_feature_branch1, residual_feature_branch2):
        
        extracted_fea1 = torch.mul(x, self.conv1(residual_feature_branch1))
        extracted_fea2 = torch.mul(x, self.conv2(residual_feature_branch2))
        
        injection_feature = self.conv_inject(injection_feature)
        out = extracted_fea1 + extracted_fea2 + injection_feature
        out = self.relu3(self.conv3(self.gn3(out)))
        return out

class DEQEQFusionModule(nn.Module):
    def __init__(self, num_out_dims):
        super(DEQEQFusionModule, self).__init__()

        self.num_branches = 3
        #self.block = SmallBlock
        #self.block = SmallBlock
        self.block = SmallBlock_V2
        #self.fusion_block = EQSimpleFusionBlock
        #self.fusion_block = EQFusionV2Block
        #self.fusion_block = EQFusionV3Block
        #self.fusion_block = EQFusionV4Block
        self.fusion_block = EQFusionV21Block
        #self.fusion_block = EQSimpleGELUFusionBlock
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
        inject_features = [injection[0], injection[1], injection[2]]
        #xyz_features = injection[3]

        x_block_out = []
        #x_block_feature = []
        for i in range(self.num_branches - 1):
            out = self.branches[i](x[i], inject_features[i])
            x_block_out.append(out)
        inject_fusion_feature = x_block_out[0] + x_block_out[1]
        x_block_out.append(self.branches[self.num_branches - 1](x[self.num_branches - 1], x_block_out[0], x_block_out[1], injection[0], injection[1]))
        #return self.branches[self.num_branches - 1](x[self.num_branches - 1], x_block_out[0], x_block_out[1], injection[0], injection[1])
        return x_block_out

class DEQFusionInfoEQV4DropZModule(nn.Module):

    def __init__(self, num_out_dims):
        super(DEQFusionInfoEQV4DropZModule, self).__init__()
        self.num_out_dims = num_out_dims
        self.linear_z = nn.Linear(num_out_dims[2], num_out_dims[2], bias=False)
        self.linear_infoext = nn.Linear(num_out_dims[2], num_out_dims[2], bias=False)
        self.linear_hv = nn.Linear(num_out_dims[0], num_out_dims[2])
        self.linear_hq = nn.Linear(num_out_dims[1], num_out_dims[2])

        self.linear_hv_infoext = nn.Linear(num_out_dims[0], num_out_dims[2])
        self.linear_hq_infoext = nn.Linear(num_out_dims[1], num_out_dims[2])

        self.drop_z = VariationalFCDropout(dropout=0.1)
        self.act = nn.GELU()

    def reset(self, bsz):
        self.drop_z.reset_mask(bsz, self.num_out_dims[2])

    def forward(self, z, x_v, x_q):
        z = self.drop_z(z)
        x_hv = x_v#self.drop_inputv(x_v)
        #x_v_trans = getattr(F, "tanh")(self.linear_hv_infoext(x_hv))
        x_v_trans = self.linear_hv_infoext(x_hv)
        x_hv = self.linear_hv(x_hv)

        x_hq = x_q#self.drop_inputq(x_q)
        #x_q_trans = getattr(F, "tanh")(self.linear_hq_infoext(x_hq))
        x_q_trans = self.linear_hq_infoext(x_hq)
        x_hq = self.linear_hq(x_hq)

        infoEQ_item = torch.mul(x_v_trans, z) + torch.mul(x_q_trans, z)
        x_qv_vec = x_hq + x_hv

        new_z = infoEQ_item + self.linear_infoext(x_qv_vec)
        
        return new_z

    
class DEQFusionInfoEQV3DropZModule(nn.Module):

    def __init__(self, num_out_dims):
        super(DEQFusionInfoEQV3DropZModule, self).__init__()
        self.num_out_dims = num_out_dims
        self.linear_z = nn.Linear(num_out_dims[2], num_out_dims[2], bias=False)
        self.linear_infoext = nn.Linear(num_out_dims[2], num_out_dims[2], bias=False)
        self.linear_hv = nn.Linear(num_out_dims[0], num_out_dims[2])
        self.linear_hq = nn.Linear(num_out_dims[1], num_out_dims[2])

        self.linear_hv_infoext = nn.Linear(num_out_dims[0], num_out_dims[2])
        self.linear_hq_infoext = nn.Linear(num_out_dims[1], num_out_dims[2])

        self.drop_z = VariationalFCDropout(dropout=0.1)
        self.act = nn.GELU()

    def reset(self, bsz):
        self.drop_z.reset_mask(bsz, self.num_out_dims[2])

    def forward(self, z, x_v, x_q):
        z = self.drop_z(z)
        x_hv = x_v#self.drop_inputv(x_v)
        #x_v_trans = getattr(F, "tanh")(self.linear_hv_infoext(x_hv))
        x_v_trans = self.linear_hv_infoext(x_hv)
        x_hv = self.linear_hv(x_hv)

        x_hq = x_q#self.drop_inputq(x_q)
        #x_q_trans = getattr(F, "tanh")(self.linear_hq_infoext(x_hq))
        x_q_trans = self.linear_hq_infoext(x_hq)
        x_hq = self.linear_hq(x_hq)

        infoEQ_item = torch.mul(x_v_trans, z) + torch.mul(x_q_trans, z)
        #x_qv_vec = torch.mul(x_hq, x_hv)
        x_qv_vec = x_hq + x_hv

        new_z = infoEQ_item + self.linear_infoext(x_qv_vec)
        
        return new_z
    

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
    def __init__(self, in_dim, hidden_dim, num_class, dropout, use_deq=True):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout
        
        self.use_deq = use_deq

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
            self.func_ = DEQEQFusionModule([hidden_dim[-1] for _ in range(3)])
            #self.func_ = DEQFusionInfoEQV4DropZModule([hidden_dim[-1] for _ in range(3)])
            self.f_thres = 105
            self.b_thres = 106
            self.stop_mode = "abs"
            self.f_solver = anderson
            self.b_solver = anderson

    def featureFusion(self, epoch_idx, audio_feature, text_feature, fusion_feature, compute_jac_loss=True):
        batch_size = audio_feature.shape[0]
        feature_dim = audio_feature.shape[1]

        x_list = [audio_feature, text_feature, fusion_feature]
        out_dim_list = [audio_feature.shape[1], text_feature.shape[1], fusion_feature.shape[1]]
        z_list = [torch.zeros(batch_size, dim_size).cuda() for dim_size in out_dim_list]
        cutoffs = [elem.size(1) for elem in z_list]
        z1 = list2vec(z_list)
        #self.func_._reset(z_list)
        func = lambda z: list2vec(self.func_(vec2list(z, cutoffs), x_list))

        deq = True#epoch_idx > self.pretraining
        #print(epoch_idx, end=" ")
        jac_loss = torch.tensor(0.0).to(fusion_feature)

        if not deq:
            for layer_ind in range(self.num_layers): 
                z1 = func(z1)
            new_z1 = z1
            if self.training:
                if compute_jac_loss:
                    z2 = z1.clone().detach().requires_grad_()
                    new_z2 = func(z2)
                    jac_loss = jac_loss_estimate(new_z2, z2)
        else:
            with torch.no_grad():
                result = self.f_solver(func, z1, threshold=self.f_thres, stop_mode=self.stop_mode)
                z1 = result['result']
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

        return net[-1], jac_loss.view(1,-1)

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
            if not self.use_deq:
                feature[view] = feature[view] * TCPConfidence[view]
        
        if self.use_deq:
            fusion_feature = feature[0] * TCPConfidence[0] + feature[1] * TCPConfidence[1]
            MMfeature, jacobian_loss = self.featureFusion(-1, feature[0], feature[1], fusion_feature)
        else:
            MMfeature = torch.cat([i for i in feature.values()], dim=1)
        loss_dict = dict()
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))
        loss_dict['MMLoss'] = MMLoss.item()
        for view in range(self.views):
            MMLoss = MMLoss+torch.mean(FeatureInfo[view])
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(F.mse_loss(TCPConfidence[view].view(-1), p_target)+criterion(TCPLogit[view], label))
            loss_dict[f'confidence_loss_view{view}'] = confidence_loss.item()
            MMLoss = MMLoss+confidence_loss
        if self.use_deq:
            loss_dict['jocabian'] = jacobian_loss.mean().item()
            MMLoss += 100 * jacobian_loss.mean()
        
        return MMLoss, MMlogit, loss_dict
    
    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

            


