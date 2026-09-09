
from Labconfig import *
from model.utils import *
from model.dataloader import *
from model.net_module import *


class FNO(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_shape_trunk, input_shape_branch1, input_shape_branch2 = args.input_shape_trunk, args.input_shape_branch1, args.input_shape_branch2
        self.device = args.device
        encoded_dim = 4
        self.feat_dim = 256   # 必须能被 num_heads 整除（72%4=0）
        
        self.pos_encoder = PositionalEncoding(encoded_dim)

        self.b2 = args.batch_size

        self.pos_scale = nn.Parameter(torch.tensor(0.1))  # 位置编码缩放因子
        self.fencoder = FourierFeatureEncoder(2,256)

        self.data_norm_coe = 1.
        self.pde_norm_coe = 1.
        self.pde_real_k = 0.
        self.pde_imag_k = 0.

        self.FNO = nn.Sequential(
            FNO2d(input_shape_branch1[1] + input_shape_branch2[1], 2, 16, 16, 128),
            # nn.Dropout2d(0.1),
        )
        

    def forward(self, vel, UU0):
        # 基础维度获取
        nn = vel.shape[-1]
        b1 = vel.shape[0]        # Batch size (B_v)
        
        # 1. 坐标预处理与 Trunk 提取 (Query)

        fin = torch.cat([vel, UU0], dim=1)
        outputs = self.FNO(fin)
        
        return outputs


