import torch
import torch.nn as nn

class DFNet(nn.Module):
    def __init__(self, input_size, hid_layer, weight_norm=True):
        super().__init__()
        output_size = 1
        dims = [input_size] + [d_hidden for d_hidden in hid_layer] + [output_size]

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.actv = nn.ReLU()
        self.out_actv = nn.ReLU()


    def forward(self, p):
        x = p.reshape(len(p), -1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x =  self.actv(x)
        x =  self.out_actv(x)
        return x

class BoneMLP(nn.Module):
    def __init__(self, bone_dim, bone_feature_dim, parent=-1):
        super().__init__()
        if parent ==-1:
            in_features = bone_dim
        else:
            in_features = bone_dim + bone_feature_dim
        n_features = bone_dim + bone_feature_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, bone_feature_dim),
            nn.ReLU()
        )

    def forward(self, bone_feat):
        return self.net(bone_feat)

class StructureEncoder(nn.Module):
    def __init__(self, local_feature_size=6,):
        super().__init__()

        self.bone_dim = 2
        self.input_dim = self.bone_dim  
        self.parent_mapping = [-1,0,0,0,0,1,5,2,7,3,9,4,11]

        self.num_joints = len(self.parent_mapping)
        self.out_dim = self.num_joints * local_feature_size

        self.net = nn.ModuleList([ BoneMLP(self.input_dim, local_feature_size, self.parent_mapping[i]) for i in range(self.num_joints) ])

    def get_out_dim(self):
        return self.out_dim

    def forward(self, x):
        features = [None] * self.num_joints
        for i, mlp in enumerate(self.net):
            parent = self.parent_mapping[i]
            if parent == -1:
                features[i] = mlp(x[:, i, :])
            else:
                inp = torch.cat((x[:, i, :], features[parent]), dim=-1)
                features[i] = mlp(inp)
        features = torch.cat(features, dim=-1) 
        return features

class PoseNDF(nn.Module):
    def __init__(self, feat_dim=6, hid_layer=[512, 512, 512, 512, 512], weight_norm=True):
        super().__init__()
        self.enc = StructureEncoder(feat_dim)
        self.dfnet = DFNet(feat_dim*13, hid_layer, weight_norm)       
       
    def train(self, mode=True):
        super().train(mode)

    def forward(self, x):
        B = x.shape[0]
        x = self.enc(x)
        dist_pred = self.dfnet(x)
        return dist_pred