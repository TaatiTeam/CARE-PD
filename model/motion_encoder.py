import torch.nn as nn
import torch
import numpy as np


class ClassifierHead(nn.Module):
    def __init__(self, params, num_classes=3):
        super(ClassifierHead, self).__init__()
        self.params = params
        input_dim = self._get_input_dim()
        if self.params['medication']:
            input_dim += 1
        if len(self.params['metadata']) > 0:
            input_dim += len(self.params['metadata'])
        self.dims = [input_dim, *self.params['classifier_hidden_dims'], num_classes]

        self.fc_layers = self._create_fc_layers()
        self.batch_norms = self._create_batch_norms()
        self.dropout = nn.Dropout(p=self.params['classifier_dropout'])
        self.activation = nn.ReLU()

    def _create_fc_layers(self):
        fc_layers = nn.ModuleList()
        mlp_size = len(self.dims)

        for i in range(mlp_size - 1):
            fc_layer = nn.Linear(in_features=self.dims[i],
                                 out_features=self.dims[i+1])
            fc_layers.append(fc_layer)
        
        return fc_layers
    
    def _create_batch_norms(self):
        batch_norms = nn.ModuleList()
        n_batchnorms = len(self.dims) - 2
        if n_batchnorms == 0:
            return batch_norms
        
        for i in range(n_batchnorms):
            batch_norm = nn.BatchNorm1d(self.dims[i+1], momentum=0.1)
            batch_norms.append(batch_norm)
        
        return batch_norms

    def _get_input_dim(self):
        backbone = self.params['backbone']
        if backbone == 'potr':
            if self.params['preclass_rem_T']:
                return self.params['model_dim']
            else:
                return self.params['model_dim'] * self.params['source_seq_len']
        elif backbone == "motionbert":
            if self.params['merge_joints']:
                return self.params['dim_rep']
            else:
                return self.params['dim_rep'] * self.params['num_joints']
        elif backbone == 'poseformerv2':
            return self.params['embed_dim_ratio'] * self.params['num_joints'] * 2
        elif backbone == "mixste":
            if self.params['merge_joints']:
                return self.params['embed_dim_ratio']
            else:
                return self.params['embed_dim_ratio'] * self.params['num_joints']
        elif backbone == "motionagformer":
            if self.params['merge_joints']:
                return self.params['dim_rep']
            else:
                return self.params['dim_rep'] * self.params['num_joints']
        elif backbone == "momask":
            if self.params['avg_over_time']:
                return self.params['dim_rep']
            else:
                return self.params['dim_rep'] * int(np.floor(self.params['source_seq_len']/self.params['downsample_ratio']))
        elif backbone == "motionclip":
            return self.params['dim_rep']


    def forward(self, feat, valid_frame_mask=None, forward_classifier=True):
        feat = self.dropout(feat)
        if self.params['backbone'] == 'motionbert':
            feat = self._forward_motionbert(feat, valid_frame_mask)
        elif self.params['backbone'] == 'potr':
            feat = self._forward_poseforemer(feat, valid_frame_mask)
        elif self.params['backbone'] == 'poseformerv2':
            feat = self._forward_poseformerv2(feat, valid_frame_mask)
        elif self.params['backbone'] == "mixste":
            feat = self._forward_mixste(feat, valid_frame_mask)
        elif self.params['backbone'] == "motionagformer":
            feat = self._forward_motionagformer(feat, valid_frame_mask)
        elif self.params['backbone'] == "momask":
            feat = self._forward_momask(feat, valid_frame_mask)
        elif self.params['backbone'] == "motionclip":
            feat = self._forward_motionclip(feat, valid_frame_mask)
        if forward_classifier:
            return self._forward_fc_layers(feat)
        else:
            return feat

    def _forward_fc_layers(self, feat):
        mlp_size = len(self.dims)
        for i in range(mlp_size - 2):
            fc_layer = self.fc_layers[i]
            batch_norm = self.batch_norms[i]

            feat = self.activation(batch_norm(fc_layer(feat)))

        last_fc_layer = self.fc_layers[-1]
        feat = last_fc_layer(feat)
        return feat

    def _forward_motionclip(self, feat, valid_frame_mask):
        """
        x: Tensor with shape (batch_size, dim_rep=512)
        """
        B, C = feat.shape
        return feat

    def _forward_momask(self, feat, valid_frame_mask):
        """
        x: Tensor with shape (batch_size, dim_rep=512, floor(num_frames / 4))
        """
        B, C, T = feat.shape
        if self.params['avg_over_time']:
            if valid_frame_mask is not None:
                # Reshape to group every 4 time steps together
                valid_frame_mask_reshaped = valid_frame_mask.view(B, feat.shape[-1], self.params['downsample_ratio'])
                # If any value in the group is zero, the whole group should be zero
                reduced_mask = valid_frame_mask_reshaped.min(dim=2)[0]  
                mask = reduced_mask.unsqueeze(1).float() # Expand mask to shape (B, 1, T)
                feat = (feat * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
            else:
                feat = feat.mean(dim=-1)  # (B, C, T) -> (B, C)
        else:
            feat = feat.reshape(B, -1) # (B, C, T) -> (B, C * T)
        return feat
    
    def _forward_motionagformer(self, feat, valid_frame_mask):
        B, T, J, C = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, T, J, C) -> (B, J, C, T)
        if valid_frame_mask is not None:
            mask = valid_frame_mask.unsqueeze(1).unsqueeze(1).float() # Expand mask to shape (B, 1, 1, T) for broadcasting
            feat = (feat * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
        else:
            # Regular mean over time
            feat = feat.mean(dim=-1)   # (B, J, C, T) -> (B, J, C)
        if self.params['merge_joints']:
            feat = feat.mean(dim=-2)  # (B, J, C) -> (B, C)
        else:
            feat = feat.reshape(B, -1)  # (B, J * C)
        return feat
    
    def _forward_mixste(self, feat, valid_frame_mask):
        """
        feat: Tensor with shape (batch_size, n_frames, n_joints, dim_representation)
        valid_frame_mask: Tensor (B, T) — 1 if valid, 0 if padded
        """
        B, T, J, C = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, T, J, C) -> (B, J, C, T)
        if valid_frame_mask is not None:
            mask = valid_frame_mask.unsqueeze(1).unsqueeze(1).float() # Expand mask to shape (B, 1, 1, T)
            feat = (feat * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
        else:
            feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        if self.params['merge_joints']:
            feat = feat.mean(dim=-2)  # (B, J, C) -> (B, C)
        else:
            feat = feat.reshape(B, -1)  # (B, J * C)
        return feat

    def _forward_poseformerv2(self, feat, valid_frame_mask):
        """
        feat: Tensor with shape (batch_size, 1, embed_dim_ratio * num_joints * 2)
        valid_frame_mask: Tensor (B, T) — 1 if valid, 0 if padded
        """
        B, _, C = feat.shape
        feat = feat.reshape(B, C)  # (B, 1, C) -> (B, C)
        return feat

    def _forward_motionbert(self, feat, valid_frame_mask):
        """
        feat: Tensor with shape (batch_size, n_frames, n_joints, dim_representation)
        valid_frame_mask: Tensor (B, T) — 1 if valid, 0 if padded
        """
        assert valid_frame_mask is not None, "valid_frame_mask should NOT be None for motionbert backbone"
        B, T, J, C = feat.shape
        feat = feat.permute(0, 2, 3, 1)  # (B, T, J, C) -> (B, J, C, T)
        if valid_frame_mask is not None:
            mask = valid_frame_mask.unsqueeze(1).unsqueeze(1).float() # Expand mask to shape (B, 1, 1, T) for broadcasting
            feat = (feat * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
        else:
            # Regular mean over time
            feat = feat.mean(dim=-1)   # (B, J, C, T) -> (B, J, C)

        if self.params['merge_joints']:
            feat = feat.mean(dim=-2)  # (B, J, C) -> (B, C)
        else:
            feat = feat.reshape(B, -1)  # (B, J * C)
        return feat

    def _forward_poseforemer(self, feat, valid_frame_mask):
        """
        feat: Tensor with shape (batch_size, n_frames, dim_representation)
        valid_frame_mask: Tensor (B, T) — 1 if valid, 0 if padded
        """
        T, B, C = feat.shape
        if self.params['preclass_rem_T']:
            # Reshape the tensor to (B, 1, C, T)   J=1
            feat = feat.permute(1, 2, 0).unsqueeze(1)
            if valid_frame_mask is not None:
                mask = valid_frame_mask.unsqueeze(1).unsqueeze(1).float() # Expand mask to shape (B, 1, 1, T) 
                feat = (feat * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-6)
            else:
                feat = feat.mean(dim=-1)  # (B, J, C, T) -> (B, J, C)
        else:
            feat = feat.permute(1, 0, 2)  # (B, T, C)

        feat = feat.reshape(B, -1)  # (B, J * C) or (B, T * C)
        return feat


class MotionEncoder(nn.Module):
    def __init__(self, backbone, params, num_classes=4, train_mode='end2end'):
        super(MotionEncoder, self).__init__()
        assert train_mode in ['end2end', 'classifier_only'], "train_mode should be either end2end or classifier_only." \
                                                             f" Found {train_mode}"
        self.backbone = backbone
        if train_mode == 'classifier_only':
            self.freeze_backbone()
        self.head = ClassifierHead(params, num_classes=num_classes)
        self.num_classes = num_classes
        self.medprob = params['medication']
        self.metadata = params['metadata']

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("[INFO - MotionEncoder] Backbone parameters are frozen")

    def forward(self, x, metadata, med=None, valid_mask=None):
        """
        x: Tensor with shape (batch_size, n_frames, n_joints, C=3) or (batch_size, n_frames, 263) for humanml3d format
        """
        feat = self.backbone(x)
        if self.medprob and med is not None:
            med = med.to(feat.device)
            med = med.view(*[-1] + [1] * (feat.dim() - 1))
            s = list(feat.shape)
            s[-1] = 1  # Set the last dimension to 1
            med = med.expand(*s)
            feat = torch.cat((feat, med), dim=-1)
        if len(self.metadata) > 0:
            metadata = metadata.view(metadata.shape[0], *([1] * (feat.dim() - 2)), metadata.shape[-1])
            metadata = metadata.expand(*feat.shape[:-1], metadata.shape[-1])
            feat = torch.cat((feat, metadata), dim=-1)
            
        if valid_mask is not None:
            out = self.head(feat, valid_mask.bool())
        else:
            out = self.head(feat)

        return out


def _test_classifier_head():
    import torch

    params = {
        "backbone": "motionbert",
        "dim_rep": 512,
        "classifier_hidden_dims": [],
        'classifier_dropout': 0.5
    }
    head = ClassifierHead(params, num_classes=3, num_joints=17)

    B, T, J, C = 4, 243, 17, 512
    feat = torch.randn(B, T, J, C)
    out = head(feat)
    assert out.shape == (4, 3)

if __name__ == "__main__":
    _test_classifier_head()