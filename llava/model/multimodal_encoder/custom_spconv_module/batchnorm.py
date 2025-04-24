import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(BatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        dtype = torch.float32

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(num_features, dtype=dtype))
        else:
            self.weight = None
            self.bias = None
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype=dtype))
            self.register_buffer('running_var', torch.ones(num_features, dtype=dtype))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, input):
        compute_dtype = input.dtype
        out = F.batch_norm(
            input.float(),  
            self.running_mean.float() if self.running_mean is not None else None, 
            self.running_var.float() if self.running_var is not None else None, 
            self.weight.float() if self.weight is not None else None,  
            self.bias.float() if self.bias is not None else None, 
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )
        
        return out.to(compute_dtype)
