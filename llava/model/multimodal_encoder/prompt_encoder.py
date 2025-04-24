import torch
import torch.nn as nn


class InstPromptEncoder(nn.Module):
    
    def __init__(self, input_dim=32, hidden_size=4096):
        super(InstPromptEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.InstPromptProjector = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, inst_feature):
        # (n, 32)
        return self.InstPromptProjector(inst_feature)
