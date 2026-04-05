import torch
import torch.nn as nn

class Coefficient(nn.Module):
    
    def __init__(self, 
                 hidden_dim=32,
                 use_4d_features=True,  
                 dropout_rate=0.1,
                 opacity_only=False):    
        
        super().__init__()
        self.use_4d_features = use_4d_features
        self.opacity_only = opacity_only
        input_dim = self._calculate_input_dim()
        
        self.net = nn.Sequential( 
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def _calculate_input_dim(self):
        if self.opacity_only:
            return 1  
        input_dim = 7  # opacity + positions + scales
        if self.use_4d_features: 
            input_dim += 2   
        return input_dim
    
    def forward(self, opacity, positions=None, scales=None):

        opacity = opacity.mul(2).sub(1)

        if self.opacity_only:
            return self.net(opacity)

        pos = positions - positions.mean(0, keepdim=True)
        pos = pos / (positions.std(0, keepdim=True, unbiased=False) + 1e-6)

        sca = torch.log(scales + 1e-6)
        sca = sca - sca.mean(0, keepdim=True)
        sca = sca / (sca.std(0, keepdim=True, unbiased=False) + 1e-6)

        x = torch.cat((opacity, pos, sca), dim=1)

        return self.net(x)