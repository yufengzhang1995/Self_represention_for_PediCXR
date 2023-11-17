import torch
import torch.nn as nn
import copy
from functools import wraps
import torch.nn.functional as F

class TRAM(nn.Module):
    """
    TRAM model class. The forward() method directly returns the tram loss.
    Inputs:
        pi_input_dim <python int>: dimension of the privileged 
            information input to the pi encoder
        pi_encode_embed_dim <python int>: dimension of the 
            embedding layer in the pi encoder
        base_input_dim <python int>: dimension of the base 
            input to the pi and base coencoder
        pi_encode_hid_dim <python int>: dimension of the 
            hidden layer in the pi encoder
        pi_encode_project_dim <python int>: dimension of 
            the projection layer in the pi encoder
    Not used:
        loss_factor <python float>: factor to scale the loss by
        loss_type <python str>: type of loss to use. Options are 'byol'
    """
    def __init__(self, 
                 pi_input_dim:int,
                 base_input_dim:int,
                 pi_encode_embed_dim:int=64,
                 pi_encode_hid_dim:int=1024,
                 pi_encode_project_dim:int=256,
                 projection_size:int=256,
                 projection_hidden_size:int=4096):
        
        super().__init__()
        self.pi_encode = nn.Sequential(nn.Linear(pi_input_dim, pi_encode_embed_dim),
                                 nn.BatchNorm1d(pi_encode_embed_dim),
                                 nn.ReLU(inplace=True))
        
        self.pi_co_encoder = MLP(dim=pi_encode_embed_dim+base_input_dim,
                                 projection_size=pi_encode_project_dim,
                                 hidden_size=pi_encode_hid_dim,)
        
        self.pi_projector = MLP(dim=pi_encode_project_dim,
                                projection_size=projection_size,
                                hidden_size=projection_hidden_size,)
    
        
    def forward(self, base_x_embed, pi_x):
        pi_embed = self.pi_encode(pi_x)
        pi_and_base_embed = torch.cat((pi_embed, base_x_embed), dim=1)
        pi_projection = self.pi_projector(self.pi_co_encoder(pi_and_base_embed))
        return pi_projection
