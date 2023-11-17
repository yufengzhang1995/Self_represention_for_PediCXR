import re
import torch
import torch.nn as nn

class MLPHead(nn.Module):
    """
    MLP Head as the base type for encoder and predictor
    """
    def __init__(self, input_dim, embedding_dim, hidden_size=512, layer=2):
        super().__init__()
        if layer == 0:
            self.net = nn.Sequential(nn.Linear(input_dim, embedding_dim))
        elif layer == 1:
            self.net = nn.Sequential(nn.Linear(input_dim, embedding_dim),
                                     nn.BatchNorm1d(embedding_dim),
                                     nn.ReLU(inplace=True))
        elif layer == 2:
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_size, embedding_dim))
        elif layer == 3:
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_size, hidden_size),
                                 nn.BatchNorm1d(hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_size, embedding_dim))
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

    def forward(self, x):
        return self.net(x)
        

class TRAM_Model(nn.Module):
    """
    Transfer and Marginalize model class. Following the paper:
        https://arxiv.org/abs/2202.09244
    Inputs:
        encoder: the base encoder that extracts the features from the input
        add_input_dim: the dimension of the additional input
        num_categories: the number of categories for final classification
        base_predictor_input_dim: the input dimension of the features extracted by the encoder
        base_predictor_hidden_dim: the hidden dimension of the base predictor
            if layer<=1, this value is not used
        base_predictor_layer: the number of layers of the base predictor, could be 0, 1, 2, 3
        add_predictor_hidden_dim: the hidden dimension of the additional predictor
        add_predictor_layer: the number of layers of the additional predictor
    """
    def __init__(self, encoder,
                add_input_dim: int,
                num_categories: int=2,
                feature_function: str=None,
                base_predictor_input_dim: int=2048,
                base_predictor_hidden_dim: int=512,
                base_predictor_layer: int=0,
                add_encoder_embed_dim: int=64,
                add_predictor_hidden_dim: int=128,
                add_predictor_layer: int=2):

        super().__init__()

        self.encoder = encoder

        self.base_predictor = MLPHead(base_predictor_input_dim, 
                                  num_categories, 
                                  hidden_size=base_predictor_hidden_dim, 
                                  layer=base_predictor_layer)
        
        self.add_encoder = MLPHead(add_input_dim, add_encoder_embed_dim, layer=1)
        
        self.add_predictor = MLPHead(base_predictor_input_dim + add_encoder_embed_dim, 
                               num_categories, 
                               hidden_size=add_predictor_hidden_dim,
                               layer=add_predictor_layer)

        self.feature_function = feature_function

    def forward(self, x, x_add=None, train=False, get_embedding=False):
        # forward on the base encoder and predictor
        if self.feature_function is not None:
            encoder = self.encoder.module if hasattr(self.encoder, "module") else self.encoder
            encoder_output = getattr(encoder, self.feature_function)(x)
        else:
            encoder_output = self.encoder(x)

        # flatten the output of the encoder, shape (batch_size, base_predictor_input_dim)
        encoder_embedding = encoder_output.view(encoder_output.size(0), -1)
        # forward on the base predictor, shape (batch_size, num_categories)
        base_prediction = self.base_predictor(encoder_embedding.detach())
        add_prediction = None
        if train: # forward on the additional encoder and predictor
            # forward on the additional encoder, x_add is of shape (batch_size, add_input_dim)
            add_encoder_output = self.add_encoder(x_add)
            # concatenate the additional input to the base encoder output
            add_encoder_input = torch.cat((
                encoder_embedding, add_encoder_output), dim=1) 
            add_prediction = self.add_predictor(add_encoder_input)
            return base_prediction, add_prediction
        if get_embedding:
            return encoder_embedding, base_prediction     
        return base_prediction
    


    
class Linear_Eval(nn.Module):
    """
    Linear eval model class to serve as classifiers in down-stream task.
    Inputs:
        encoder: the base encoder that extracts the features from the input
        input_dim: the dimension of the input
        num_categories: the number of categories for final classification

    """
    def __init__(self, encoder, 
                 input_dim: int,
                 feature_function: str=None,
                 num_categories: int=2,
                 hidden_dim: int=512,
                 predictor_layer: int=0):
        
        super().__init__()

        self.encoder = encoder
        self.input_dim = input_dim
        self.feature_function = feature_function
        self.linear_probing = MLPHead(input_dim, num_categories, 
                                 hidden_size=hidden_dim, layer=predictor_layer)

    def forward(self, x, freeze_encoder=False, get_embedding=False):
        encoder_func = None
        if self.feature_function is not None:
            encoder = self.encoder.module if hasattr(self.encoder, "module") else self.encoder
            encoder_func = getattr(encoder, self.feature_function)

        if freeze_encoder:
            with torch.no_grad():
                encoder_output = self.encoder(x) if encoder_func is None else encoder_func(x)
        else:
            encoder_output = self.encoder(x) if encoder_func is None else encoder_func(x)
            
        encoder_embedding = encoder_output.view(encoder_output.size(0), -1)
        if get_embedding:
            return encoder_embedding, self.linear_probing(encoder_embedding)
        return self.linear_probing(encoder_embedding)


class Vits_Linear_Eval(nn.Module):
    """
    Linear eval model class to serve as classifiers in down-stream task.
    Inputs:
        encoder: the base encoder that extracts the features from the input
        input_dim: the dimension of the input
        num_categories: the number of categories for final classification

    """
    def __init__(self, encoder, 
                 input_dim: int,
                 feature_function: str=None,
                 num_categories: int=2,
                 hidden_dim: int=512,
                 predictor_layer: int=0):
        super().__init__()

        self.encoder = encoder
        self.input_dim = input_dim
        self.feature_function = feature_function
        self.linear_probing = MLPHead(input_dim, num_categories, 
                                    hidden_size=hidden_dim, layer=predictor_layer)

    def forward(self, x, get_embedding=False, 
                register_hook=False, return_attentions=False):
        
        encoder_output = self.encoder(x, register_hook=register_hook, 
                                    return_attentions=return_attentions)
        if return_attentions:
            (encoder_embedding, attentions) = encoder_output
        else:
            encoder_embedding = encoder_output
        
        output = self.linear_probing(encoder_embedding)

        if return_attentions:
            if get_embedding:
                return encoder_embedding, output, attentions
            return output, attentions
        else:
            if get_embedding:
                return encoder_embedding, output
            return output