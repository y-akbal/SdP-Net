import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Callable
from torch.nn.attention import SDPBackend, sdpa_kernel

class ConvPatcher(nn.Module):
    def __init__(self, 
                 embedding_dim = 128, 
                 patch_size = 4):
        super().__init__()

        self.conv = nn.Conv2d(in_channels = 3, 
                              out_channels = embedding_dim,
                              kernel_size = patch_size,
                              stride = patch_size, 
                              bias = False
                              )
        self.patch_size = patch_size
    def forward(self, x:torch.Tensor)->torch.Tensor:
        ## B, 3, H, W --> B, out_channels, H//PATCH_SIZE, W//PATCH_SIZE ### 
        return self.conv(x) 
"""
conv_patcher(patch_size=16)(torch.randn(5, 3, 224,224)).mean()
"""

class ConvMixer(nn.Module):
    def __init__(self, 
                 embedding_dim:int = 768, 
                 kernel_size:int = 5, 
                 activation:Callable = nn.GELU()
                 ):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels = embedding_dim, 
                              out_channels = embedding_dim,
                              kernel_size = kernel_size,
                              groups = embedding_dim,
                              padding = "same",
                              bias = True,
                              )
        self.conv1d = nn.Conv2d(in_channels = embedding_dim,
                                out_channels = embedding_dim,
                                kernel_size =1,
                                bias = True,
                                )
        self.layer_norm_1 = nn.GroupNorm(1, embedding_dim)
        self.layer_norm_2 = nn.GroupNorm(1, embedding_dim)
        self.activation = activation

    def forward(self, x_:torch.Tensor):
        x = self.conv2d(x_)
        x = self.activation(x)
        x = self.layer_norm_1(x)+x_
        x = self.conv1d(x)
        x = self.activation(x)
        x = self.layer_norm_2(x)
        return x

"""
layer = conv_mixer().cuda()
x = torch.randn(100, 768, 14, 14).cuda()
layer(x)
"""

class EmbeddingLayer(nn.Module):
    ## We isolated this layer in the case that you want to 
    ## do something like enumerating the pixels...
    def __init__(self, 
                 embedding_dim: int = 768,
                 max_num_registers:int = 5,
                 max_image_size:list[int, int] = [14,14],
                 activation:Callable = None
                 ):
        super().__init__()
        ### -- ###
        self.max_num_registers = max_num_registers
        self.activation = activation if activation != None else lambda x: x
        ###
        self.register_embedding_layer = nn.Embedding(max_num_registers, embedding_dim)
        self.vertical_embedding_layer = nn.Embedding(max_image_size[0], embedding_dim)
        self.horizontal_embedding_layer = nn.Embedding(max_image_size[1], embedding_dim)
        ### --- ###
        ### --- ###
        ### --- ###
        ### ----###
        self.register_buffer(
            "register_embeddings",
            torch.tensor(
                [i for i in range(self.max_num_registers)],
                dtype=torch.int,
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "vertical_embedding",
            torch.tensor(
                [i for i in range(max_image_size[0])],
                dtype=torch.int,
                requires_grad=False,
            ),
        )

        self.register_buffer(
            "horizontal_embedding",
            torch.tensor(
                [i for i in range(max_image_size[1])],
                dtype=torch.int,
                requires_grad=False,
            ),
        )
    
    def forward(self, 
                x:torch.Tensor, 
                num_registers:int = 0)->tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        ## NO NEED TO COMPUTE THESE DUDES FOR EACH FORWARD PASS!!!!! Do we have kind a caching mechanism?
        register_embeddings = self.register_embedding_layer(self.register_embeddings[:num_registers+1])
        horizontal_embeddings = self.horizontal_embedding_layer(self.horizontal_embedding[:H]).transpose(-1, -2).unsqueeze(-1)
        vertical_embeddings = self.vertical_embedding_layer(self.vertical_embedding[:W]).transpose(-1,-2).unsqueeze(-2)
        ## We are adding embeddings both vertically and horizontally!!!
        
        x += horizontal_embeddings 
        x += vertical_embeddings
        
        #Expand the embeddings though not the best memory efficient way!!!
        expanded_register_embeddings = register_embeddings.expand(B, register_embeddings.shape[-2] ,C)

        return  self.activation(x), expanded_register_embeddings

"""
x = embedding_layer(max_image_size=[15,15])(torch.randn(2, 768, 15, 15), 0)
"""
class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        n_head: int = 8,
        activation_func: Callable = F.gelu,
        multiplication_factor: int = 4,
        ff_dropout: float = 0.2,
        att_dropout: float = 0.2,
        fast_att:bool = True,
        normalize_qv:bool = True
    ):
        super().__init__()
        assert embedding_dim % n_head == 0, "Number of embedding_dim must be divisible by n_head"
        
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.head_dim = embedding_dim // n_head
        self.att_dropout = att_dropout
        self.fast_att = fast_att
        self.normalize_qv = normalize_qv

        # Multi-head self-attention
        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        self.o_proj = nn.Linear(embedding_dim, embedding_dim, bias = False)
        
        # Feed-forward network
        self.ff_linear1 = nn.Linear(embedding_dim, multiplication_factor * embedding_dim, bias = True)
        self.ff_linear2 = nn.Linear(multiplication_factor * embedding_dim, embedding_dim, bias = True)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Activation and dropout
        self.activation = activation_func
        self.dropout = nn.Dropout(ff_dropout)

    def forward(self, 
                x: torch.tensor, 
                register: torch.tensor, 
                mask:torch.tensor = None)->tuple[torch.tensor, torch.tensor]:
        # x shape: (B, C, H, W) --> 
        
        B, C, H, W = x.shape
        
        B, R, C = register.shape
        
        # Flatten spatial dimensions and transpose: (B, C, H*W) -> (B, H*W, C)

        x_flat = x.flatten(2).transpose(1, 2)

        # Concat register tokens to x_flat here!!!  that is of shape (B, R, C)  

        x_flat_register = torch.concat([register, x_flat], axis = 1)  ###(B, R+H*W, C)

        # Multi-head self-attention
        residual = x_flat_register
        x_norm = self.norm1(x_flat_register)
        
        q = self.q_proj(x_norm).view(B, R+H*W, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, R+H*W, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, R+H*W, self.n_head, self.head_dim).transpose(1, 2)
        
        if self.normalize_qv:
            q = F.normalize(q, p = 2, dim = -1)
            k = F.normalize(k, p = 2, dim = -1)
        ## Glad to use flash attention here!!!
        if self.fast_att:
            with sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]):
                attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p = self.att_dropout if self.training else 0.0)
        else:
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            if mask is not None:
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, R+H*W, C)
        attn_output = self.o_proj(attn_output)
        
        
        x_flat = residual + self.dropout(attn_output)
        # Feed-forward network !!!
        residual = x_flat
        x_norm = self.norm2(x_flat)
        x_ff = self.ff_linear2(self.dropout(self.activation(self.ff_linear1(x_norm))))
        x_flat = residual + self.dropout(x_ff)
        
        # we split the register token from x_flat!!!
        register, x_flat = x_flat.split([R, H*W], dim = -2)
        
        # Reshape back to (B, C, H, W) !!!
        x = x_flat.transpose(1, 2).view(B, C, H, W).contiguous()
        
        return x, register  # Output shape: (B, C, H, W), (B, R, C)


"""
from training_utilities import MeasureTime
torch.manual_seed(0)
layer = EncoderLayer(fast_att=False)    
x = torch.randn(10, 768, 55, 55)
reg = torch.randn(10, 5, 768)
with MeasureTime():
    layer.eval()
    layer(x, reg)
"""
class Block(nn.Module):
    def __init__(
            self, 
            embedding_dim: int = 768,
            n_head: int = 8,
            activation_func: Callable = F.gelu,
            multiplication_factor: int = 2,
            ff_dropout: float = 0.2,
            att_dropout: float = 0.2,
            conv_kernel_size:int = 5,
            conv_activation:Callable = F.gelu,
            conv_first = False,
            normalize_qv:bool = True):
        super().__init__()
        self.t_block = EncoderLayer(
            embedding_dim= embedding_dim,
            n_head = n_head,
            activation_func=activation_func,
            multiplication_factor= multiplication_factor,
            ff_dropout=ff_dropout,
            att_dropout=att_dropout,
            normalize_qv=normalize_qv
        )
        self.conv_block = ConvMixer(
            embedding_dim= embedding_dim,
            kernel_size=conv_kernel_size,
            activation=conv_activation
        )
        self.conv_first = conv_first
    def forward(self, x:torch.tensor, 
                register:torch.tensor, 
                mask:torch.tensor = None)->tuple[torch.tensor, torch.tensor]:
        
        if not self.conv_first:
            x, register = self.t_block(x, register, mask)
            x = self.conv_block(x)
            return x, register
        x = self.conv_block(x)
        return self.t_block(x, register, mask)


"""
bl = block(conv_first=False)
bl.eval()
x, register = bl(x, register)
x.shape
register.shape
"""

class FinalBlock(nn.Module):
    def __init__(
            self, 
            embedding_dim: int = 768,
            n_head: int = 8,
            activation_func: Callable = F.gelu,
            multiplication_factor: int = 2,
            ff_dropout: float = 0.2,
            att_dropout: float = 0.2,
            normalize_qv:bool = True):
        super().__init__()
        self.t_block = EncoderLayer(
            embedding_dim= embedding_dim,
            n_head = n_head,
            activation_func=activation_func,
            multiplication_factor= multiplication_factor,
            ff_dropout=ff_dropout,
            att_dropout=att_dropout,
            normalize_qv=normalize_qv,
        )
    def forward(self, x:torch.tensor, 
                register:torch.tensor, 
                mask:torch.tensor = None)->tuple[torch.tensor, torch.tensor]:
        return self.t_block(x, register, mask)


class ClassificationHead(nn.Module):
    ## Here we embed C instead of the batch H*W
    ## this may sound a bit weirdo!!! 
    def __init__(self, 
                 embedding_dim:int = 768, 
                 output_classes:int=1000,
                 dropout:float = 0.2,
                 from_register:bool = True,
                 simple_output:bool = False, 
                 bias:bool = False,
                 ):
        
        super().__init__()
        self.from_register = from_register
        if from_register:
            if simple_output:
                self.output_head = nn.Sequential(*[nn.LayerNorm(embedding_dim),
                                        nn.Linear(embedding_dim, output_classes, bias = bias),
                                        ])
            else:
                self.output_head = nn.Sequential(*[nn.LayerNorm(embedding_dim),
                                        nn.Linear(embedding_dim, output_classes, bias = bias),
                                        nn.Tanh(),
                                        nn.Dropout(dropout),
                                        nn.Linear(output_classes, output_classes, bias = bias)
                                        ])
        else:
            self.output_head = nn.Sequential(*[
                nn.AdaptiveAvgPool2d((1,1)), 
                nn.Flatten(),
                nn.Linear(embedding_dim, output_classes, bias = bias)
            ])

    def forward(self, x:torch.tensor, registers:torch.tensor)-> torch.tensor:
        if self.from_register:
            return self.output_head(registers)[:, 0, :]
        return self.output_head(x)

"""
torch.manual_seed(0)
head = classification_head(768, 1000, from_register= True, simple_output= False)
head(torch.randn(10, 14*14, 768)).std()
"""




if __name__ == "__main__":
    print("Okkayy!!")

