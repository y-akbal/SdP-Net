import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Callable
from torch.nn.attention import SDPBackend, sdpa_kernel
from utility_layers import StochasticDepth as SD
from typing import Tuple, Callable


class LayerNorm(nn.Module):
    def __init__(self, 
                 embedding_dim:int, 
                 eps:float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        self.eps = eps
    def forward(self, x:torch.Tensor)->torch.Tensor:
        mean = x.mean([1], keepdims=True)
        var = x.var([1], keepdims=True, unbiased=False)
        x = (x-mean)/(var+self.eps)**0.5
        return self.gamma[:, None, None]*x + self.beta[:, None, None]



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
    def forward(self, x:torch.Tensor)->torch.Tensor:
        ## B, 3, H, W --> B, out_channels, H//PATCH_SIZE, W//PATCH_SIZE ### 
        return self.conv(x) 
"""
ConvPatcher(patch_size=16)(torch.randn(5, 3, 224,224)+1).mean()
x_r = x.reshape(5, 32, 4, 14, 14)
x = (x_r - x_r.mean([-1,-2,-3], keepdim = True))/(x_r.var([-1,-2,-3], keepdim = True, unbiased = False)+1e-05)**0.5
x_r = x.reshape(5, 128, 14, 14)
x_r ## This dudes agree!!!
x_n
"""


"""#let's write a simple tenosr of shape 1, C, H, W
#sample from laplace distribution in torch
x = 10*torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0])).sample((2, 3, 2, 2)).squeeze(-1)

(x- x.mean(dim = [-1,-2], keepdim = True))/x.var(dim = [-1,-2], keepdim = True, unbiased = False)**0.5

(x-x.mean((-1,-2,-3)))/x.std((-1,-2,-3), unbiased = False)

nn.GroupNorm(1, 3, eps = 0.0)(x)"""

class ConvMixer(nn.Module):
    def __init__(self, 
                 embedding_dim:int = 768, 
                 kernel_size:int = 5, 
                 activation:Callable = nn.GELU(),
                 drop_p:float = 0.0,
                 mixer_ffn_bias:bool = True,
                 mixer_deptwise_bias:bool = True,
                 ):
        super().__init__()
        self.conv2d = nn.Sequential(*[nn.Conv2d(in_channels = embedding_dim, 
                              out_channels = embedding_dim,
                              kernel_size = kernel_size,
                              groups = embedding_dim,
                              padding = "same",
                              bias = mixer_deptwise_bias,
                              ), nn.Conv2d(in_channels = embedding_dim,
                                out_channels = embedding_dim,
                                kernel_size =1,
                                bias = mixer_ffn_bias)])
        self.conv1d = nn.Sequential(*[nn.Conv2d(in_channels = embedding_dim,
                                out_channels = 4*embedding_dim,
                                kernel_size =1,
                                bias = mixer_ffn_bias),
                                activation,
                                nn.Conv2d(in_channels = 4*embedding_dim, 
                                          out_channels = embedding_dim, 
                                          kernel_size = 1, 
                                          bias = mixer_ffn_bias)
                                ])

        self.layer_norm_1 = LayerNorm(embedding_dim)
        self.layer_norm_2 = LayerNorm(embedding_dim)
        self.activation = activation
        self.drop_path_1 = SD(drop_p) if drop_p > 1e-5 else nn.Identity()
        self.drop_path_2 = SD(drop_p) if drop_p > 1e-5 else nn.Identity()


    def forward(self, x:torch.Tensor):
        x_ = self.drop_path_2(self.activation(self.conv2d(self.layer_norm_1(x)))) + x
        x =  self.drop_path_1(self.conv1d(self.layer_norm_2(x_))) + x_
        return x

"""
layer = ConvMixer(768, kernel_size=7, drop_p=0.5)
x = torch.randn(2, 768, 14, 14)
layer(x).mean()
q = 0
for p in layer.parameters():
    q += p.shape.numel()
print(q)
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
        self.activation = activation if activation != None else torch.nn.Identity()
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
            torch.arange(max_image_size[0], dtype=torch.int),
        )

        self.register_buffer(
            "horizontal_embedding",
            torch.arange(max_image_size[1], dtype=torch.int),
        )
    
    def forward(self, 
                x:torch.Tensor, 
                num_registers:int = 0)->Tuple[torch.Tensor, torch.Tensor]:
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
EmbeddingLayer(max_image_size=[15,15])(torch.randn(3, 768, 15, 15), 2)[0].shape
"""

class ConvEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim:int = 768,
                 kernel_size:int = 5,
                 activation:Callable = nn.GELU(),
                 max_image_size:list[int, int] = [14,14],
                 seed:int = 0,
                 trainable_bone:bool = False,
                 ):
        super().__init__()
        torch.manual_seed(seed)
        self.conv2d = nn.Conv2d(in_channels = embedding_dim, 
                      out_channels = embedding_dim,
                      kernel_size = kernel_size,
                      groups = embedding_dim,
                      bias = False,
                      )
        self.kernel_size = kernel_size
        if trainable_bone:
            self.register_parameter("bone", Parameter(0.02*torch.randn(1, embedding_dim, 
                             max_image_size[0]+kernel_size, 
                             max_image_size[1]+kernel_size)))
        else:
            self.register_buffer("bone", 0.02*torch.randn(1, embedding_dim, 
                             max_image_size[0]+kernel_size, 
                             max_image_size[1]+kernel_size,
                             requires_grad=False))
        self.activation = activation
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
            return self.activation(x + self.conv2d(self.bone[:,:,:x.shape[-2]+self.kernel_size-1, :x.shape[-1]+self.kernel_size-1]))
     
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
        normalize_qv:bool = True,
        drop_p:float = 0.1
    ):
        super().__init__()
        assert embedding_dim % n_head == 0, "Number of embedding_dim must be divisible by n_head"
        
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.head_dim = embedding_dim // n_head
        self.att_dropout = att_dropout
        self.fast_att = fast_att
        self.q_norm = nn.LayerNorm(self.head_dim) if normalize_qv else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if normalize_qv else nn.Identity()
        ## Stochastic Depth for attention and feed-forward network
        self.drop_path1, self.drop_path2 = (SD(drop_p), SD(drop_p)) if drop_p > 1e-5 else (nn.Identity(), nn.Identity())

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
    
        q, k = self.q_norm(q), self.k_norm(k) 

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
        attn_output = self.dropout(self.o_proj(attn_output))

        x_flat = residual + self.drop_path1(attn_output)

        # Feed-forward network !!!
        residual = x_flat
        x_norm = self.norm2(x_flat)
        x_ff = self.dropout(self.ff_linear2(self.dropout(self.activation(self.ff_linear1(x_norm)))))
        x_flat = residual + self.drop_path2(x_ff)
        # we split the register token from x_flat!!!
        register, x_flat = x_flat.split([R, H*W], dim = -2)
        
        # Reshape back to (B, C, H, W) !!!
        x = x_flat.transpose(1, 2).view(B, C, H, W).contiguous()
        
        return x, register  # Output shape: (B, C, H, W), (B, R, C)




"""
from training_utilities import MeasureTime
q = 0
for p in layer.parameters():
    q += p.shape.numel()
print(q)
torch.manual_seed(0)
layer = EncoderLayer(embedding_dim=768, 
                    n_head=16, 
                    multiplication_factor = 4, 
                    fast_att=True, 
                    normalize_qv=True)    
x = torch.randn(1, 768, 14, 14)
reg = torch.randn(1, 5, 768)
layer(x, reg)[1].std()
"""
class Block(nn.Module):
    def __init__(
            self, 
            embedding_dim: int = 768,
            n_head: int = 8,
            conv_block_num:int = 2,
            activation_func: Callable = nn.GELU(),
            multiplication_factor: int = 2,
            ff_dropout: float = 0.2,
            att_dropout: float = 0.2,
            conv_kernel_size:int = 5,
            conv_activation:Callable = nn.GELU(),
            conv_first = False,
            normalize_qv:bool = True,
            mixer_ffn_bias:bool = False,
            mixer_deptwise_bias:bool = False,
            drop_p:float = 0.1,
            fast_att:bool = True,):
        super().__init__()
        self.t_block = EncoderLayer(
            embedding_dim= embedding_dim,
            n_head = n_head,
            activation_func=activation_func,
            multiplication_factor= multiplication_factor,
            ff_dropout=ff_dropout,
            att_dropout=att_dropout,
            normalize_qv=normalize_qv,
            drop_p = drop_p,
            fast_att=fast_att)
        
        self.conv_blocks = nn.Sequential(*[ConvMixer(
            embedding_dim= embedding_dim,
            kernel_size=conv_kernel_size,
            activation=conv_activation,
            drop_p=drop_p,
            mixer_deptwise_bias=mixer_deptwise_bias,
            mixer_ffn_bias=mixer_ffn_bias) for _ in range(conv_block_num)]) 

        self.conv_first = conv_first

    def forward(self, x:torch.tensor, 
                register:torch.tensor, 
                mask:torch.tensor = None)->tuple[torch.tensor, torch.tensor]:
        
        if not self.conv_first:
            x, register = self.t_block(x, register, mask)
            x = self.conv_blocks(x)
            return x, register
        x = self.conv_blocks(x)
        return self.t_block(x, register, mask)


"""
Block()
bl = Block(conv_first=False, conv_block_num=1, mixer_ffn_bias=True, mixer_deptwise_bias=True)
x = torch.randn(1, 768, 14, 14)
register = torch.randn(1, 5, 768)
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
            normalize_qv:bool = True,
            drop_p:float = 0.0):
        super().__init__()
        self.t_block = EncoderLayer(
            embedding_dim= embedding_dim,
            n_head = n_head,
            activation_func=activation_func,
            multiplication_factor= multiplication_factor,
            ff_dropout=ff_dropout,
            att_dropout=att_dropout,
            normalize_qv=normalize_qv,
            drop_p=drop_p,
        )
    def forward(self, x:torch.tensor, 
                register:torch.tensor, 
                mask:torch.tensor = None
                )->tuple[torch.tensor, torch.tensor]:
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
            return self.output_head(registers.mean(-2))
        return self.output_head(x)



if __name__ == "__main__":
    print("Okkayy!!")

