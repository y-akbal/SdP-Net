import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from typing import Callable
from torch.nn.attention import SDPBackend, sdpa_kernel



## Here we will have layers to be used 
## We shall mostly use the optimized torch layers
## rather than coming up with our own implementations

class conv_patcher(nn.Module):
    def __init__(self, 
                 embedding_dim = 128, 
                 patch_size = 4):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.conv = nn.Conv2d(in_channels = 3, 
                              out_channels = embedding_dim,
                              kernel_size = patch_size,
                              stride = patch_size, 
                              bias = False
                              )
        self.patch_size = patch_size
    def forward(self, x:torch.Tensor)->torch.Tensor:
        ## B, 3, H, W --> B, 3, H*W ### 
        B, C, _, _ = x.shape
        x = self.conv(x) 
        return x
"""
conv_patcher(patch_size=16)(torch.randn(5, 3, 224,224)).is_contiguous()
"""

class classification_head(nn.Module):
    ## Here we embed C instead of the batch H*W
    ## this may sound a bit weirdo!!! 
    def __init__(self, 
                 embedding_dim:int = 768, 
                 output_classes:int=1000,
                 dropout:float = 0.2,
                 bias:bool = False,
                 ):
        
        super().__init__()
        self.output_head = nn.Sequential(*[nn.LayerNorm(embedding_dim),
                                        nn.Linear(embedding_dim, output_classes),
                                        nn.Tanh(),
                                        nn.Dropout(dropout),
                                        nn.Linear(output_classes, output_classes, bias = bias)
                                        ])   

    def forward(self, x):
        return self.output_head(x)


"""
torch.manual_seed(0)
classification_head(768, 10)(torch.randn(10, 12, 768))    
"""    


class conv_mixer(nn.Module):
    def __init__(self, 
                 embedding_dim:int = 768, 
                 kernel_size:int = 5, 
                 activation:Callable = nn.GELU()):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels = embedding_dim, 
                              out_channels = embedding_dim,
                              kernel_size = kernel_size,
                              groups = embedding_dim,
                              padding = "same",
                              )
        self.conv1d = nn.Conv2d(in_channels = embedding_dim,
                                out_channels = embedding_dim,
                                kernel_size =1,
                                )
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.activation = activation

    def forward(self, x_:torch.Tensor)->torch.Tensor:
        """
        Shape here is to be of the form: B, C, H, W --> B, C, H, W
        """
        x = self.conv2d(x_)
        x = self.activation(x)
        x = self.conv1d(x_+self.layer_norm_1(x))
        x = self.activation(x)
        x = self.layer_norm_2(x)
        return x


class embedding_layer(nn.Module):
    ## We isolated this layer in the case that you want to 
    ## do something like enumerating the pixels...
    def __init__(self, 
                 embedding_dim: int = 768,
                 num_registers:int = 1,
                 max_image_size:list[int, int] = [14,14],
                 activation:Callable = None
                 ):
        super().__init__()
        ### -- ###
        self.num_registers = num_registers
        self.vertical_im_size = max_image_size[0]
        self.horizontal_im_size = max_image_size[1]
        self.activation = activation if activation != None else lambda x: x
        ###
        self.register_embedding_layer = nn.Embedding(num_registers, embedding_dim)
        self.vertical_embedding_layer = nn.Embedding(max_image_size[0], embedding_dim)
        self.horizontal_embedding_layer = nn.Embedding(max_image_size[1], embedding_dim)
        ### --- ###
        ### --- ###
        ### --- ###
        ### ----###
        self.register_buffer(
            "num_register",
            torch.tensor(
                [i for i in range(self.num_registers)],
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


    def forward(self, x:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
        
        B, C, H, W = x.shape

        ## NO NEED TO COMPUTE THESE DUDES FOR EACH FORWARD PASS!!!!! Do we have kind a caching mechanism?

        register_embeddings = self.register_embedding_layer(self.num_register)
        horizontal_embeddings = self.horizontal_embedding_layer(self.horizontal_embedding[:H]).transpose(-1, -2).unsqueeze(-1)
        vertical_embeddings = self.vertical_embedding_layer(self.vertical_embedding[:W]).transpose(-1,-2).unsqueeze(-2)

        ## Do something here!!! to add the vectors both verticall and horizontally!!!
        ## Here you may repeat some tokens --- !!!
        x += horizontal_embeddings
        x += vertical_embeddings
        ## and the concat with register tokens, and give the output to the 
        return  self.activation(x), register_embeddings

"""
_, x = embedding_layer(max_image_size=[15,15])(torch.randn(1, 768, 14, 15))
x.shape
"""
x = torch.randn(4, 3, 2)
y = torch.randn(3, 2)

# Step 1: Expand y to match the shape of x
y_expanded = y.unsqueeze(0).expand(2, -1, -1).shape
# Now y_expanded has shape (4, 3, 2)

class EncoderLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        n_head: int,
        activation_func: Callable = F.gelu,
        multiplication_factor: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        assert channels % n_head == 0, "Number of channels must be divisible by n_head"
        
        self.channels = channels
        self.n_head = n_head
        self.head_dim = channels // n_head

        # Multi-head self-attention
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.o_proj = nn.Linear(channels, channels)
        
        # Feed-forward network
        self.ff_linear1 = nn.Linear(channels, multiplication_factor * channels)
        self.ff_linear2 = nn.Linear(multiplication_factor * channels, channels)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Activation and dropout
        self.activation = activation_func
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor, register: torch.tensor, mask:torch.tensor)->tuple[torch.tensor, torch.tensor]:
        # x shape: (B, C, H, W) --> 
        
        B, C, H, W = x.shape
        
        R, C = register.shape
        
        # Flatten spatial dimensions and transpose: (B, C, H*W) -> (B, H*W, C)

        x_flat = x.flatten(2).transpose(1, 2)

        # Concat register tokens to x_flat here!!!  that is of shape (R, C)  

        x_flat_register = torch.concat([x_flat, register.unsqueeze(0).repeat(B, -1, -1)], axis = 1)  ###(B, H*W+R, C)

        # Multi-head self-attention
        residual = x_flat_register
        x_norm = self.norm1(x_flat_register)
        
        q = self.q_proj(x_norm).view(B, H*W+R, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, H*W+R, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, H*W+R, self.n_head, self.head_dim).transpose(1, 2)
        
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p = self.attn_dropout)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, H*W+R, C)
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
        x = x_flat.transpose(1, 2).view(B, C, H, W)
        
        return x, register[0, :, :]  # Output shape: (B, C, H, W), (R, C)
    

"""
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(10, 196, 768, device=device)
layer = EncoderLayer_conv().to(device)
layer = torch.compile(layer)
layer(x)
# Measure transpose
start = time.time()
for _ in range(1000):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        y = layer(x)
        torch.cuda.synchronize()
transpose_time = (time.time() - start) / 1000

print(f"Transpose time: {transpose_time*1e6:.2f} microseconds")
"""



if __name__ == "__main__":
    print("Okkayy!!")

