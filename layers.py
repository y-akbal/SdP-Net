import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.nn.functional as F

## Here we will have layers to be used 
## We shall mostly use the optimized torch layers
## rather than coming up with our own implementations

class conv_int(nn.Module):
    def __init__(self, embedding_dim = 128, patch_size = 4):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.conv = nn.Conv2d(in_channels = 3, 
                              out_channels = embedding_dim,
                              kernel_size = patch_size,
                              stride = patch_size, 
                              
                              )
        self.batch_norm = nn.SyncBatchNorm(embedding_dim)
    def forward(self, x):
        x = self.conv(x)
        return self.batch_norm(x) 


class conv_mixer(nn.Module):
    def __init__(self, 
                 embedding_dim = 512, 
                 kernel_size = 5, 
                 activation = nn.GELU()):
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
        self.batch_norm_1 = nn.SyncBatchNorm(embedding_dim)
        self.batch_norm_2 = nn.SyncBatchNorm(embedding_dim)
        self.activation = activation
    def forward(self, x_):
        x = self.conv2d(x_)
        x = self.activation(x)
        x = self.conv1d(x_+self.batch_norm_1(x))
        x = self.activation(x)
        x = self.batch_norm_2(x)
        return x
"""
con = conv_mixer(512)
k = 0
for i in con.parameters():
     k += i.shape.numel()
print(k)
""" 
class squeezer(nn.Module):
    def __init__(self, 
                 embedding_dim = 512,
                 squeeze_ratio = 5,
                 activation = nn.GELU()
                 ):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(embedding_dim, 
                              embedding_dim, 
                              kernel_size = squeeze_ratio,
                              stride = squeeze_ratio,
                              groups = embedding_dim
                              ),
                              nn.SyncBatchNorm(embedding_dim),
                              activation
        )
        
    def forward(self, x):
        return self.seq(x)
"""
squeezer(128, squeeze_ratio = 7)(torch.randn(1, 128, 224,224)).shape     
for i in con.parameters():
     k += i.shape.numel()
print(k) 
"""
"""
k = 0
for i in nn.TransformerEncoderLayer(512, nhead = 8, batch_first=True,
                                    dim_feedforward=256).parameters():
    k += i.shape.numel()
print(k) 
"""
class embedding_layer(nn.Module):
    ## We isolated this layer in the case that you want to 
    ## do something like enumerating the pixels...
    def __init__(self, embedding_dim_in:int = 512,
                 embedding_dim_out:int = 512,
                 num_registers:int = 1,
                 ):
        super().__init__()
        ### -- ###
        self.embedding = nn.Embedding(num_registers, embedding_dim_in)
        self.num_registers = num_registers

        if embedding_dim_in != embedding_dim_out:
            self.Linear = nn.Linear(embedding_dim_in, embedding_dim_out)
        else:
            self.Linear = lambda x: x 
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
    def forward(self, x, y = None):
        ### Here y will be localtions as there will be 2 more inputs that we save for extra 
        
        B, C, H, W = x.shape
        # C here is the embedding dimension!!!
        x = x.view(B, C, H*W).contiguous().transpose(-1,-2)

        if y == None:
            embeddings = self.embedding(self.num_register)
        else:
            embeddings = self.embedding(y)
        
        x = torch.cat((embeddings.repeat(B, 1, 1), x), 1)
        return self.Linear(x)

class encoder_layer(nn.Module):
    ## Here we embed C instead of the batch H*W
    ## this may sound a bit weirdo!!! 
    def __init__(self, 
                 embedding_dim, 
                 n_head:int,
                 activation_func = nn.GELU(),
                 multiplication_factor:float = 2,
                 dropout = 0.2
                 ):
        assert embedding_dim*multiplication_factor > 1, "Come on dude, do not squeeze to much"
        super().__init__()
        self.embedding_dim = embedding_dim
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = n_head,
            activation = activation_func,
            batch_first = True,
            dim_feedforward = int(multiplication_factor*self.embedding_dim),
            dropout = dropout,
            norm_first= True
            )
        
    def forward(self, x):
        return self.transformer_layer(x)

## Below is just for extreme testing purposes. I would like to see how attention stuff works in MLP part by just doing 
## some experiments on them. 

class libido_killer(nn.Module):
  def __init__(self, 
               d_in = 512, 
               squeeze_ratio = 0.5, 
               dropout = 0.2,
               activation = nn.GELU("tanh")): ## Silu might be better here
    super().__init__()
    self.intermediate_dim = int(d_in*squeeze_ratio)
    self.W = nn.Parameter(torch.randn(d_in, self.intermediate_dim)/((d_in+self.intermediate_dim)/2)**0.5)
    self.bias_out = nn.Parameter(torch.zeros(d_in))
    self.bias_int = nn.Parameter(torch.zeros(self.intermediate_dim))
    self.activation = activation
    self.dropout = nn.Dropout1d(dropout)
  def forward(self, X):
    X = self.dropout(X)
    intermediate = self.activation(X @ self.W+self.bias_int)
    return intermediate @ self.W.transpose(-1,-2)+self.bias_out


class cheap_attention(nn.Module):
  def __init__(self, embed_dim = 512, num_heads = 8, dropout=0.1, **kwargs):
    super().__init__(**kwargs)
    self.dropout = dropout
    self.Linear = nn.Linear(embed_dim, embed_dim, bias = False)
    #torch.nn.init.orthogonal_(self.Linear.weight, gain = 0.3)
    
    assert (embed_dim/num_heads).is_integer(), "Embedding dimension is supposed be divisible by num_heads comrade!!!"
    self.num_heads = num_heads
    
  def forward(self, x):
    B, H, W = x.shape
    x = self.Linear(x)
    x = x.view(-1, self.num_heads, H, int(W//self.num_heads))
    with torch.backends.cuda.sdp_kernel(enable_math=False):
        x_ = F.scaled_dot_product_attention(x,x,x, dropout_p = self.dropout, 
                                          is_causal= False)
    return x_.contiguous().view(B,H,W)

class freak_attention_encoder(nn.Module):
    def __init__(self, d_dim = 512, 
                 num_heads = 8,
                 dropout_att = 0.1,
                 dropout_ffn = 0.2,
                 squeeze_ratio = 0.5,
                 activation = nn.GELU("tanh"),
                 **kwargs,
                 
    ):
        super().__init__(**kwargs)
        self.ffn = libido_killer(d_in = d_dim,
                                 dropout= dropout_ffn,
                                 squeeze_ratio= squeeze_ratio,
                                 activation = activation
                                 )
        self.norm_1 = nn.LayerNorm(d_dim)

        self.cheap_attention = cheap_attention(embed_dim= d_dim,
                                               num_heads= num_heads,
                                               dropout = dropout_att,
                                               )
        self.norm_2 = nn.LayerNorm(d_dim)

    def forward(self, x):
        x_ = self.norm_1(x)
        x_ = self.cheap_attention(x_)
        x_ += x
        x__ = self.norm_2(x_)
        return self.ffn(x__) + x_
"""
q = 0
for p in freak_attention_encoder(squeeze_ratio=0.5, d_dim = 512).parameters():
    q += p.shape.numel()
s = 0
for p in nn.Conv2d(3, 768, 14).parameters():
    s += p.shape.numel()

6*q+3*s
"""
if __name__ == "__main__":
    print("Okkayy!!")

