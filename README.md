
# A lighTweight ConvNet: AT-Net


<a href="https://www.youtube.com/shorts/3BW1lBgtbbs" class="follow"> 
<img align="left" width="350" height="200" src="at_net.JPG"> 
</a>
This is actually a less serious weekend project called AT-Net which can be considered a variation on Mobile-Net architecture. We start with a convolutional layer for patching then use depthwise and 1D convs for mixing channels. Finally use some encoder type transformer layers. Together with some register tokens and a class token. The convolution part is highly motivated by "Patches are all you need" paper. The dudes will be trained on ImageNet1k dataset. We shall offer five different sizes: XXXS (1.7M params), XXS (3M params), XS (4M params), S (19M params), L (55M params). 
 <h1> Our motto in coming up with AT-Net:</h1>
 <ul> 
  <li> No promise to get very high accuracy,</li>
  <li> No prior assumption that exactly the same idea might have been used elsewhere,</li>
  <li> No attempt to tweak hyperparameters more than needed,</li>
  <li> We would like to hybridize things,</li>
  <li> We do bizzare combinations for the mere reason: because we would like to!!!,</li>
  <li> In AT-Net we trust!</li>
  
</ul> 
Enjoy AT-Net!!!

# Training Details

(The following table will be updated!!!)
| #Size  |  #Params  |  ConvMix | TransMix |  Patch_size | Conv_Size | Embed_Dim | Top1 Acc | 
| :---:  | :-------: | :-----:  | :------: | :------:    | :------:  | :-----:   | :-----:  | 
|  XXXS  |  3        |  5       |  7       |  8          |           |           | ??       | 
|  XXS   |  6        |  6       |  4       |  9          |           |           | ??       | 
|  S     |  19       |  5       |  10      |  12         |           |           | 79%?(exp)| 
|  S (*) |  19       |  5       |  10      |  14         |     7     | 512C 512T | 70.2%    | 
|  M     |  33       |  5       |  15      |  14         |     7     | 768C 512T | ?        | 
|  L     |  55       | 100      |  100     |  100        |           |           | ??       | 
|  XL    |  ~70      | 5        |  10      |  14         |     7     | 512C 768T | ??       | 


*This dude will have multiple decision heads, coming from different register tokens.

Bitter lesson: I have trained S (*) model, with modicum of augmentation and and small dropout rate for 300 epochs. 70% accuracy is not really good. Initial learning rate was 0.0001, with a linear warming up period for 5 epochs. I was expecting at least 75% Top1 accuracy. Probably I will keep the patch size small (say 7) and conv-kernel size 5, in which case together with some more augmentation methods things will be better. 

As of 20.11.23 training XL and M models. Unlike classical ViT's (that is using MLP heads) I am using global average pooling.  

# Optimizers
AdamW: lr = 0.0009 -
CosineAnnealing with warm starts in addition to 5 warming up epochs.
 
# Augmentation and Regularization

RandAugment + Random erase + Random resize+ CutMix + MixUp + Dropout(0.2) (Only to FFN parts of Attention layers) 

