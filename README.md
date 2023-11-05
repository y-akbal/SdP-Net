
# A lighTweight ConvNet: AT-Net


<a href="https://www.youtube.com/shorts/3BW1lBgtbbs" class="follow"> 
<img align="left" width="350" height="200" src="at_net.JPG"> 
</a>
This is actually a less serious weekend project called AT-Net which can be considered a variation on Mobile-Net architecture. We start with a convolutional layer for patching then use depthwise and 1D convs for mixing channels. Finally use some encoder type transformer layers. Together with some register tokens and a class token. The convolution part is highly motivated by "Patches are all you need" paper. The dudes will be trained on ImageNet1k dataset. We shall offer five different sizes: XXXS (1.7M params), XXS (3M params), XS (4M params), S (27M params), L (55M params). 
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
A quick and dirty note: Training started few days ago. S model (25M params) obtains currently %52 percent accuracy on validation set (At the end of 32th epoch). Compared to other imagenet training logs there is an obvious overfitting here, I will let it be to see where its going to screw up. I will publicize the weights on HF soon after learning saturates. 
1
(The following table will be updated!!!)
| #Size  |  #Params  |  ConvMix | TransMix |  Patch_size | Conv_Size | Embed_Dim | Top1 Acc |  Top5 Acc | 
| :---:  | :-------: | :-----:  | :------: | :------:    | :------:  | :-----:   | :-----:  |   :-----:   | 
|  XXXS  |  3        |  5       |  7       |  8          |           | :-----:   | ??       |     | 
|  XXS   |  6        |  6       |  4       |  9          |           | :-----:   | ??       |     | 
|  S     |  25       |  10      |  11      |  12         |           | :-----:   | %79?(exp)|     |
|  S (*) |  25.5     |  10      |  11      |  12         |           | :-----:   | %??      |       |
|  L     |  55       | 100      |  100     |  100        |           | :-----:   | ??       |        |

*This dude will have multiple decision heads, coming from different register tokens.

# Optimizers
AdamW:
CosineAnnealing with warm starts::
 
# Augmentation and Regularization

RandAugment + Random erase + Random resize
