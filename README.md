
# SdP-Net - SlapDash Net 

This is actually a less serious weekend project called SlapDash-Net which can be considered a variation on hybrid Mobile-Net architecture. We start with a convolutional layer for patching then use depthwise and 1D convs for mixing channels. Finally use some encoder type transformer layers, together with some register tokens and a class token. The convolution part is highly motivated by "Patches are all you need" paper. The dudes will be trained on ImageNet1k dataset. 

 <h1> Our motto in coming up with SdP-Net:</h1>
 <ul> 
  <li> No promise to get very high accuracy,</li>
  <li> No prior assumption that exactly the same idea might have been used elsewhere,</li>
  <li> No attempt to tweak hyperparameters more than needed,</li>
  <li> We would like to hybridize things,</li>
  <li> We do bizzare combinations for the mere reason: because we would like to!!!,</li>
  <li> In SdP-Net we trust!</li>
  
</ul> 

# Training Details


| #Size  |  #Params  |  ConvMix | TransMix |  Patch_size | Conv_Size | Embed_Dim | Top1 Acc | 
| :---:  | :-------: | :-----:  | :------: | :------:    | :------:  | :-----:   | :-----:  | 
|  L     |  55M      | 5        |  7       |  16         |     7     | 768C 768T | 70.1%    | 
|  XL    |  76M      | 5        |  10      |  16         |     7     | 768C 768T | 75.0%    | 
|  XXL   |  86M      | 20       |  10      |  16         |     7     | 768C 768T | ?        | 


Bitter lesson: I have trained L and XL models with modicum of augmentation for over 200 epochs (XL is still being trained). Initial learning rate was 0.001*batch_size/512, with a linear warming up period for 5 epochs and cos-decay. The classification head is the same as that of the original VIT paper. A quick take away is that VIT like models suffer a lot from inductive bias issue. Even though adding some convolutional prior layers does not mitigate this.
On availabilty of better GPUs (Currently two V100s), I will increase the depth of the convolutional section and use some stochastic depth + EMA kinda stuff hoping to get at least 81 or 82% accuracy.  


# Optimizers
AdamW: lr = 0.001875
Weight decay 0.05
CosineAnnealing with warm starts in addition to 5 warming up epochs.
 
# Augmentation and Regularization

RandAugment + Random erase + Random resize+ CutMix + MixUp + Dropout(0.2) (Only to FFN parts of Attention layers) 

