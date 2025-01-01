# SdP-Net - SlapDash Net

This is actually a less serious weekend project called SlapDash-Net which can be considered a less serious variation on VIT architecture. We use some encoder type transformer layers, together with some register tokens. Prior to encoder encoder layer we introduce some convolution layers in a highly slapdash manner. The dudes will be trained on ImageNet1k/22k dataset.  

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


| #Size  |  #Params  |  #Blocks  |  Patch_size | Conv_Size | Embed_Dim | Top1 Acc | 
| :---:  | :-------: |  :------: | :------:    | :------:  | :-----:   | :-----:  | 
|  XXS   |  55M      |   7       |  16         |     7     | 128       | ?        | 
|  S     |  76M      |   12      |  16         |     7     | 512       | ?        | 
|  M     |  86M      |   12      |  16         |     7     | 768       | ?        | 
|  L     |  86M      |   12      |  16         |     7     | 768       | ?        | 
|  XL    |  101      |   15      |  14         |     7     | 768       | 82.1     | 



Bitter lesson: The biggest model gives 82.1 acc on Imagenet1K. Still training the others on time permitting. 

# Optimizers

AdamW: lr = 0.001875 (=0.001*batch_size/512)
Weight decay 0.05
CosineAnnealing with warm starts in addition to 5 warming up epochs.
 
# Augmentation and Regularization

RandAugment + Random erase + Random resize+ CutMix + MixUp + Dropout(0.2) (Only to FFN parts of Attention layers) 

# Augmentation and Regularization

#TODO

0) Gating mechanism in FFN?
1) LayerScale --> This will be needed for deeper networks!
2) Neighborhood embedding --> See layers ConvEmbedding layer (A larger embedding look up dictionary is used and for an individual patch, a neighbour of embeddings are averaged!)
9) Use KeLü activation instead of GeLu (KeLü implemented but may not be really optimized!)
