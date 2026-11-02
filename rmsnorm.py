#why normalisation?
#while training the value of the weight might raise unexpectedly high and due to that
#following problems may occur
#activations can become excessively large or small
#training can become unstable
#optimization can become more difficult
#different layers can operate at very different scales

import torch
import torch.nn as nn
class RMSnorm(nn.Module):
  def __init_(self,d_model:int,eps:float=1e-5):#eps is a no very very small to prevent the division from zero
    super().__init__()#constructor for intialisation
    self.eps=eps
    self.weight=
    
