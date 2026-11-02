from dataclasses import dataclass

@dataclass
class Cofig:
  vocab_siz: int =8192
  context_len: int =512
  d_model: int=256
  n_layer: int=4
  n_head: int=4
  d_ff: int =688
  rope_theta:float=10000.0
  norm_eps:float=1e-5
  tie_embeddings: bool=True

#before starting of with coding a transformer or large language model 
#we need to fix some of the features or functuionalities as constants

def __post_init__(self):
  assert self.d_model % self.n_head==0
  assert self.n_head % self.n_kv_head==0

@property
def head_dim(self)->int:
  return self.d_model//self.n_head

@property
def n_rep(self)->int:
  return self.n_head//self.n_kv_head
