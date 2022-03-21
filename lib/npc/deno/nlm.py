
# -- python --
import math

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- nn_fxn --
import torch.nn.functional as nnf

def nlm_deno(patches,sigma):

    # -- run wnnm alg --
    deno = exec_nlm(patches,sigma)

    return deno

def exec_nlm(patches,sigma):

    # -- weights --
    print("patches.shape: ",patches.shape)
    pnoisy = patches.noisy/255.
    dims = (2,3,4,5)
    weights = th.exp(-th.mean((pnoisy[:,[0],:,:1] - pnoisy[:,:,:1])**2,dims,True))
    print("weights.shape: ",weights.shape)
    weights /= th.sum(weights,1,True)
    print("weights.shape: ",weights.shape)

    # -- compute deno --
    n = patches.shape[1]
    deno = th.sum(weights * pnoisy,1,True)
    patches.noisy[:,0] = deno[:,0]

    return deno

