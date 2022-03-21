
# -- python --
import math

# -- linalg --
import torch as th
import numpy as np
from einops import rearrange,repeat

# -- nn_fxn --
import torch.nn.functional as nnf

# -- local --
# from .utils import expand_patches

def wnnm_deno(patches,sigma):

    # -- reshape --
    pnoisy = patches.noisy
    print("pnoisy.shape: ",pnoisy.shape)
    pnoisy = rearrange(pnoisy,'b n pt c ph pw -> b n (pt c ph pw)')

    # -- run wnnm alg --
    deno = exec_wnnm(pnoisy/255.,sigma/255.)
    print("deno.shape: ",deno.shape)
    print(deno)

    # -- reshape --
    b,n,pt,c,ps,ps = patches.shape
    shape_str = 'b n (pt c ph pw) -> b n pt c ph pw'
    deno = rearrange(deno,shape_str,pt=pt,c=c,ph=ps)
    print("deno.shape: ",deno.shape)
    patches.noisy[...] = deno*255.

    return deno

def exec_wnnm(patches,sigma):

    # -- hyperparam --
    c = .1

    # -- svd --
    U,S,V = th.linalg.svd(patches,full_matrices=True)

    # -- re-weight diag --
    num = patches.shape[1]
    reweight(S,num,sigma**2,c)

    # -- denoised patches [U @ S @ V] --
    deno = th.matmul(th.matmul(U, th.diag_embed(S)), V.transpose(-2, -1)[:,:num])

    return deno

def reweight(sigma,num,sigma2,c,eps=1e-8):
    sigmaEst = th.sqrt(nnf.relu(sigma**2  - num*sigma2))
    weight = c*math.sqrt(num)/(sigmaEst+eps)
    sigma[...] = nnf.relu(sigma-weight)
