"""

Utils for batching pixels

"""

import torch as th
from .utils import divUp

def batch_params(mask,bsize,nstreams):
    # -- batching height and width --
    tsize = bsize*nstreams
    nelems = th.sum(mask).item()
    # nbatches = divUp(divUp(nelems,nstreams),bsize)
    nbatches = divUp(nelems,nstreams*bsize)
    return nelems,nbatches

def batch_params_faiss(shape,bsize,stride):
    # -- unpack --
    t,c,h,w = shape
    nelems = t*h*w

    # -- num iters total --
    nqueries = (nelems - 1)//stride + 1

    # -- size per "search" call --
    nbatches = (nqueries-1) // bsize + 1
    return nqueries,nbatches

def divUp(a,b): return (a-1)//b+1

def get_hw_batches(h,w,bsize):
    hbatches = th.arange(0,h,bsize)
    wbatches = th.arange(0,w,bsize)
    return hbatches,wbatches

def view_batch(tensor,start,size):
    if tensor is None: return None
    bslice = slice(start*size,(start+1)*size)
    return tensor[bslice]

def view_image(tensor,h_start,w_start,size):
    if tensor is None: return None
    hslice = slice(h_start,h_start+size)
    wslice = slice(w_start,w_start+size)
    return tensor[...,hslice,wslice]


