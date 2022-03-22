
# -- python deps --
import torch
import scipy
import numpy as np
from einops import rearrange

# -- faiss --
from faiss.contrib import kn3
# import faiss

# -- package --
# import npc.search_mask as imask
# from npc.utils import groups2patches
from npc.utils import Timer
from npc.utils import bufs_npc2faiss,args_npc2faiss

@Timer("agg_patches_faiss")
def agg_patches_faiss(step,patches,images,bufs,args):

    # -- default stream --
    # if cs_ptr is None:
    #     cs_ptr = torch.cuda.default_stream().cuda_stream

    # -- filter by valid --
    # valid = torch.nonzero(torch.all(bufs.inds!=-1,1))[:,0]
    # vnoisy = patches.noisy[valid]
    # vinds = bufs.inds[valid]
    # vvals = bufs.vals[valid]
    faiss_args = args_npc2faiss(args)
    # faiss_bufs = bufs_npc2faiss(patches,bufs)
    bsize = patches.shape[0]
    t,c,h,w = images.shape
    npix = t*h*w
    numQueriesTotal = ((npix-1) // args.stride) + 1
    bsize = min(bsize,numQueriesTotal)

    qstart = args.bsize*step
    # kn3.run_search(srch_img,qstart,bsize,flows,args.sigma,
    #                faiss_args,faiss_bufs,pfill=True)
    pnoisy = patches.noisy[:bsize]
    # pnoisy = patches.clean[:bsize]
    imin,imax = bufs.inds.min().item(),bufs.inds.max().item()
    # print("bufs.inds[min,max]: ",imin,imax)
    # print("bufs.inds.shape: ",bufs.inds.shape)
    # print("qstart,step: ",qstart,step)
    # print("images.deno.shape: ",images.deno.shape,images.deno.device)
    # print("pnoisy.shape: ",pnoisy.shape)

    kn3.run_fill(images.deno,pnoisy,qstart,faiss_args,"p2b",None)#bufs.inds)
    # print("post fill.")

    # # -- iterate over "nkeep" --
    # if args.nkeep != -1:
    #     vinds = bufs.inds[:,:args.nkeep]

    # -- agg batch --
    # compute_agg_batch(images.deno,vnoisy,vinds,images.weights,
    #                   vvals,images.vals,args.ps,args.ps_t,cs_ptr)
