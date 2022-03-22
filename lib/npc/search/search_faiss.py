"""
Search for similar patches across batches
"""



# -- python imports --
import torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- [a required package] --
from faiss.contrib import kn3

# -- local package --
import npc.search_mask as search_mask

# -- utils --
from npc.utils.batching import view_batch
from npc.utils.logger import vprint
from npc.utils import divUp
from npc.utils import Timer
from npc.utils import bufs_npc2faiss,args_npc2faiss

@Timer("exec_search_faiss")
def exec_search_faiss(step,patches,imgs,flows,bufs,args):

    # -- setup --
    bsize = args.bsize
    cs = th.cuda.default_stream()
    cs_ptr = th.cuda.default_stream().cuda_stream
    done = False

    # --reset values --
    bufs.inds[...] = -1
    bufs.vals[...] = float("inf")

    # -- max size --
    t,c,h,w = imgs.shape
    npix = t*h*w
    numQueriesTotal = ((npix-1) // args.stride) + 1

    # -- smaller batch sizes impact quality --
    assert args.nstreams == 1
    for index in range(args.nstreams):

        # -- compute batch size --
        qstart = step * (args.bsize + index) # the query index
        bsize_b = min(bsize,numQueriesTotal - qstart)

        # -- grab batch --
        vbufs = edict()
        for key in bufs.keys():
            vbufs[key] = view_batch(bufs[key],bsize_b,index)

        vpatches = edict()
        for key in patches.keys():
            vpatches[key] = view_batch(patches[key],bsize_b,index)

        # -- exec search --
        assert index == 0
        search_and_fill(imgs,vpatches,vbufs,qstart,flows,args)

        # -- wait for all streams --
        torch.cuda.synchronize()

    # -- update term. condition --
    done = False

    return done

def search_and_fill(imgs,patches,bufs,qstart,flows,args):

    # -- select search image --
    if args.srch_img == "noisy":
        srch_img = imgs.noisy
    elif args.srch_img == "basic":
        srch_img = imgs.basic
    elif args.srch_img == "clean":
        srch_img = imgs.clean
    elif args.srch_img == "search":
        srch_img = imgs.search
    else:
        raise ValueError(f"uknown search image [{srch_img}]")

    # -- sim search block --
    bufs.inds[...] = -1
    bufs.vals[...] = float("inf")

    # -- exec search --
    # srch_img = imgs.clean
    t,c,h,w = imgs.shape
    faiss_args = args_npc2faiss(args)
    faiss_bufs = bufs_npc2faiss(patches,bufs)
    bsize = faiss_bufs.patches.shape[0]
    kn3.run_search(srch_img,qstart,bsize,flows,args.sigma,
                   faiss_args,faiss_bufs,pfill=True)
    nl_inds = faiss_bufs.inds # non-local inds

    # -- fill patches --
    for key in imgs.patch_images:

        # -- skip --
        pass_key = (imgs[key] is None) or (patches[key] is None)
        if pass_key: continue

        # -- fill --
        kn3.run_fill(imgs[key],patches[key],qstart,faiss_args,"b2p",nl_inds)
