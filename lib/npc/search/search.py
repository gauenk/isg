"""
Search for similar patches across batches
"""



# -- python imports --
import torch
import torch as th
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- [a required package] --
import vpss

# -- local package --
import npc.search_mask as search_mask

# -- utils --
from npc.utils.batching import view_batch
from npc.utils.logger import vprint
from npc.utils import divUp
from npc.utils import Timer

@Timer("exec_search_eccv2022")
def exec_search_eccv2022(patches,imgs,flows,mask,bufs,args):

    # -- setup --
    bsize = args.bsize
    cs = th.cuda.default_stream()
    cs_ptr = th.cuda.default_stream().cuda_stream
    done = False

    # --reset values --
    bufs.inds[...] = -1
    bufs.vals[...] = float("inf")

    # -- smaller batch sizes impact quality --
    for index in range(args.nstreams):

        # -- grab access --
        srch_inds = search_mask.mask2inds(mask,bsize,args.rand_mask)
        if srch_inds.shape[0] == 0:
            done = True
            break

        # -- grab batch --
        vbufs = edict()
        for key in bufs.keys():
            vbufs[key] = view_batch(bufs[key],bsize,index)

        vpatches = edict()
        for key in patches.keys():
            vpatches[key] = view_batch(patches[key],bsize,index)

        # -- exec search --
        search_and_fill(imgs,vpatches,vbufs,srch_inds,flows,args)

        # -- update mask naccess --
        before = mask.sum().item()
        search_mask.update_mask_inds(mask,vbufs.inds,args.c,args.nkeep)
        after = mask.sum().item()

        # -- wait for all streams --
        torch.cuda.synchronize()

    # -- update term. condition --
    done = done or (mask.sum().item() == 0)

    return done

def search_and_fill(imgs,patches,bufs,srch_inds,flows,args):

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
    # srch_img = imgs.noisy if args.step == 0 else imgs.basic
    # srch_img = srch_img if (imgs.clean is None) else imgs.clean

    # -- sim search block --
    bufs.inds[...] = -1
    bufs.vals[...] = float("inf")
    vpss.exec_sim_search_burst(srch_img,srch_inds,bufs.vals,
                               bufs.inds,flows,args.sigma,args)

    # -- fill patches --
    for key in imgs.patch_images:

        # -- skip --
        pass_key = (imgs[key] is None) or (patches[key] is None)
        if pass_key: continue

        # -- fill --
        vpss.fill_patches(patches[key],imgs[key],bufs.inds)
