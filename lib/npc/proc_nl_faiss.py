
# -- python deps --
from tqdm import tqdm
import copy,math
import torch
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

import torchvision.utils as tvu

# -- package --
import npc
from npc.testing.file_io import save_images

# -- imports --
import npc.agg as agg
import npc.utils as utils
import npc.alloc as alloc
import npc.search_mask as search_mask
import npc.search as search
import npc.deno as deno
import npc.utils as utils
from npc.utils import update_flat_patch
from npc.utils import idx2coords,coords2idx,patches2groups,groups2patches

# -- project imports --
from npc.utils.gpu_utils import apply_color_xform_cpp,yuv2rgb_cpp
from npc.utils import groups2patches,patches2groups,optional,divUp,save_burst,batch_params_faiss
from npc.utils.video_io import read_nl_sequence
from npc.testing import save_images
from npc.utils.streams import init_streams,wait_streams
import pprint
pp = pprint.PrettyPrinter(indent=4)

def proc_nl_faiss(images,flows,args):

    # -- init --
    # pp.pprint(args)
    args.version = "faiss"

    # -- batching params --
    assert args.nstreams == 1
    nsteps,nbatches = batch_params_faiss(images.shape,args.bsize,args.stride)
    print("nsteps: ",nsteps)

    # -- allocate memory --
    patches = alloc.allocate_patches(args.patch_shape,images.clean,args.device)
    bufs = alloc.allocate_bufs_faiss(args.bufs_shape,args.device)

    # -- color xform --
    utils.rgb2yuv_images(images)

    # -- logging --
    if args.verbose: print(f"Processing NPC [step {args.step}]")

    # -- over batches --
    if args.verbose: pbar = tqdm(total=nbatches)
    for batch in range(nbatches):

        # -- info --
        alloc_bytes = th.cuda.memory_allocated()
        # alloc_bytes = th.cuda.memory_reserved()
        alloc_gb = alloc_bytes# / (1.*1e9)

        # -- exec search --
        done = search.exec_search(batch,patches,images,flows,bufs,args)

        # -- flat patches --
        update_flat_patch(patches,args)

        # -- valid patches --
        vpatches = get_valid_patches(patches,bufs)
        if vpatches.shape[0] == 0:
            break

        # -- denoise patches --
        deno.denoise(vpatches,args,args.deno)

        # -- fill valid --
        fill_valid_patches(vpatches,patches,bufs)

        # -- aggregate patches --
        agg.agg_patches_faiss(batch,patches,images,bufs,args)

        # -- misc --
        th.cuda.empty_cache()

        # -- loop update --
        msg = "[Batch %d/%d]" % (batch+1,nbatches)
        if args.verbose:
            tqdm.write(msg)
            pbar.update(1)

        # - terminate --
        if done: break

    # # -- reweight deno --
    # weights = repeat(images.weights,'t h w -> t c h w',c=args.c)
    # index = torch.nonzero(weights,as_tuple=True)
    # images.deno[index] /= weights[index]

    # # -- fill zeros with basic --
    # fill_img = images.basic if args.step==1 else images.noisy
    # index = torch.nonzero(weights==0,as_tuple=True)
    # images.deno[index] = fill_img[index]

    # -- color xform --
    utils.yuv2rgb_images(images)

    # -- synch --
    th.cuda.synchronize()

def reweight_vals(images):
    nmask_before = images.weights.sum().item()
    index = th.nonzero(images.weights,as_tuple=True)
    images.vals[index] /= images.weights[index]
    irav = images.vals.ravel().cpu().numpy()
    print(np.quantile(irav,[0.1,0.2,0.5,0.8,0.9]))
    # thresh = 0.00014
    thresh = 1e-3
    nz = th.sum(images.vals < thresh).item()
    noupdate = th.nonzero(images.vals > thresh,as_tuple=True)
    images.weights[noupdate] = 0
    th.cuda.synchronize()
    nmask_after = images.weights.sum().item()
    delta_nmask = nmask_before - nmask_after
    print("tozero: [%d/%d]" % (nmask_after,nmask_before))


def fill_valid_patches(vpatches,patches,bufs):
    valid = th.nonzero(th.all(bufs.inds!=-1,1),as_tuple=True)
    for key in patches:
        if (key in patches.tensors) and not(patches[key] is None):
            patches[key][valid] = vpatches[key]

def get_valid_vals(bufs):
    valid = th.nonzero(th.all(bufs.inds!=-1,1),as_tuple=True)
    nv = len(valid[0])
    vals = bufs.vals[valid]
    return vals

def get_valid_patches(patches,bufs):
    valid = th.nonzero(th.all(bufs.inds!=-1,1),as_tuple=True)
    nv = len(valid[0])
    vpatches = edict()
    for key in patches:
        if (key in patches.tensors) and not(patches[key] is None):
            vpatches[key] = patches[key][valid]
        else:
            vpatches[key] = patches[key]
    vpatches.shape[0] = nv
    return vpatches

def proc_nl_cache(vid_set,vid_name,sigma):
    return read_nl_sequence(vid_set,vid_name,sigma)
