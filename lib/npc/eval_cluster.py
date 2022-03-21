
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
from npc.utils.patch_utils import yuv2rgb_patches
from npc.utils import idx2coords,coords2idx,patches2groups,groups2patches

# -- project imports --
from npc.utils.gpu_utils import apply_color_xform_cpp,yuv2rgb_cpp
from npc.utils import groups2patches,patches2groups,optional,divUp,save_burst
from npc.utils.video_io import read_nl_sequence
from npc.testing import save_images
from npc.utils.streams import init_streams,wait_streams
import pprint
pp = pprint.PrettyPrinter(indent=4)



def compute_cluster_quality(images,flows,args):

    # -- dont run if no comparison
    if images.clean is None: return -th.ones(1)

    # -- init --
    # pp.pprint(args)

    # -- create access mask --
    mask,ngroups = search_mask.init_mask(images.shape,args)
    mask_r = repeat(mask,'t h w -> t c h w',c=3)

    # -- allocate memory --
    patches = alloc.allocate_patches(args.patch_shape,images.clean,args.device)
    bufs = alloc.allocate_bufs(args.bufs_shape,args.device)

    # -- batching params --
    nelems,nbatches = utils.batch_params(mask,args.bsize,args.nstreams)
    cmasked_prev = nelems

    # -- create error acc --
    error_index = 0
    errors = -th.ones(nelems).to(images.clean.device)

    # -- color xform --
    utils.rgb2yuv_images(images)

    # -- logging --
    if args.verbose: print(f"Computing NPC Cluster Quality")

    # -- over batches --
    for batch in range(nbatches):

        # -- exec search --
        done = search.exec_search(patches,images,flows,mask,bufs,args)

        # -- flat patches --
        update_flat_patch(patches,args)

        # -- valid patches --
        vpatches = get_valid_patches(patches,bufs)
        if vpatches.shape[0] == 0:
            break

        # -- compute error per patch --
        compute_error(errors,vpatches,batch,error_index,args.c,args.pt)
        error_index += len(vpatches)

    # -- synch --
    torch.cuda.synchronize()

    # -- drop -1 terms --
    errors = errors[th.where(errors > -1)]

    return errors

def compute_cluster_quality_at_ind(ind,images,flows,args):

    """

    Compute the cluster quality for a single index of the image

    """

    # -- dont run if no comparison
    if images.clean is None: return -th.ones(1)

    # -- init --
    # pp.pprint(args)

    # -- create access mask --
    mask,ngroups = search_mask.init_mask(images.shape,args)
    mask_r = repeat(mask,'t h w -> t c h w',c=3)

    # -- allocate memory --
    patches = alloc.allocate_patches(args.patch_shape,images.clean,args.device)
    bufs = alloc.allocate_bufs(args.bufs_shape,args.device)

    # -- batching params --
    nelems,nbatches = utils.batch_params(mask,args.bsize,args.nstreams)
    cmasked_prev = nelems

    # -- create error acc --
    error_index = 0
    errors = -th.ones(nelems).to(images.clean.device)

    # -- color xform --
    utils.rgb2yuv_images(images)

    # -- logging --
    if args.verbose: print(f"Computing NPC Cluster Quality")

    # -- over batches --
    for batch in range(nbatches):

        # -- exec search --
        done = search.exec_search(patches,images,flows,mask,bufs,args)

        # -- flat patches --
        update_flat_patch(patches,args)

        # -- valid patches --
        vpatches = get_valid_patches(patches,bufs)
        if vpatches.shape[0] == 0:
            break

        # -- compute error per patch --
        compute_error(errors,vpatches,batch,error_index,args.c,args.pt)
        error_index += len(vpatches)

    # -- synch --
    torch.cuda.synchronize()

    # -- drop -1 terms --
    errors = errors[th.where(errors > -1)]

    return errors


def compute_error(errors,vpatches,batch,start,c,pt):

    # -- unpack --
    clean = vpatches.clean

    # -- compute end --
    stop = clean.shape[0] + start

    # -- yuv -> rgb --
    # clean = yuv2rgb_patches(clean,c=c,pt=pt)/255.
    # delta = (clean[:,[0]] - clean)**2

    # -- compute error (yuv) --
    clean = rearrange(clean,'b n pt c ph pw -> b n c (pt ph pw)')
    clean = clean/255.
    delta = (clean[:,[0],0] - clean[...,0,:])**2

    # -- fill --
    errors[start:stop] = delta.mean((1,2))

def reweight_vals(images):
    nmask_before = images.weights.sum().item()
    index = torch.nonzero(images.weights,as_tuple=True)
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




