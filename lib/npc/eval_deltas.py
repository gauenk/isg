
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


def compute_nn_patches(images,flows,args,patch_index,nn_inds):


    # -- dont run if no comparison
    if images.clean is None: return -th.ones(1)

    # -- init --
    # pp.pprint(args)

    # -- create access mask --
    mask,ngroups = search_mask.init_mask(images.shape,args)
    mask[...] = 0
    mask.ravel()[:256] = 1
    args.rand_mask = False
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
    nn_patches = []
    c_patches = []
    assert nbatches == 1,"Should be one here."
    for batch in range(nbatches):

        # -- exec search --
        done = search.exec_search(patches,images,flows,mask,bufs,args)

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

        """
        We only fill the 1st step! so we must
        denoise the _entire image_ before
        we reselect top-k "basic" estimate
        patches
        """
        # -- select inds --
        for nn_ind in nn_inds:
            pdeno = vpatches.basic[:,nn_ind-1]
            pclean = vpatches.clean[:,nn_ind-1]
            nn_patches.append(pdeno)
            c_patches.append(pclean)


    nn_patches = th.stack(nn_patches)
    c_patches = th.stack(c_patches)

    return nn_patches,c_patches

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

