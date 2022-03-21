
# -- python deps --
import copy
import numpy as np
import torch
import torch as th
from PIL import Image
from einops import rearrange
from easydict import EasyDict as edict

# -- warnings --
import warnings
from numba import NumbaPerformanceWarning
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

# -- project imports --
from vnlb import denoise_npc
import npc.alloc as alloc
from .step import exec_step
from .proc_nl import proc_nl
from .params import get_args,get_params
from .utils import Timer,optional
from .eval_cluster import compute_cluster_quality
from .eval_deltas import compute_nn_patches

def get_size_grid(sched,niters,kmax):
    if sched == "default":
        kgrid = np.linspace(2,kmax,niters).astype(np.int)
        return kgrid
    else:
        raise ValueError(f"Uknown K-Scheduler [{ksched}]")

def get_keep_grid(sched,niters,kmax):
    if sched == "default":
        kgrid = np.array([1 for _ in range(niters)]).astype(np.int)
        # kgrid[5:] = 10
        return kgrid
    else:
        raise ValueError(f"Uknown K-Scheduler [{ksched}]")

def vnlb_denoise(noisy, sigma, **kwargs):

    # -- exec our method --
    inpc,pm_errors = exec_npc(noisy, sigma, **kwargs)

    # -- run a denoiser --
    deno,basic,dtime = denoise_npc(noisy,sigma,inpc)
    return deno,basic,dtime,inpc,pm_errors

def denoise(noisy, sigma, **kwargs):

    # -- exec our method --
    inpc,pm_errors = exec_npc(noisy, sigma, **kwargs)

    # -- run a denoiser --
    dtype = optional(kwargs,"post_deno","bayes")
    deno,basic,dtime = denoise_npc(noisy,sigma,inpc,deno="dtype")
    return deno,basic,dtime,inpc,pm_errors

def compute_pm_error(noisy,clean,sigma,search=None,flows=None,
                     params=None,kmax=100,pm_deno="bayes",verbose=False):

    # -- load [params] --
    if params is None:
        params = get_params(sigma,verbose)
        params['bsize'][0] = 512
        params['cpatches'][0] = "noisy"
        params['srch_img'][0] = "search"
        params['deno'] = [pm_deno,pm_deno]
        params['offset'][0] = 0.
        params['sizePatchTime'][0] = 1
        params.sizeSearchTimeBwd[0] = 10
        params.sizeSearchTimeFwd[0] = 10
        params.sizeSearchWindow[0] = 10

    # -- load [flows] --
    if flows is None:
        c = int(noisy.shape[-3])
        shape,device = noisy.shape,noisy.device
        flows = alloc.allocate_flows(None,shape,device)

    # -- set params --
    c = int(noisy.shape[-3])
    params['nSimilarPatches'][0] = kmax
    params['nkeep'][0] = 1
    params['offset'][0] = 0.
    params['srch_img'][0] = "search"

    # -- set [search] --
    if search is None:
        if verbose :print("Oracle matching.")
        search = clean

    # -- create structs --
    args = get_args(params,c,0,noisy.device)
    images = alloc.allocate_images(noisy,None,clean,search)

    # -- exec --
    error = compute_cluster_quality(images,flows,args)

    return error

def exec_npc(noisy, sigma, niters=3, ksched="default", kmax=50,
             gpuid=0, clean=None, verbose=True, oracle=False,
             pm_deno="bayes",full_basic=False):

    # -- to torch --
    if not th.is_tensor(noisy):
        noisy = th.from_numpy(noisy)
        if not(clean is None): clean = th.from_numpy(clean)

    # -- to device --
    device = 'cuda:%d' % gpuid
    noisy = noisy.to(device)
    if not(clean is None): clean = clean.to(device)

    # -- allocation --
    c = int(noisy.shape[-3])
    shape,device = noisy.shape,noisy.device
    flows = alloc.allocate_flows(None,shape,device)

    # -- parameters --
    params = get_params(sigma,verbose)

    # -- set misc --
    params['bsize'][0] = 512
    params['cpatches'][0] = "noisy"
    # params['srch_img'][0] = "search"
    params['srch_img'][0] = "search"
    # params['srch_img'][0] = "basic"
    params['deno'] = [pm_deno,pm_deno]
    params['offset'][0] = 0.
    params['sizePatchTime'][0] = 1

    # -- burst search space --
    params.sizeSearchTimeBwd[0] = 10
    params.sizeSearchTimeFwd[0] = 10
    params.sizeSearchWindow[0] = 10

    # -- handle edge case --
    if oracle and not(clean is None):
        errors = compute_error(noisy,None,clean,clean,flows,params,kmax)
        # errors_i = compute_error(noisy,basic,clean,flows,params,kmax)
        return clean,[errors]
    elif oracle and clean is None:
        raise ValueError("We want oracle case but clean image is None.")
    elif oracle is True:
        print("what?")
        exit(0)

    # -- get kschdule --
    k_grid = get_size_grid(ksched,niters,kmax)
    keep_grid = get_keep_grid(ksched,niters,kmax)

    # -- exec iters --
    errors = []

    # -- compute cluster value --
    # params['nSimilarPatches'][0] = kmax
    # params['nkeep'][0] = 1
    # args = get_args(params,c,0,noisy.device)
    # basic = noisy.clone()
    # images = alloc.allocate_images(noisy,basic,clean)
    # images.search = noisy.clone()
    # errors_0 = compute_cluster_quality(images,flows,args)
    # errors_clean = compute_error(noisy,None,clean,clean,flows,params,kmax)
    # print("errors_clean: ",errors_clean.mean().item())

    # -- compute first cluster error --
    errors_0 = compute_error(noisy,None,clean,noisy,flows,params,kmax)
    if verbose:
        print("iter: %d" % -1)
        print("errors_0: ",errors_0.mean().item(),errors_0.std().item())
    errors.append(errors_0)

    # -- loop over iters --
    basic = noisy.clone()
    if full_basic: basic_l = [noisy.clone()]
    for i in range(niters):

        # -- keep grid --
        k = k_grid[i]
        nkeep = keep_grid[i]

        # -- set params --
        params['nSimilarPatches'][0] = k
        params['nkeep'][0] = nkeep
        params['srch_img'][0] = "search"
        params['cpatches'][0] = "noisy"

        # -- parse args --
        args = get_args(params,c,0,noisy.device)

        # -- reallocate images --
        images = alloc.allocate_images(noisy,basic,clean)
        images.search = basic.clone()

        # -- take step --
        proc_nl(images,flows,args)
        basic = images['deno'].clone()

        # -- update new basic image --
        if full_basic: basic_l.append(basic.clone())

        # -- compute cluster value --
        # params['nSimilarPatches'][0] = kmax
        # params['nkeep'][0] = 1
        # args = get_args(params,c,0,noisy.device)
        # images = alloc.allocate_images(noisy,basic,clean)
        # images.search = basic.clone()
        # errors_i = compute_cluster_quality(images,flows,args)
        if not(clean is None):
            errors_i = compute_error(noisy,basic,clean,basic,flows,params,kmax)
            errors.append(errors_i)
        else: errors.append(-th.ones(1))

        # -- info --
        if verbose:
            print("iter: %d" % i)
            print("nSimilarPatches: %d" % k)
            print("nkeep: %d" % nkeep)
            if not(clean is None):
                print("errors_i: ",errors_i.mean().item(),errors_i.std().item())

    # -- create errors --
    k_grid = [-1] + list(k_grid)
    errors = {str(int(k)):e for k,e in zip(k_grid,errors)}

    # -- return all basic frames is "full_basic" is true --
    return_basic = basic
    if full_basic:
        return_basic = basic_l

    return return_basic,errors


"""

Experiment to compute an approximate estimate of

P(Delta_i,Delta_j|Filter,K)

"""
def exec_npc_exp1(noisy, sigma, kgrid, clean, pm_deno,
                  patch_index, nn_inds, kmax=100):

    # -- fixed --
    gpuid = 0
    verbose = True
    oracle = False

    # -- to torch --
    if not th.is_tensor(noisy):
        noisy = th.from_numpy(noisy)
        if not(clean is None): clean = th.from_numpy(clean)
    assert noisy.max().item() > 50, "must be [0,255.]"

    # -- to device --
    device = 'cuda:%d' % gpuid
    noisy = noisy.to(device)
    if not(clean is None): clean = clean.to(device)

    # -- allocation --
    c = int(noisy.shape[-3])
    shape,device = noisy.shape,noisy.device
    flows = alloc.allocate_flows(None,shape,device)

    # -- parameters --
    params = get_params(sigma,verbose)

    # -- set misc --
    params['bsize'][0] = 512
    params['cpatches'][0] = "noisy"
    params['srch_img'][0] = "search"
    params['deno'] = [pm_deno,pm_deno]
    params['offset'][0] = 0.
    params['sizePatchTime'][0] = 1

    # -- burst search space --
    params.sizeSearchTimeBwd[0] = 10
    params.sizeSearchTimeFwd[0] = 10
    params.sizeSearchWindow[0] = 10

    # -- exec iters --
    pm_errors = []
    nn_patches = []
    c_patches = []

    # -- [1st] patch matching errors --
    pm_errors_0 = compute_error(noisy,None,clean,noisy,flows,params,kmax)
    pm_errors.append(pm_errors_0)

    # -- [2nd] patch deltas --
    nn_patches_0,c_patches_0 = get_nn_patches(noisy,noisy,clean,flows,
                                              params,patch_index,nn_inds)
    nn_patches.append(nn_patches_0)
    c_patches.append(c_patches_0)

    # -- loop over iters --
    niters = len(kgrid)
    basic = noisy.clone()
    for i in range(niters):

        # -- keep grid --
        k = kgrid[i]
        nkeep = 1

        # -- set params --
        params['nSimilarPatches'][0] = k
        params['nkeep'][0] = nkeep
        params['srch_img'][0] = "search"
        params['cpatches'][0] = "noisy"

        # -- parse args --
        args = get_args(params,c,0,noisy.device)

        # -- reallocate images --
        images = alloc.allocate_images(noisy,basic,clean)
        images.search = basic.clone()

        # -- take step --
        proc_nl(images,flows,args)
        basic = images['deno'].clone()

        # -- error [patch matching]  --
        pm_errors_i = compute_error(noisy,basic,clean,basic,flows,params,kmax)
        pm_errors.append(pm_errors_i)

        # -- error [patch delta]  --
        nn_patches_i,c_patches_i = get_nn_patches(noisy,basic,clean,flows,
                                                  params,patch_index,nn_inds)
        nn_patches.append(nn_patches_i)
        c_patches.append(c_patches_i)

        # -- info --
        if verbose:

            # -- message --
            print("iter: %d" % i)
            print("nSimilarPatches: %d" % k)
            print("nkeep: %d" % nkeep)

            # -- patch match error --
            a,b = pm_errors_i.mean().item(),pm_errors_i.std().item()
            print("pm_errors_i: ",a,b)

    # -- create errors --
    pm_errors = {str(int(k)):e for k,e in zip(kgrid,pm_errors)}
    nn_patches = {str(int(k)):e for k,e in zip(kgrid,nn_patches)}
    c_patches = {str(int(k)):e for k,e in zip(kgrid,c_patches)}

    return pm_errors,nn_patches,c_patches


def get_nn_patches(noisy,basic,clean,flows,params,patch_index,nn_inds):
    c = int(noisy.shape[-3])
    params['nSimilarPatches'][0] = max(nn_inds)
    params['nkeep'][0] = 1
    params['offset'][0] = 0.
    params['srch_img'][0] = "clean"
    args = get_args(params,c,0,noisy.device)
    images = alloc.allocate_images(noisy,basic,clean,clean)
    nn_patches = compute_nn_patches(images,flows,args,patch_index,nn_inds)
    return nn_patches


def compute_error(noisy,basic,clean,search,flows,params,kmax):
    c = int(noisy.shape[-3])
    params['nSimilarPatches'][0] = kmax
    params['nkeep'][0] = 1
    params['offset'][0] = 0.
    params['srch_img'][0] = "search"
    args = get_args(params,c,0,noisy.device)
    images = alloc.allocate_images(noisy,basic,clean,search)
    error = compute_cluster_quality(images,flows,args)
    return error
