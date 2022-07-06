
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
from .not_only_deno import exec_not_only_deno
from .step import exec_step
from .proc_nl import proc_nl
from .proc_nl_faiss import proc_nl_faiss
from .params import get_args,get_params
from .utils import Timer,optional,optional_rm,compute_sigma_vid
from .eval_cluster import compute_cluster_quality,compute_cluster_quality_faiss
from .eval_deltas import compute_nn_patches

def get_size_grid(sched,niters,kmax):
    if sched == "default":
        kgrid = np.linspace(2,kmax,niters).astype(np.int32)
        return kgrid
    elif sched == "k-weight":
        if kmax == 50: kgrid = [2,5,10,15,25,35,45,50]
        else: raise ValueError(f"Uknow kmax={kmax}")
        # kgrid = np.linspace(2,kmax,niters).astype(np.int)
        return kgrid
    elif sched == "kmax_only":
        assert niters == 1
        return [kmax]
    else:
        raise ValueError(f"Uknown K-Scheduler [{ksched}]")

def get_keep_grid(sched,kgrid):
    niters = len(kgrid)
    if sched in ["default","use_one"]:
        keeps = np.array([1 for _ in range(niters)]).astype(np.int)
        return keeps
    elif sched == "use_two":
        keeps = np.array([2 for _ in range(niters)]).astype(np.int)
        return keeps
    elif sched == "use_log":
        keeps = []
        for i in range(niters):
            logk = int(max(1,np.log(kgrid[i])))
            keeps.append(logk)
        keeps = np.array(keeps).astype(np.int)
        return keeps
    elif sched == "use_same":
        keeps = list(kgrid)
        return keeps
    elif sched == "k-weight":
        # kgrid = np.array([10 for _ in range(niters)]).astype(np.int)
        # kgrid = list(kgrid)
        keeps = np.array([1 for _ in range(niters)]).astype(np.int)
        return keeps
    else:
        raise ValueError(f"Uknown K-Scheduler [{sched}]")

def vnlb_denoise(noisy, sigma, **kwargs):

    # -- timer --
    clock = Timer()
    clock.tic()

    # -- exec our method --
    inpc,pm_errors = exec_npc(noisy, sigma, **kwargs)
    th.cuda.empty_cache()

    # -- run a denoiser --
    eigh_method = optional(kwargs,'eigh_method','faiss')
    flows = optional(kwargs,'flows',None)
    deno,basic,dtime = denoise_npc(noisy,sigma,inpc,flows=flows,eigh_method=eigh_method)
    # deno,basic,dtime = denoise_npc(noisy,sigma,inpc,eigh_method="torch")

    # -- timer --
    clock.toc()
    dtime = clock.diff

    return deno,basic,dtime,inpc,pm_errors

def denoise(noisy, sigma, **kwargs):

    # -- exec our method --
    inpc,pm_errors = exec_npc(noisy, sigma, **kwargs)
    th.cuda.empty_cache()

    # -- run a denoiser --
    dtype = optional(kwargs,"post_deno","bayes")
    deno,basic,dtime = denoise_npc(noisy,sigma,inpc,deno="dtype")
    return deno,basic,dtime,inpc,pm_errors

def get_default_params(sigma,verbose,pm_deno):
    params = get_params(sigma,verbose)
    params['bsize'][0] = 1024
    params['cpatches'][0] = "noisy"
    params['srch_img'][0] = "search"
    params['deno'] = [pm_deno,pm_deno]
    params['sizePatch'] = [5,5]
    params['nstreams'] = [1,1]
    params['offset'][0] = 0.
    params['sizePatchTime'][0] = 1
    # params.sizeSearchTimeBwd[0] = 10
    # params.sizeSearchTimeFwd[0] = 10
    params.sizeSearchTimeBwd[0] = 10
    params.sizeSearchTimeFwd[0] = 10
    params.sizeSearchWindow[0] = 10
    return params

def deno_step(noisy,sigma,search=None,flows=None,
              params=None,kmax=100,pm_deno="bayes",verbose=False,
              pm_method="eccv2022"):

    # -- load [params] --
    if params is None:
        params = get_default_params(sigma,verbose,pm_deno)

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
        if verbose :print("Noisy matching.")
        search = noisy

    # -- create structs --
    args = get_args(params,c,0,noisy.device)
    images = alloc.allocate_images(noisy,None,None,search)

    # -- take step --
    import vnlb
    vnlb.proc_nl(images,flows,args)
    deno = images['deno'].clone()

    return deno


def run_not_only_deno(noisy,clean,sigma,lam_c,ref_sigma,search=None,flows=None,
                      params=None,kmax=100,pm_deno="bayes",verbose=False,
                      pm_method="eccv2022"):
    # -- load [params] --
    if params is None:
        params = get_default_params(sigma,verbose,pm_deno)

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
    deno,psnrs,pme,pme_noisy,pme_clean = exec_not_only_deno(lam_c,ref_sigma,
                                                            images,flows,args)
    deno = deno.cpu().numpy()

    # -- summary --
    print("-"*30)
    print("PSNRS: ",psnrs)
    print("PM Error: ",pme)
    print("PM Error[noisy]: ",pme_noisy)
    print("PM Error[clean]: ",pme_clean)

    return deno,psnrs,pme,pme_noisy,pme_clean


def compute_pm_error(search,clean,sigma,flows=None,
                     params=None,kmax=100,pm_deno="bayes",verbose=False,
                     pm_method="eccv2022"):

    # -- load [params] --
    if params is None:
        params = get_default_params(sigma,verbose,pm_deno)

    # -- load [flows] --
    if flows is None:
        c = int(search.shape[-3])
        shape,device = search.shape,search.device
        flows = alloc.allocate_flows(None,shape,device)

    # -- set params --
    c = int(search.shape[-3])
    params['nSimilarPatches'][0] = kmax
    params['nkeep'][0] = 1
    params['offset'][0] = 0.
    params['srch_img'][0] = "search"

    # -- set [search] --
    # if search is None:
    #     if verbose :print("Oracle matching.")
    #     search = clean

    # -- create structs --
    args = get_args(params,c,0,search.device)
    images = alloc.allocate_images(search,None,clean,search)

    # -- exec --
    args.stride = 1
    if pm_method == "eccv2022":
        args.version = "eccv2022"
        error = compute_cluster_quality(images,flows,args)
    elif pm_method == "faiss":
        args.version = "faiss"
        error = compute_cluster_quality_faiss(images,flows,args)
    else:
        raise ValueError(f"method [{pm_method}]")
    # error = compute_cluster_quality(images,flows,args,pm_method)

    return error

def exec_npc(noisy, sigma, **kwargs):
    version = optional(kwargs,"version","eccv2022")
    # version = optional(kwargs,"version","faiss")
    optional_rm(kwargs,"version")
    if version == "eccv2022":
        return exec_npc_eccv2022(noisy,sigma,**kwargs)
    elif version == "faiss":
        return exec_npc_faiss(noisy,sigma,**kwargs)
    else:
        raise ValueError(f"Uknown search method [{version}]")

def exec_npc_eccv2022(noisy, sigma, niters=3, ksched="default", kmax=50,
                      gpuid=0, clean=None, verbose=True, oracle=False,
                      flows = None, pm_deno="bayes", steps=None,basic=None,
                      full_basic = False,eigh_method="faiss",
                      mix_param=0.,agg_type="default",
                      agg_p_lamb=0.,agg_k_lamb=0.,keep_sched="default",
                      pool_type="default",pool_lamb=1.,cpatches="noisy"):

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
    flows = alloc.allocate_flows(flows,shape,device)

    # -- parameters --
    params = get_params(sigma,verbose)

    # -- set misc --
    params['bsize'][0] = 512
    params['bsize'][0] = 3*1024
    params['cpatches'][0] = cpatches
    # params['srch_img'][0] = "search"
    params['srch_img'][0] = "search"
    # params['srch_img'][0] = "basic"
    params['deno'] = [pm_deno,pm_deno]
    params['offset'][0] = 0.
    params['sizePatch'] = [7,7]
    params['sizePatchTime'][0] = 1
    params['nstreams'] = [1,1]
    params['eigh_method'] = [eigh_method,eigh_method]
    params['agg_type'] = [agg_type,agg_type]
    params['agg_k_lamb'] = [agg_k_lamb,agg_k_lamb]
    params['agg_p_lamb'] = [agg_p_lamb,agg_p_lamb]
    params['pool_type'] = [pool_type,pool_type]
    params['pool_lamb'] = [pool_lamb,pool_lamb]
    params['mix_param'] = [mix_param,mix_param]


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
    if steps is None:
        k_grid = get_size_grid(ksched,niters,kmax)
    else:
        k_grid = steps
        niters = len(steps)
        kmax = steps[-1]
    keep_grid = get_keep_grid(keep_sched,k_grid)

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
    basic = noisy.clone() if basic is None else basic
    if full_basic: basic_l = [basic.clone()]
    for i in range(niters):

        # -- keep grid --
        k = k_grid[i]
        nkeep = keep_grid[i]

        # -- update sigma --
        sigma_p = compute_sigma(noisy,basic,sigma,cpatches,mix_param)

        # -- set params --
        params['nSimilarPatches'][0] = k
        params['nkeep'][0] = nkeep
        params['srch_img'][0] = "search"
        params['cpatches'][0] = "noisy"
        params['sigma'][0] = sigma_p

        # -- parse args --
        args = get_args(params,c,0,noisy.device)

        # -- reallocate images --
        images = alloc.allocate_images(noisy,basic,clean)
        images.search = basic.clone()

        # -- take step --
        proc_nl(images,flows,args)
        basic = images['deno'].clone()
        th.cuda.empty_cache()

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

def compute_sigma(noisy,basic,sigma,cpatches,mix_param):
    if cpatches == "mixed":
        m = mix_param
        interp = m * noisy + (1 - m) * basic
        sigma_p = compute_sigma_vid(interp)
        return sigma_p
    elif cpatches == "noisy":
        return sigma
    elif cpatches == "basic":
        sigma_b = compute_sigma_vid(basic)
        return sigma_b
    else:
        raise ValueError(f"Uknown cpatches: [{cpatches}]")
    pass

def exec_npc_faiss(noisy, sigma, niters=3, ksched="default", kmax=50,
                   gpuid=0, clean=None, verbose=True, oracle=False,
                   flows = None,pm_deno="bayes",steps=None,
                   mix_param=0.,
                   agg_type="default",
                   pool_type="default",
                   pool_lamb=0.,
                   full_basic=False,eigh_method="faiss",cpatches="noisy"):

    # -- to torch --
    if not th.is_tensor(noisy):
        noisy = th.from_numpy(noisy)
        if not(clean is None): clean = th.from_numpy(clean)

    # -- to device --
    device = 'cuda:%d' % gpuid
    noisy = noisy.to(device)
    noisy = noisy.type(th.float32).contiguous()
    if not(clean is None):
        clean = clean.to(device)
        clean = clean.type(th.float32).contiguous()

    # -- allocation --
    c = int(noisy.shape[-3])
    shape,device = noisy.shape,noisy.device
    flows = alloc.allocate_flows(None,shape,device)

    # -- parameters --
    params = get_params(sigma,verbose)

    # -- set misc --
    params['bsize'][0] = int(1024*5)
    # params['bsize'][0] = 4096
    params['cpatches'][0] = cpatches
    # params['srch_img'][0] = "search"
    params['srch_img'][0] = "search"
    # params['srch_img'][0] = "basic"
    params['deno'] = [pm_deno,pm_deno]
    params['offset'][0] = 0.
    params['sizePatchTime'][0] = 1
    params['stride'] = [5,1]
    params['nstreams'] = [1,1]
    params['eigh_method'] = [eigh_method,eigh_method]
    params['verbose'] = [verbose,verbose]
    params['agg_type'] = [agg_type,agg_type]
    params['pool_type'] = [pool_type,pool_type]
    params['mix_param'] = [mix_param,mix_param]
    # params['eigh_method'] = ["faiss","faiss"]
    # params['eigh_method'] = ["torch","torch"]

    # -- burst search space --
    params.sizeSearchWindow[0] = 10
    params.sizeSearchTimeBwd[0] = 10
    params.sizeSearchTimeFwd[0] = 10

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
    if steps is None:
        k_grid = get_size_grid(ksched,niters,kmax)
    else:
        k_grid = steps
        niters = len(steps)
        kmax = steps[-1]
    keep_grid = get_keep_grid(ksched,k_grid)

    # -- exec iters --
    errors = []

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
        print("Starting iteration: ",i)
        proc_nl_faiss(images,flows,args)
        basic = images['deno'].clone()
        # th.cuda.empty_cache()

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


def compute_error(noisy,basic,clean,search,flows,params,kmax,method="eccv2022"):
    c = int(noisy.shape[-3])
    params['nSimilarPatches'][0] = kmax
    params['nkeep'][0] = 1
    params['offset'][0] = 0.
    params['srch_img'][0] = "search"
    params['version'] = ["eccv2022","eccv2022"]
    args = get_args(params,c,0,noisy.device)
    images = alloc.allocate_images(noisy,basic,clean,search)
    if method == "eccv2022":
        error = compute_cluster_quality(images,flows,args)
    elif method == "faiss":
        error = compute_cluster_quality_faiss(images,flows,args)
    else:
        raise ValueError(f"method [{method}]")
    return error
