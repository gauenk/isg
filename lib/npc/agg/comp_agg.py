
# -- python deps --
import torch
import scipy
import numpy as np
from einops import rearrange

# -- numba --
from numba import njit,cuda

# -- package --
import npc.search_mask as imask
from npc.utils import groups2patches
from npc.utils import Timer

# -- local --
from .agg_midpix import agg_patches_midpix
from .agg_kweight import agg_patches_kweight
from .agg_pweight import agg_patches_pweight
from .agg_mixed import agg_patches_mixed

def computeAggregation(deno,group,indices,weights,mask,nSimP,params=None,step=0):

    # # -- create python-params for parser --
    # params,swig_params,_,_ = parse_args(deno,0.,None,params)
    # params = edict({k:v[0] for k,v in params.items()})

    # -- extract info for explicit call --
    ps = params['sizePatch'][step]
    ps_t = params['sizePatchTime'][step]
    onlyFrame = params['onlyFrame'][step]
    aggreBoost =  params['aggreBoost'][step]

    # -- convert groups to patches  --
    t,c,h,w = deno.shape
    nSimP = len(indices)
    patches = groups2patches(group,c,ps,ps_t,nSimP)

    # -- exec search --
    deno_clone = deno.copy()
    nmasked = exec_aggregation(deno,patches,indices,weights,mask,
                               ps,ps_t,onlyFrame,aggreBoost)

    # -- pack results --
    results = {}
    results['deno'] = deno
    results['weights'] = weights
    results['mask'] = mask
    results['nmasked'] = nmasked
    results['psX'] = ps
    results['psT'] = ps_t

    return results

@Timer("agg_patches")
def agg_patches(patches,images,bufs,args,cs_ptr=None,denom="chw"):
    stype = args.agg_type
    if stype == "default":
        agg_patches_default(patches,images,bufs,args,cs_ptr,denom=denom)
    elif stype == "k-weight":
        agg_patches_kweight(patches,images,bufs,args,cs_ptr,denom=denom)
    elif stype == "p-weight":
        agg_patches_pweight(patches,images,bufs,args,cs_ptr,denom=denom)
    elif stype == "midpix":
        agg_patches_midpix(patches,images,bufs,args,cs_ptr,denom=denom)
    elif stype == "mixed":
        agg_patches_mixed(patches,images,bufs,args,cs_ptr,denom=denom)
    else:
        raise ValueError(f"Uknown aggregate type {stype}.")

def agg_patches_default(patches,images,bufs,args,cs_ptr=None,denom="chw"):

    # -- default stream --
    if cs_ptr is None:
        cs_ptr = torch.cuda.default_stream().cuda_stream

    # -- filter by valid --
    valid = torch.nonzero(torch.all(bufs.inds!=-1,1))[:,0]
    vnoisy = patches.noisy[valid]
    vinds = bufs.inds[valid]
    vvals = bufs.vals[valid]

    # -- iterate over "nkeep" --
    if args.nkeep != -1:
        vinds = bufs.inds[:,:args.nkeep]

    # if args.nkeep == 1:
    #     # deno[t1,ci,h1,w1] += gval
    #     # weights[t1,h1,w1] += 1.
    #     scatter_agg(images.deno,vnoisy,vinds,images.weights,
    #                 vvals,images.vals,args.ps,args.ps_t,cs_ptr)
    #     # return compute_agg_batch_single(images.deno,vnoisy,vinds,images.weights,
    #     #                                 vvals,images.vals,args.ps,args.ps_t,cs_ptr)
    # else:
    compute_agg_batch(images.deno,vnoisy,vinds,images.weights,
                      vvals,images.vals,args.ps,args.ps_t,cs_ptr,denom=denom)

# def scatter_agg(deno,patches,inds,weights,vals,ivals,ps,ps_t,cs_ptr):
# conflicts across patches
#     print("deno.shape: ",deno.shape)
#     print("weights.shape: ",weights.shape)
#     print("inds.shape: ",inds.shape)
#     t,c,h,w = deno.shape
#     aug_inds = imask.get_3d_inds(inds,c,h,w)
#     print("aug_inds.shape: ",aug_inds.shape)
#     print("patches.shape: ",patches.shape)
#     # for ci in range(c):
#     #     th.scatter_add(deno,
#     # -- numbify the torch tensors --
#     deno_nba = cuda.as_cuda_array(deno)
#     patches_nba = cuda.as_cuda_array(patches)
#     inds_nba = cuda.as_cuda_array(inds)
#     weights_nba = cuda.as_cuda_array(weights)
#     vals_nba = cuda.as_cuda_array(vals)
#     ivals_nba = cuda.as_cuda_array(ivals)
#     cs_nba = cuda.external_stream(cs_ptr)

#     # -- launch params --
#     nwork = 4
#     bsize,num = inds.shape
#     nblocks = (bsize-1)//(nwork+1)
#     c,ph,pw = patches.shape[-3:]
#     threads = (c,ph,pw)
#     blocks = (nblocks)

#     # -- launch kernel --
#     scatter_kernel[blocks,threads,cs_nba](deno_nba,weights_nba,
#                                           patches_nba,inds_nba,nwork)

# @cuda.jit(max_registers=64)
# def scatter_kernel(deno,weights,patches,inds):

#     # -- shape --
#     nframes,color,height,width = deno.shape
#     chw = color*height*width
#     hw = height*width

#     # -- access with blocks and threads --
#     bidx = cuda.blockIdx.x
#     nidx = cuda.blockIdx.y
#     tidx = cuda.threadIdx.x
#     hidx = cuda.threadIdx.y
#     widx = cuda.threadIdx.z
#     nwork = 4

#     # -> no race condition across "batches [t,h,w]" since "inds" is unique --
#     # -- we want enough work per thread, so we keep the "t" loop --
#     for work in range(nwork):

#         # -- unpack ind --
#         ind = inds[pindex,ti,hi,wi]
#         t0 = ind // chw
#         c0 = (ind % chw) // hw
#         h0 = (ind % hw) // width
#         w0 = ind % width

#         # -- set using patch info --
#         for pt in range(ps_t):
#             for pi in range(ps):
#                 for pj in range(ps):
#                     # for ci in range(color):
#                     #     gval = 1#patches[pindex,ti,pt,ci,pi,pj,hi,wi]
#                     #     deno[t0+pt,ci,h0+pi,w0+pj] += gval
#                     # deno[t0+pt,0,h0+pi,w0+pj] += 1.
#                     # deno[t0+pt,1,h0+pi,w0+pj] += 1.
#                     # deno[t0+pt,2,h0+pi,w0+pj] += 1.

#                     deno[0,0,0,0] += 1.
#                     weights[0,0,0] += 1.

#                     # deno[t0,0,h0,w0] += 1.
#                     # weights[t0,h0,w0] += 1.
#                     # weights[t0+pt,h0+pi,w0+pj] += 1.



def compute_agg_batch(deno,patches,inds,weights,vals,ivals,ps,ps_t,cs_ptr,denom="chw"):

    # -- numbify the torch tensors --
    deno_nba = cuda.as_cuda_array(deno)
    patches_nba = cuda.as_cuda_array(patches)
    inds_nba = cuda.as_cuda_array(inds)
    weights_nba = cuda.as_cuda_array(weights)
    vals_nba = cuda.as_cuda_array(vals)
    ivals_nba = cuda.as_cuda_array(ivals)
    cs_nba = cuda.external_stream(cs_ptr)

    # -- launch params --
    bsize,num = inds.shape
    c,ph,pw = patches.shape[-3:]
    threads = (c,ph,pw)
    blocks = (bsize,num)

    # -- launch kernel --
    # exec_agg_cuda[blocks,threads,cs_nba](deno_nba,patches_nba,inds_nba,weights_nba,
    #                                      vals_nba_,ivals_nba,ps,ps_t)
    exec_agg_simple(deno,patches,inds,weights,vals,ivals,ps,ps_t,denom=denom)


def exec_agg_simple(deno,patches,inds,weights,vals,ivals,ps,ps_t,denom="chw"):

    # -- numbify --
    device = deno.device
    deno_nba = deno.cpu().numpy()
    patches_nba = patches.cpu().numpy()
    inds_nba = inds.cpu().numpy()
    weights_nba = weights.cpu().numpy()
    vals_nba = vals.cpu().numpy()
    ivals_nba = ivals.cpu().numpy()

    # -- exec numba --
    exec_agg_simple_numba(deno_nba,patches_nba,inds_nba,
                          weights_nba,vals_nba,ivals_nba,ps,ps_t,
                          denom=denom)

    # -- back pack --
    deno_nba = torch.FloatTensor(deno_nba).to(device)
    deno[...] = deno_nba
    weights_nba = torch.FloatTensor(weights_nba).to(device)
    weights[...] = weights_nba
    ivals_nba = torch.FloatTensor(ivals_nba).to(device)
    ivals[...] = ivals_nba


@njit
def exec_agg_simple_numba(deno,patches,inds,weights,vals,ivals,ps,ps_t,denom="chw"):

    # -- shape --
    nframes,color,height,width = deno.shape
    chw = color*height*width
    hw = height*width
    bsize,npatches = inds.shape # "npatches" _must_ be from "inds"
    Z = chw if denom == "chw" else hw

    for bi in range(bsize):
        for ni in range(npatches):
            ind = inds[bi,ni]
            if ind == -1: continue
            t0 = ind // Z
            h0 = (ind % hw) // width
            w0 = ind % width

            # print(t0,h0,w0)
            for pt in range(ps_t):
                for pi in range(ps):
                    for pj in range(ps):
                        t1 = (t0+pt)# % nframes
                        h1 = (h0+pi)# % height
                        w1 = (w0+pj)# % width

                        if t1 < 0 or t1 >= nframes: continue
                        if h1 < 0 or h1 >= height: continue
                        if w1 < 0 or w1 >= width: continue

                        for ci in range(color):
                            gval = patches[bi,ni,pt,ci,pi,pj]
                            deno[t1,ci,h1,w1] += gval
                        weights[t1,h1,w1] += 1.
                        # if ni > 0:
                        #     ivals[t0+pt,h0+pi,w0+pj] += vals[bi,ni]


def exec_agg_cuda_launcher(deno,patches,inds,weights,ps,ps_t):
    pass

@cuda.jit(max_registers=64)
def exec_agg_cuda(deno,patches,inds,weights,ps,ps_t):

    # -- shape --
    nframes,color,height,width = deno.shape
    chw = color*height*width
    hw = height*width
    t_bsize = inds.shape[1]

    # -- access with blocks and threads --
    bidx = cuda.blockIdx.x
    nidx = cuda.blockIdx.y
    tidx = cuda.threadIdx.x
    hidx = cuda.threadIdx.y
    widx = cuda.threadIdx.z

    # -> race condition across "batches [t,h,w]"
    # -- we want enough work per thread, so we keep the "t" loop --
    for ti in range(t_bsize):

        # -- unpack ind --
        ind = inds[pindex,ti,hi,wi]
        t0 = ind // chw
        c0 = (ind % chw) // hw
        h0 = (ind % hw) // width
        w0 = ind % width

        # -- set using patch info --
        for pt in range(ps_t):
            for pi in range(ps):
                for pj in range(ps):
                    # for ci in range(color):
                    #     gval = 1#patches[pindex,ti,pt,ci,pi,pj,hi,wi]
                    #     deno[t0+pt,ci,h0+pi,w0+pj] += gval
                    # deno[t0+pt,0,h0+pi,w0+pj] += 1.
                    # deno[t0+pt,1,h0+pi,w0+pj] += 1.
                    # deno[t0+pt,2,h0+pi,w0+pj] += 1.

                    deno[0,0,0,0] += 1.
                    weights[0,0,0] += 1.

                    # deno[t0,0,h0,w0] += 1.
                    # weights[t0,h0,w0] += 1.
                    # weights[t0+pt,h0+pi,w0+pj] += 1.



@njit
def exec_aggregation(deno,patches,indices,weights,mask,
                     ps,ps_t,onlyFrame,aggreBoost):

    # -- def functions --
    def idx2coords(idx,width,height,color):

        # -- get shapes --
        whc = width*height*color
        wh = width*height

        # -- compute coords --
        t = (idx      ) // whc
        c = (idx % whc) // wh
        y = (idx % wh ) // width
        x = idx % width

        return t,c,y,x

    def pixRmColor(ind,c,w,h):
        whc = w*h*c
        wh = w*h
        ind1 = (ind // whc) * wh + ind % wh;
        return ind1

    # -- init --
    nmasked = 0
    t,c,h,w = deno.shape
    nSimP = len(indices)

    # -- update [deno,weights,mask] --
    for n in range(indices.shape[0]):

        # -- get the sim locaion --
        ind = indices[n]
        ind1 = pixRmColor(ind,c,h,w)
        t0,c0,h0,w0 = idx2coords(ind,w,h,c)
        t1,c1,h1,w1 = idx2coords(ind1,w,h,1)

        # -- handle "only frame" case --
        if onlyFrame >= 0 and onlyFrame != t0:
            continue

        # -- set using patch info --
        for pt in range(ps_t):
            for pi in range(ps):
                for pj in range(ps):
                    for ci in range(c):
                        ij = ind + ci*w*h
                        gval = patches[n,pt,ci,pi,pj]
                        deno[t0+pt,ci,h0+pi,w0+pj] += gval
                    weights[t1+pt,h1+pi,w1+pj] += 1

        # -- apply paste trick --
        if (mask[t1,h1,w1] == 1): nmasked += 1
        mask[t1,h1,w1] = False

        if (aggreBoost):
            if ( (h1 > 2*ps) and (mask[t1,h1-1,w1]==1) ): nmasked += 1
            if ( (h1 < (h - 2*ps)) and (mask[t1,h1+1,w1]==1) ): nmasked += 1
            if ( (w1 > 2*ps) and (mask[t1,h1,w1-1]==1) ): nmasked += 1
            if ( (w1 < (w - 2*ps)) and (mask[t1,h1,w1+1]==1) ): nmasked += 1

            if (h1 > 2*ps):  mask[t1,h1-1,w1] = False
            if (h1 < (h - 2*ps)): mask[t1,h1+1,w1] = False
            if (w1 > 2*ps):  mask[t1,h1,w1-1] = False
            if (w1 < (w - 2*ps)): mask[t1,h1,w1+1] = False

    return nmasked

