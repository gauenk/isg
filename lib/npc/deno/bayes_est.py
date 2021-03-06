
# -- misc --
import torch,math

# -- this package --
import npc

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- extra linalg --
import scipy
from scipy import linalg as scipy_linalg

# -- fast eigh --
from faiss.contrib import kn3

# from .cov_mat import computeCovMat
from npc.utils import groups2patches,patches2groups
from npc.utils.gpu_utils import apply_yuv2rgb
# from npc.gpu.patch_subset import exec_patch_subset
# from npc.gpu.patch_utils import yuv2rgb_patches,patches_psnrs


def denoise(patches,args):

    # -- copy original patches --
    pnoisy = patches.noisy
    pbasic = patches.basic.clone()
    bsize = pnoisy.shape[0]

    # -- flatten dims --
    flat_pdim(patches)

    # -- center patches --
    pinput = get_image(patches.noisy,patches.basic,args.cpatches,args.mix_param)
    cnoisy,cbasic = center_patches(patches.noisy,patches.basic,pinput,
                                   patches.flat,args.step==1,args.c,
                                   args.pool_type,args.pool_lamb)

    # -- flatten across batch --
    flat_bdim(patches)

    # -- cov --
    pinput = get_image(patches.noisy,patches.basic,args.cpatches,args.mix_param)
    covMat,eigVals,eigVecs = compute_cov_mat(pinput,args.rank,args.eigh_method)

    # -- eigen values --
    eigVals_rs = rearrange(eigVals,'(b c) p -> b c p',b=bsize)
    rank_var = torch.mean(torch.sum(eigVals_rs,dim=2),dim=1)
    denoise_eigvals(eigVals_rs,args.sigmab2,args.mod_sel,args.rank)
    bayes_filter_coeff(eigVals,args.sigma2,args.thresh)

    # -- filter --
    filter_patches(patches.noisy,covMat,eigVals,eigVecs,args.sigma2,args.rank)

    # -- expand batch and color --
    expand_bdim(patches,args)

    # -- re-center --
    patches.noisy[...] += cnoisy
    if args.step == 1: patches.basic[...] += cbasic

    # -- reshape --
    expand_pdim(patches,args)

    # -- fill? --
    # pnoisy[...] = pbasic[...]#patches.basic[...]
    # pnoisy[...] = patches.noisy[...]
    # patches.noisy[...] = patches.basic[...]#pbasic[...]

    return rank_var


def get_image(cnoisy,cbasic,cpatches,mix_param):
    if cpatches == "mix":
        a = mix_param
        return a * cnoisy + (1 - a) * cbasic
    elif cpatches == "noisy":
        return cnoisy
    elif cpatches == "basic":
        return cbasic
    else:
        raise ValueError("Uknown cpatches.")

def flat_bdim(patches):
    shape_str = "b c n p -> (b c) n p"
    reshape_patches(patches,shape_str)

def flat_pdim(patches):
    shape_str = "b n pt c ph pw -> b c n (pt ph pw)"
    reshape_patches(patches,shape_str)

def expand_pdim(patches,args):
    shape_str = "b c n (pt ph pw) -> b n pt c ph pw"
    shape_key = {'ph':args.ps,'pw':args.ps}
    reshape_patches(patches,shape_str,**shape_key)

def expand_bdim(patches,args):
    shape_str = "(b c) n p -> b c n p"
    shape_key = {'c':args.c}
    reshape_patches(patches,shape_str,**shape_key)

def reshape_patches(patches,shape_str,**shape_key):
    for img in patches.images:
        if patches[img] is None: continue
        patches[img] = rearrange(patches[img],shape_str,**shape_key)

def center_patches(pnoisy,pbasic,pinput,flat_patch,step2,c,pool_type,pool_lamb):

    # -- center basic --
    cbasic = None
    if step2: # "step2" does not exist in "npc"
        cbasic = pool_samples(pbasic,pbasic,pool_type,pool_lamb)
        pbasic[...] -= cbasic

    # -- choose center for noisy --
    cnoisy = pool_samples(pnoisy,pinput,pool_type,pool_lamb)
    if step2:
        flat_inds = torch.where(flat_patch == 1)[0]
        cnoisy[flat_inds] = cbasic[flat_inds]

    # -- center noisy --
    pnoisy[...] -= cnoisy

    return cnoisy,cbasic

def pool_samples(pnoisy,pbasic,pname,plamb):
    if pname == "default":
        return pnoisy.mean(dim=2,keepdim=True)
    elif pname == "weighted":
        dist = th.mean(((pbasic[:,:,[0],:]-pbasic)/255.)**2,-1,True)
        weight = th.exp(-plamb * dist)
        weight /= th.sum(weight,-2,True)
        wpatches = weight * pnoisy
        mp = wpatches.sum(dim=2,keepdim=True)
        return mp
    else:
        raise ValueError(f"Uknown pooling type [{pname}]")

def compute_cov_mat(patches,rank,eigh_method):

    with torch.no_grad():

        # -- cov mat --
        bsize,num,pdim = patches.shape
        covMat = torch.matmul(patches.transpose(2,1),patches)
        covMat /= num

        # -- eigen stuff --
        if eigh_method == "torch":
            eigVals,eigVecs = torch.linalg.eigh(covMat)
            eigVals= torch.flip(eigVals,dims=(1,))
            eigVecs = torch.flip(eigVecs,dims=(2,))[...,:rank]
        elif eigh_method == "faiss":
            print(covMat.shape)
            print(covMat.ravel()[:3])
            print(covMat.ravel().max(),covMat.ravel().min())
            eigVals,eigVecs = kn3.tiny_eigh(covMat.clone())
            eigVecs = eigVecs[...,:rank]
        else:
            raise ValueError(f"Uknown eigh method [{eigh_method}]")

    return covMat,eigVals,eigVecs


def denoise_eigvals(eigVals,sigmab2,mod_sel,rank):
    if mod_sel == "clipped":
        th_sigmab2 = torch.FloatTensor([sigmab2]).reshape(1,1,1)
        th_sigmab2 = th_sigmab2.to(eigVals.device)
        emin = torch.min(eigVals[...,:rank],th_sigmab2)
        eigVals[...,:rank] -= emin
    elif mod_sel == "paul":
        pass
    else:
        raise ValueError(f"Uknown eigen-stuff modifier: [{mod_sel}]")

def bayes_filter_coeff(eigVals,sigma2,thresh):
    geq = torch.where(eigVals > (thresh*sigma2))
    leq = torch.where(eigVals <= (thresh*sigma2))
    eigVals[geq] = 1. / (1. + sigma2 / eigVals[geq])
    eigVals[leq] = 0.

def filter_patches(patches,covMat,eigVals,eigVecs,sigma2,rank):
    bsize = patches.shape[0]
    Z = torch.matmul(patches,eigVecs)
    R = eigVecs * eigVals[:,None,:rank]
    tmp = torch.matmul(Z,R.transpose(2,1))
    patches[...] = tmp

