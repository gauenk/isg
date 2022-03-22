"""

Example use of VNLB for Denoising

"""

import npc
import raft
# import snpc
import torch as th
import numpy as np
import torchvision.utils as tvu
import torch.nn.functional as tnnf
from einops import rearrange,repeat
from easydict import EasyDict as edict
import vnlb
from npc.utils import Timer

# -- set seed [randomly order denoised pixels] --
seed = 123
np.random.seed(seed)
th.manual_seed(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

# -- get data --
# clean = npc.testing.load_dataset("davis",vnlb=False)[0]['clean'].copy()[:20]
# clean = npc.testing.load_dataset("cup_crop",vnlb=False,nframes=30)[0]['clean'].copy()
# clean = npc.testing.load_dataset("candles",vnlb=False,nframes=30)[0]['clean'].copy()
# clean = npc.testing.load_dataset("buildings",vnlb=False,nframes=9)[0]['clean'].copy()
# clean = npc.testing.load_dataset("grass_cloves",vnlb=False,nframes=30)[0]['clean'].copy()
# clean = npc.testing.load_dataset("pinecone_tree_breeze",vnlb=False,nframes=30)[0]['clean'].copy()
# clean = npc.testing.load_dataset("hat_nomotion_stable",vnlb=False,nframes=30)[0]['clean'].copy()
# dname = "hat_wiggle_stable"
# dname = "pinecone_tree_breeze"
# dname = "brickwall"
dname = "motorbike"
# dname = "grass_cloves"
# dname = "pinecone_tree_breeze"
# dname = "text_bus"
clean = npc.testing.load_dataset(dname,vnlb=False,nframes=10,ext="png")[0]['clean']
# clean = npc.testing.load_dataset("brickwall",vnlb=False,nframes=30)[0]['clean'].copy()
# clean = npc.testing.load_dataset("pinecone_brick",vnlb=False,nframes=30)[0]['clean']
# clean = npc.testing.load_dataset("pinecone_brick",vnlb=False,nframes=30)[0]['clean']
# clean = clean[:,:,512+128-32+32:512+256-32,:128-64]
# clean = clean[:,:,200:328,-128-64:-64]
# clean = clean[:,:,264:328,-128-64:-64]
# clean = clean[:,:,256:256+128,256+64:256+128]
# (nframes,channels,height,width)
print("clean.shape: ",clean.shape)
clean = th.from_numpy(clean)/255.
# clean = tnnf.interpolate(clean,scale_factor=.5,mode="bicubic",align_corners=False)
clean *= 255.
clean = clean.numpy()
print("clean.shape: ",clean.shape)
# clean = clean[:,:,32:128+32,128+16:128+128+16] # default
# clean = clean[:,:,32:64+32,128+16:128+64+16]
# clean = clean[:,:,128:256,128:128+64] # default
# clean = clean[:,:,-128:,:128] # default
# clean = clean[:,:,-128:,-128:] # default
# clean = clean[:,:,:128,:128] # default
# clean = clean[:,:,-256:,128:128+256] # default
# clean = clean[:,:,-64:,128:128+64] # default
print("clean.shape: ",clean.shape)
# dname = "davis_catcher"
# dname = "davis_baseball"
# dname = "davis_flat"

# -- Compute Flows --
# ftype = "comp"
# ftype = "load"
ftype = "none"
if ftype == "comp":
    fflow,bflow = raft.burst2flow(clean)
    th.save(fflow,"fflow.pth")
    th.save(bflow,"bflow.pth")
    flows = edict({"fflow":fflow,"bflow":bflow})
elif ftype == "load":
    fflow = th.load("fflow.pth")
    bflow = th.load("bflow.pth")
    flows = edict({"fflow":fflow,"bflow":bflow})
else:
    flows = None

# -- Add Noise --
std = 50.
std_r = 10.
alpha = 20.
noisy = np.random.normal(clean,scale=std)


# -- Save Examples --
path = "output/example/"
nframes = clean.shape[0]
for t in range(nframes):
    npc.utils.save_image(clean[t]/255.,path,"clean_%05d.png" % t)
    npc.utils.save_image(noisy[t]/255.,path,"noisy_%05d.png" % t)
    if t < (nframes-1):
        delta_t = np.abs(clean[t+1] - clean[t])/255.
        delta_t = np.clip(delta_t,0,1.)
        npc.utils.save_image(delta_t,path,"dclean_%05d.png" % t)
    if not(flows is None):
        flo_t = flows.fflow[t].cpu().numpy().transpose(1,2,0)
        flo_t = raft.flow_viz.flow_to_image(flo_t)
        flo_t = flo_t.transpose(2,0,1)/255.
        npc.utils.save_image(flo_t,path,"fflow_%05d.png" % t)

npix = clean.shape[-2]# * clean.shape[-1]

# -- Fake Adj Matrix --
# npix = 20
# adj = np.zeros((npix,npix),dtype=np.float)
# rands = np.random.rand(1)
# rands = np.random.rand(npix,npix) > 0.90
# rands = np.random.rand(npix,npix) > 0.90
# adj = rands * np.tri(*adj.shape)
# np.fill_diagonal(adj,1.)
# adj = adj + adj.T - np.diag(np.diag(adj))
# print(adj)
# adj = repeat(adj,'h w -> c h w',c=3)
# adj = th.from_numpy(adj)[None,:]
# adj = tnnf.interpolate(adj,size=(200,200),mode="nearest")#,align_corners=False)
# adj = adj.numpy()[0]
# npc.utils.save_image(adj,path,"adj.png")

# exit(0)

# adj = np.zeros((nframes,nframes))
# cclean = clean/255.
# for i in range(nframes):
#     for j in range(nframes):
#         delta = np.mean( np.abs(cclean[i] - cclean[j]) ).item()
#         adj[i,j] = delta
#         print("(%d,%d): %2.3e" % (i,j,delta))
# adj = repeat(adj,'h w -> c h w',c=3)
# npc.utils.save_image(adj,path,"adj.png")
# print(adj)

# hs = 32
# ws = 32
# ps = 7
# clean_cc = th.from_numpy(clean[:,:,hs:hs+ps,ws:ws+ps])/255.
# print("clean_cc.shape: ",clean_cc.shape)
# clean_cc_grid = tvu.make_grid(clean_cc,nrow=t//4)
# npc.utils.save_image(clean_cc_grid,path,"clean_cc.png")



# noisy = np.random.normal(clean,scale=std) + clean*np.random.normal(clean,scale=1.)
# noisy = np.random.poisson(alpha*clean/255.)*255./alpha
# std = (noisy-clean).std()
print("std: ",std)
# print(np.c_[clean.ravel(),noisy.ravel()])
# print(np.mean((noisy - clean)**2))
# exit(0)
th.save(clean,"clean.pth")
th.save(noisy,"noisy.pth")

# exit(0)


# 31.415 = standard deno
# .75 -> 2
# .75 -> 10
# .5 -> 2

# -- Video Non-Local Bayes --

sigma = std
# methods = ["npc","oracle","oracle_og","vnlb"]
# methods = ["npc-nlm","npc"]
# methods = ["npc","vnlb"]
# methods = ["npc-eccv2022"]
methods = ["npc"]
for method in methods:
    clock = Timer()
    clock.tic()
    if method == "oracle":
        output = npc.vnlb_denoise(noisy,sigma,niters=3,clean=clean,kmax=50,
                                  oracle=True,verbose=True)
        deno,basic,dtime,inpc,pm_errors = output
    elif method == "npc":
        output = npc.vnlb_denoise(noisy,sigma,niters=3,clean=None,kmax=50,
                                  oracle=False,verbose=True,version="faiss")
        deno,basic,dtime,inpc,pm_errors = output
    elif method == "npc-eccv2022":
        output = npc.vnlb_denoise(noisy,sigma,niters=3,clean=None,kmax=50,
                                  oracle=False,verbose=True,version="eccv2022")
        deno,basic,dtime,inpc,pm_errors = output
    elif method == "npc-wnnm":
        output = npc.denoise(noisy,sigma,niters=3,clean=clean,kmax=50,
                             oracle=False,verbose=True,pm_deno="wnnm")
        deno,basic,dtime,inpc,pm_errors = output
    elif method == "npc-nlm":
        output = npc.vnlb_denoise(noisy,sigma,niters=3,clean=clean,kmax=50,
                                  oracle=False,verbose=True,pm_deno="nlm")
        deno,basic,dtime,inpc,pm_errors = output
    elif method == "oracle_og":
        deno,basic,dtime = vnlb.denoise(noisy, sigma, None, 0, clean, True)
        pm_errors = {"0":th.zeros(0)}
    elif method == "vnlb":
        deno,basic,dtime = vnlb.denoise(noisy, sigma, None, 0, None, True)
        pm_errors = {"0":th.zeros(0)}
    else:
        print("huh?")
        exit(0)
    clock.toc()
    dtime = clock.diff

    # -- Message --
    print(f"method: [{method}]")

    # -- Denoising Quality --
    noisy_psnrs = npc.utils.compute_psnrs(clean,noisy)
    basic_psnrs = npc.utils.compute_psnrs(clean,basic)
    deno_psnrs = npc.utils.compute_psnrs(clean,deno)

    print("Denoised PSNRs:")
    print(deno_psnrs,deno_psnrs.mean())
    print("Basic PSNRs:")
    print(basic_psnrs,basic_psnrs.mean())
    print("Noisy PSNRs:")
    print(noisy_psnrs)
    print("PM Error:")
    order = sorted(pm_errors.keys())
    print([pm_errors[o].mean().item() for o in order])
    print([pm_errors[o].std().item() for o in order])
    print("Exec Time (sec): %2.2e" % dtime)


    # -- Save Examples --
    path = f"output/example/{dname}/{method}/{int(std)}/"
    nframes = deno.shape[0]
    for t in range(nframes):
        npc.utils.save_image(deno[t]/255.,path,"deno_%05d.png" % t)
        npc.utils.save_image(noisy[t]/255.,path,"noisy_%05d.png" % t)
        npc.utils.save_image(basic[t]/255.,path,"basic_%05d.png" % t)
