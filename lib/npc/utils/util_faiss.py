from easydict import EasyDict as edict

def bufs_npc2faiss(npc_patches,npc_bufs):
    bufs = edict()
    bufs.patches = npc_patches.noisy
    bufs.dists = npc_bufs.vals
    bufs.inds = npc_bufs.inds
    return bufs

def args_npc2faiss(npc_args):
    args = edict()
    args.ps = int(npc_args.sizePatch)
    args.pt = int(npc_args.sizePatchTime)
    assert args.pt == 1
    args.ws = int(npc_args.sizeSearchWindow)
    args.wf = int(npc_args.sizeSearchTimeFwd)
    args.wb = int(npc_args.sizeSearchTimeBwd)
    args.bmax = 255.
    args.queryStride = npc_args.stride
    args.k = int(npc_args.nSimilarPatches)
    return args


