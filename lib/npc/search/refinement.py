
# -- python imports --
import torch
import torch as th
import numpy as np
from einops import rearrange,repeat
from easydict import EasyDict as edict

# -- utils --
from npc.utils.batching import view_batch
from npc.utils.logger import vprint
from npc.utils import divUp


def exec_refinement(patches,bufs,sigma):

    # -- compute ave --
    vals = bufs.vals
    ave_vals = th.mean(vals[:,1:]/vals[:,[1]],1)

    # -- remove nan --
    # ave_vals[th.nonzero(th.isnan(ave_vals))] = th.inf
    nonan = ave_vals[th.nonzero(th.isnan(ave_vals)==False)]
    print(np.quantile(nonan.cpu().numpy(),[0.9,0.8,0.5,0.1,0.01]))

    # -- noupdate --
    noupdate = th.nonzero(ave_vals>2.)
    bufs.inds[noupdate] = -1

