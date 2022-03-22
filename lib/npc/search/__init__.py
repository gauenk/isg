from .search import exec_search_eccv2022
from .search_faiss import exec_search_faiss
from .refinement import exec_refinement


# -- searching --
def exec_search(patches,imgs,flows,mask,bufs,args):
    if args.version == "eccv2022":
        return exec_search_eccv2022(patches,imgs,flows,mask,bufs,args)
    elif args.version == "faiss":
        return exec_search_faiss(patches,imgs,flows,mask,bufs,args)
    else:
        raise ValueError(f"Uknown search method [{args.srch_method}]")

