from torch import Tensor, tensor, cat

"""
    Given the fact that our beloved library does not have an append method
    (as numpy) I made one for simplicity
"""


def tensorAppend(lst: Tensor, elem: Tensor, retain_dim: bool = False) -> Tensor:
    if not lst.numel():
        return elem if not retain_dim else elem.unsqueeze(dim=0)
    else:
        return th.cat((lst, elem)) if not retain_dim else th.cat((lst, elem.unsqueeze(dim=0)))
