from torch import Tensor, cat

"""
    Given the fact that our beloved library does not have an append method
    (as numpy) I made one for simplicity
"""


def tensor_append(lst: Tensor, elem: Tensor, retain_dim: bool = False) -> Tensor:
    if not lst.numel():
        return elem if not retain_dim else elem.unsqueeze(dim=0)
    else:
        return cat((lst, elem)) if not retain_dim else cat((lst, elem.unsqueeze(dim=0)))
