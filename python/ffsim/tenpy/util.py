from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfFermionSite

import ffsim


def product_state_as_mps(norb, nelec, idx):
    r"""Return the product state as an MPS.

    Return type:
        `TeNPy MPS <https://tenpy.readthedocs.io/en/latest/reference/tenpy.networks.mps.MPS.html#tenpy.networks.mps.MPS>`__

    Returns:
        The product state as an MPS.
    """

    dim = ffsim.dim(norb, nelec)

    strings = ffsim.addresses_to_strings(
        range(dim), norb=norb, nelec=nelec, bitstring_type=ffsim.BitstringType.STRING
    )

    string = strings[idx]
    up_sector = list(string[0:norb].replace("1", "u"))
    down_sector = list(string[norb : 2 * norb].replace("1", "d"))
    product_state = list(map(lambda x, y: x + y, up_sector, down_sector))

    for i, site in enumerate(product_state):
        if site == "00":
            product_state[i] = "empty"
        elif site == "u0":
            product_state[i] = "up"
        elif site == "0d":
            product_state[i] = "down"
        elif site == "ud":
            product_state[i] = "full"
        else:
            raise ValueError("undefined site")

    # note that the bit positions increase from right to left
    product_state = product_state[::-1]

    shfs = SpinHalfFermionSite(cons_N="N", cons_Sz="Sz")
    psi_mps = MPS.from_product_state([shfs] * norb, product_state)

    return psi_mps
