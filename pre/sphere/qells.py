# Contains functions to compute or Q_L(s) and Q_L(k) analytically for the sphere

import numpy as np
from scipy import special
from mpmath import meijerg, gammainc, re

## Analytical Q_L(s) for the sphere (details in notebook)

# start by defining Legendre polynomials indefinite integrals (the F_L)


def prim_L_0(a):
    return a


def prim_L_1(a):
    return 0.5 * a**2


def prim_L_2(a):
    return 0.5 * a**3 - 0.5 * a


def prim_L_3(a):
    return (5.0 / 8.0) * a**4 - (3.0 / 4.0) * a**2


def prim_L_4(a):
    return ((35.0 / 5.0) * a**5 - (30.0 / 3.0) * a**3 + 3.0 * a) / 8.0


# Then get the big indefinite integral for the first term (prim_gen_L(s) = \int_{1-s}^1 ...)


def prim_gen1_0(s):
    return s / 2.0 - s**2 + 3.0 * s**3 / 8.0


def prim_gen1_1(s):
    return s / 6.0 - 3.0 * s**2 / 8.0 + 7.0 * s**3 / 30.0


def prim_gen1_2(s):
    return (
        1.0 / 12.0 * (-2.0 + s) * s * (-6.0 + s * (-6.0 + s * (13.0 + s * (2 + 7 * s))))
        - (-1 + s**2) ** 3 * np.log(1 - s)
    ) / (-16 * s**3)


def prim_gen1_3(s):
    return -s * (560.0 - 1120.0 * s + 528.0 * s**2 + 175.0 * s**3) / 4480.0


def prim_gen1_4(s):
    return (
        s
        * (
            840.0
            + 420.0 * s
            - 2120.0 * s**2
            - 990.0 * s**3
            + 1528.0 * s**4
            + 748.0 * s**5
            - 120.0 * s**6
            - 539.0 * s**7
            + 84.0 * s**9
        )
        - 120.0 * (-1 + s**2) ** 3 * (7 + s**2) * np.log(1 - s)
    ) / (6144.0 * s**5)


# Get the big indefinite integral for the first term (prim_gen1_L(s) = \int_{s-1}^1 ...)


def prim_gen2_0(s):
    return s / 2.0 - s**2 + 3.0 * s**3 / 8.0


def prim_gen2_1(s):
    return (
        -1.0 / 3.0
        + 2.0 / (15.0 * s**2)
        - s / 6.0
        + 5.0 * s**2 / 8.0
        - 7.0 * s**3 / 30.0
    )


def prim_gen2_2(s):
    return -(
        (
            1.0
            / 12.0
            * (-2.0 + s)
            * s
            * (-6.0 + s * (-6.0 + s * (13.0 + s * (2.0 + 7.0 * s))))
            - (-1.0 + s**2) ** 3 * np.log(-1.0 + s)
        )
        / (16.0 * s**3)
    )


def prim_gen2_3(s):
    return (
        -3.0 / 4.0
        - 2.0 / (7.0 * s**4)
        + 4.0 / (5.0 * s**2)
        + s / 8.0
        + 33.0 * s**3 / 280.0
        - 5.0 * s**4 / 128.0
    )


def prim_gen2_4(s):
    return (
        s
        * (
            840.0
            + 420.0 * s
            - 2120.0 * s**2
            - 990.0 * s**3
            + 1528.0 * s**4
            + 748.0 * s**5
            - 120.0 * s**6
            - 539.0 * s**7
            + 84.0 * s**9
        )
        - 120.0 * (-1.0 + s**2) ** 3 * (7.0 + s**2) * np.log(-1.0 + s)
    ) / (6144.0 * s**5)


# Define a call dictionnay for easy calling of the right function without testing the value of L

call_dict_L = {
    "0": prim_L_0,
    "1": prim_L_1,
    "2": prim_L_2,
    "3": prim_L_3,
    "4": prim_L_4,
}
call_dict_gen1 = {
    "0": prim_gen1_0,
    "1": prim_gen1_1,
    "2": prim_gen1_2,
    "3": prim_gen1_3,
    "4": prim_gen1_4,
}
call_dict_gen2 = {
    "0": prim_gen2_0,
    "1": prim_gen2_1,
    "2": prim_gen2_2,
    "3": prim_gen2_3,
    "4": prim_gen2_4,
}


def qell_s_spherical(L, s, R):
    """
    Return Q_L(s) for a spherical window function of radius R.
    """
    assert L in [0, 1, 2, 3, 4], "Can only compute Q_L up to L = 4"
    assert R > 0, "Must give a non-zero positive radius!"

    # normalize s
    S = s / R

    prefactor = 3 / 2 * (2 * L + 1)  # normalization so that Q_L(0) = 1

    ell = str(L)

    if S >= 2:
        return 0
    elif 0 <= S < 1:
        return prefactor * (
            call_dict_gen1[ell](S)
            - call_dict_L[ell](-1.0) / 3.0
            + call_dict_L[ell](1.0) * (1.0 - S) ** 3 / 3.0
        )
    elif 1 <= S < 2:
        return prefactor * (
            call_dict_gen2[ell](S)
            - call_dict_L[ell](-1.0) * (1.0 - (S - 1.0) ** 3) / 3.0
        )

    return "Oops, something went wrong (no condition on s matched)"

def qell_s_spherical_vectorized(L, s, R):
    """
    Return Q_L(s) for a spherical window function of radius R.
    """
    assert L in [0, 1, 2, 3, 4], "Can only compute Q_L up to L = 4"
    assert R > 0, "Must give a non-zero positive radius!"
    s = np.atleast_1d(s)
    S = s / R   # Normalize s
    prefactor = 3 / 2 * (2 * L + 1)  # normalization so that Q_L(0) = 1
    ell = str(L) # char L to call for the dictionnary

    # divide s values into < R, < 2R, > 2R
    sbins = np.array([1.0, 2.0])
    bin_idx = np.digitize(S, sbins)
    mask0 = (bin_idx == 0)
    mask1 = (bin_idx == 1)
    mask2 = (bin_idx == 2)

    # prepare result array
    Qls = np.empty_like(s)
    Qls[mask0] = prefactor * (
            call_dict_gen1[ell](S[mask0])
            - call_dict_L[ell](-1.0) / 3.0
            + call_dict_L[ell](1.0) * (1.0 - S[mask0]) ** 3 / 3.0
        )
    Qls[mask1] = prefactor * (
        call_dict_gen2[ell](S[mask1])
            - call_dict_L[ell](-1.0) * (1.0 - (S[mask1] - 1.0) ** 3) / 3.0
        )
    Qls[mask2] = 0.0

    return Qls


## Analytical Q_L(k) for the sphere (details in notebook)


def qellk_0(k):
    termA = -(
        (
            np.pi
            * (
                -24 * (1 + k**2)
                + (24 + 12 * k**2 + 5 * k**4) * np.cos(k)
                + 4 * k * (6 + k**2) * np.sin(k)
            )
        )
        / (6 * k**6)
    )
    termB = (
        np.pi
        * (
            (24 + 12 * k**2 + 5 * k**4) * np.cos(k)
            + 24 * (-1 + k**2) * np.cos(2 * k)
            + 4 * k * (6 + k**2 - 24 * np.cos(k)) * np.sin(k)
        )
    ) / (6 * k**6)
    return termA + termB


def qellk_2_unvectorized(k):
    termA = (1 / (48 * k**6)) * (
        np.pi
        * (
            768
            + 192 * k**2
            - 3 * k**5 * np.pi
            - 768 * np.cos(k)
            + 576 * np.euler_gamma * np.cos(k)
            + 48 * k**2 * np.cos(k)
            - 4 * k**4 * np.cos(k)
            + 6 * 1j * k**5 * gammainc(-2, (-1j) * k)
            - 6 * 1j * k**5 * gammainc(-2, 1j * k)
            - 6 * k**4 * gammainc(-1, (-1j) * k)
            + 6 * 1j * k**5 * gammainc(-1, (-1j) * k)
            - 6 * k**4 * gammainc(-1, 1j * k)
            - 6 * 1j * k**5 * gammainc(-1, 1j * k)
            + 576 * np.cos(k) * np.log(k)
            - 768 * k * np.sin(k)
            + 576 * np.euler_gamma * k * np.sin(k)
            + 46 * k**3 * np.sin(k)
            + 576 * k * np.log(k) * np.sin(k)
            + 12 * special.sici(k)[1] * (k**4 - 48 * np.cos(k) - 48 * k * np.sin(k))
            + 6 * (k**5 + 96 * k * np.cos(k) - 96 * np.sin(k)) * special.sici(k)[0]
        )
    )
    termB = (1 / (48 * k**6)) * (
        (
            np.pi
            * (
                -384
                + 384 * np.exp(1j * k)
                + 384 * np.exp(3 * 1j * k)
                - 384 * np.exp(4 * 1j * k)
                - 288 * np.exp(1j * k) * np.euler_gamma
                - 288 * np.exp(3 * 1j * k) * np.euler_gamma
                - 480 * 1j * k
                + 384 * 1j * np.exp(1j * k) * k
                - 384 * 1j * np.exp(3 * 1j * k) * k
                + 480 * 1j * np.exp(4 * 1j * k) * k
                - 288 * 1j * np.exp(1j * k) * np.euler_gamma * k
                + 288 * 1j * np.exp(3 * 1j * k) * np.euler_gamma * k
                + 204 * k**2
                - 24 * np.exp(1j * k) * k**2
                - 24 * np.exp(3 * 1j * k) * k**2
                + 204 * np.exp(4 * 1j * k) * k**2
                + 54 * 1j * k**3
                - 32 * 1j * np.exp(1j * k) * k**3
                + 32 * 1j * np.exp(3 * 1j * k) * k**3
                - 54 * 1j * np.exp(4 * 1j * k) * k**3
                + 5 * np.exp(1j * k) * k**4
                + 5 * np.exp(3 * 1j * k) * k**4
                - 72 * 1j * np.exp(1j * k) * k**2 * np.pi
                + 72 * 1j * np.exp(3 * 1j * k) * k**2 * np.pi
                - 3 * np.exp(1j * k) * k**3 * np.pi
                + 93 * np.exp(3 * 1j * k) * k**3 * np.pi
                - 3 * 1j * np.exp(1j * k) * k**4 * np.pi
                - 21 * 1j * np.exp(3 * 1j * k) * k**4 * np.pi
                + 12
                * np.exp(1j * k)
                * k**3
                * (-4 * 1j + k + np.exp(2 * 1j * k) * (4 * 1j + k))
                * special.sici(-k)[1]
                + 6
                * np.exp(1j * k)
                * (
                    48
                    + 48 * 1j * k
                    - 24 * k**2
                    - 7 * 1j * k**3
                    + k**4
                    + np.exp(2 * 1j * k)
                    * (48 - 48 * 1j * k - 24 * k**2 + 7 * 1j * k**3 + k**4)
                )
                * special.sici(k)[1]
                - 288 * np.exp(1j * k) * np.log(k)
                - 288 * np.exp(3 * 1j * k) * np.log(k)
                - 288 * 1j * np.exp(1j * k) * k * np.log(k)
                + 288 * 1j * np.exp(3 * 1j * k) * k * np.log(k)
                - 54
                * np.exp(3 * 1j * k)
                * k**2
                * meijerg([[], [1]], [[0, 0], []], (-1j) * k)
                + 54
                * 1j
                * np.exp(3 * 1j * k)
                * k**3
                * meijerg([[], [1]], [[0, 0], []], (-1j) * k)
                + 18
                * np.exp(3 * 1j * k)
                * k**4
                * meijerg([[], [1]], [[0, 0], []], (-1j) * k)
                - 54 * np.exp(1j * k) * k**2 * meijerg([[], [1]], [[0, 0], []], 1j * k)
                - 54
                * 1j
                * np.exp(1j * k)
                * k**3
                * meijerg([[], [1]], [[0, 0], []], 1j * k)
                + 18 * np.exp(1j * k) * k**4 * meijerg([[], [1]], [[0, 0], []], 1j * k)
                - 54
                * np.exp(3 * 1j * k)
                * k**2
                * meijerg([[], [1, 1]], [[0, 0, 2], []], (-1j) * k)
                + 36
                * 1j
                * np.exp(3 * 1j * k)
                * k**3
                * meijerg([[], [1, 1]], [[0, 0, 2], []], (-1j) * k)
                - 54
                * np.exp(1j * k)
                * k**2
                * meijerg([[], [1, 1]], [[0, 0, 2], []], 1j * k)
                - 36
                * 1j
                * np.exp(1j * k)
                * k**3
                * meijerg([[], [1, 1]], [[0, 0, 2], []], 1j * k)
                - 18
                * np.exp(3 * 1j * k)
                * k**2
                * meijerg([[], [1, 1]], [[0, 0, 3], []], (-1j) * k)
                - 18
                * np.exp(1j * k)
                * k**2
                * meijerg([[], [1, 1]], [[0, 0, 3], []], 1j * k)
                - 288 * 1j * np.exp(1j * k) * special.sici(k)[0]
                + 288 * 1j * np.exp(3 * 1j * k) * special.sici(k)[0]
                + 288 * np.exp(1j * k) * k * special.sici(k)[0]
                + 288 * np.exp(3 * 1j * k) * k * special.sici(k)[0]
                + 144 * 1j * np.exp(1j * k) * k**2 * special.sici(k)[0]
                - 144 * 1j * np.exp(3 * 1j * k) * k**2 * special.sici(k)[0]
                - 90 * np.exp(1j * k) * k**3 * special.sici(k)[0]
                - 90 * np.exp(3 * 1j * k) * k**3 * special.sici(k)[0]
                - 18 * 1j * np.exp(1j * k) * k**4 * special.sici(k)[0]
                + 18 * 1j * np.exp(3 * 1j * k) * k**4 * special.sici(k)[0]
            )
        )
        / np.exp(2 * 1j * k)
    )
    return termA + termB


def qellk_4(k):
    termA = (
        (1 / ((3072 * k**8) * np.exp(1j * k)))
        * np.pi
        * (
            92160
            * np.exp(1j * k)
            * k**2
            * special.sici(k)[1]
            * (k * np.sin(k) + np.cos(k))
            + 149 * np.exp(2 * 1j * k) * k**6
            + 149 * k**6
            + 1718 * 1j * np.exp(2 * 1j * k) * k**5
            - 1718 * 1j * k**5
            + 12288 * np.exp(1j * k) * k**4
            - 2307 * np.exp(2 * 1j * k) * k**4
            - 2307 * k**4
            - 94851 * 1j * np.exp(2 * 1j * k) * k**3
            + 46080 * np.euler_gamma * 1j * np.exp(2 * 1j * k) * k**3
            + 94851 * 1j * k**3
            - 46080 * 1j * np.euler_gamma * k**3
            + 46080 * 1j * np.exp(2 * 1j * k) * k**3 * np.log(k)
            - 46080 * 1j * k**3 * np.log(k)
            - 92160
            * np.exp(1j * k)
            * k**2
            * special.sici(k)[0]
            * (k * np.cos(k) - np.sin(k))
            - 509952 * np.exp(1j * k) * k**2
            + 93696 * np.exp(2 * 1j * k) * k**2
            - 46080 * np.euler_gamma * np.exp(2 * 1j * k) * k**2
            - 46080 * np.euler_gamma * k**2
            + 93696 * k**2
            - 46080 * np.exp(2 * 1j * k) * k**2 * np.log(k)
            - 46080 * k**2 * np.log(k)
            - 322560 * 1j * np.exp(2 * 1j * k) * k
            + 322560 * 1j * k
            - 645120 * np.exp(1j * k)
            + 322560 * np.exp(2 * 1j * k)
            + 322560
        )
    )
    termB = (1 / (3072 * k**8)) * (
        (
            np.pi
            * (
                322560
                - 322560 * np.exp(1j * k)
                - 322560 * np.exp(3 * 1j * k)
                + 322560 * np.exp(4 * 1j * k)
                + 645120 * 1j * k
                - 322560 * 1j * np.exp(1j * k) * k
                + 322560 * 1j * np.exp(3 * 1j * k) * k
                - 645120 * 1j * np.exp(4 * 1j * k) * k
                - 390144 * k**2
                - 93696 * np.exp(1j * k) * k**2
                - 93696 * np.exp(3 * 1j * k) * k**2
                - 390144 * np.exp(4 * 1j * k) * k**2
                + 46080 * np.exp(1j * k) * np.euler_gamma * k**2
                + 46080 * np.exp(3 * 1j * k) * np.euler_gamma * k**2
                - 73728 * 1j * k**3
                - 94851 * 1j * np.exp(1j * k) * k**3
                + 94851 * 1j * np.exp(3 * 1j * k) * k**3
                + 73728 * 1j * np.exp(4 * 1j * k) * k**3
                + 46080 * 1j * np.exp(1j * k) * np.euler_gamma * k**3
                - 46080 * 1j * np.exp(3 * 1j * k) * np.euler_gamma * k**3
                + 6144 * k**4
                + 2307 * np.exp(1j * k) * k**4
                + 2307 * np.exp(3 * 1j * k) * k**4
                + 6144 * np.exp(4 * 1j * k) * k**4
                + 1718 * 1j * np.exp(1j * k) * k**5
                - 1718 * 1j * np.exp(3 * 1j * k) * k**5
                - 149 * np.exp(1j * k) * k**6
                - 149 * np.exp(3 * 1j * k) * k**6
                - 12600 * np.exp(1j * k) * k**3 * np.pi
                + 12600 * np.exp(3 * 1j * k) * k**3 * np.pi
                + 10440 * 1j * np.exp(1j * k) * k**4 * np.pi
                + 10440 * 1j * np.exp(3 * 1j * k) * k**4 * np.pi
                - 1680 * np.exp(1j * k) * k**5 * np.pi
                + 1680 * np.exp(3 * 1j * k) * k**5 * np.pi
                - 120 * 1j * np.exp(1j * k) * k**6 * np.pi
                - 120 * 1j * np.exp(3 * 1j * k) * k**6 * np.pi
                + 120
                * np.exp(1j * k)
                * k**3
                * (
                    -105 * 1j
                    - 87 * k
                    - 14 * 1j * k**2
                    + k**3
                    + np.exp(2 * 1j * k) * (105 * 1j - 87 * k + 14 * 1j * k**2 + k**3)
                )
                * special.sici(-k)[1]
                - 120
                * np.exp(1j * k)
                * k**2
                * (
                    384
                    + 279 * 1j * k
                    - 87 * k**2
                    - 14 * 1j * k**3
                    + k**4
                    + np.exp(2 * 1j * k)
                    * (384 - 279 * 1j * k - 87 * k**2 + 14 * 1j * k**3 + k**4)
                )
                * special.sici(k)[1]
                + 46080 * np.exp(1j * k) * k**2 * np.log(k)
                + 46080 * np.exp(3 * 1j * k) * k**2 * np.log(k)
                + 46080 * 1j * np.exp(1j * k) * k**3 * np.log(k)
                - 46080 * 1j * np.exp(3 * 1j * k) * k**3 * np.log(k)
                + 46080 * 1j * np.exp(1j * k) * k**2 * special.sici(k)[0]
                - 46080 * 1j * np.exp(3 * 1j * k) * k**2 * special.sici(k)[0]
                - 46080 * np.exp(1j * k) * k**3 * special.sici(k)[0]
                - 46080 * np.exp(3 * 1j * k) * k**3 * special.sici(k)[0]
            )
        )
        / np.exp(2 * 1j * k)
    )
    return termA + termB


def qell_k_spherical_(L, k):
    assert L in [0, 2, 4], "Can only compute Q_L(k) for L = 0, 2, 4"
    if L == 0:
        return qellk_0(k)
    elif L == 2:
        if hasattr(k, "__iter__"):
            return np.array([re(qellk_2_unvectorized(kk)) for kk in k])
        else:
            return re(qellk_2_unvectorized(k))
    elif L == 4:
        return qellk_4(k)

def qell_k_spherical(L, k, radius):
    return (2*L+1.) * np.array(qell_k_spherical_(L, radius * k), dtype=np.float32)
