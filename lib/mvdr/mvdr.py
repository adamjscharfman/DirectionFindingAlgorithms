import numpy as np
from lib.util import general_df

def compute_mvdr_metric(signal:np.ndarray,steering_vectors:np.ndarray,frequency_index:int):
    '''
    Inputs:
    signal - (Num Antennas x Num Samples) IQ data
    steering_vectors - (Num Frequencies x Num Az x Num El x Num Antennas) array manifold at all combinations of frequencies, az, and el angles for a given array geometry
    frequency_index - Index into frequency list of steering vectors

    Returns:
    mvdr_metric - (Num Az x Num El) grid of MUSIC beamforming output
    '''
    R = general_df.compute_correlation_matrix(signal)

    A = steering_vectors[frequency_index]
    # Convert A to a flattened matrix for faster numerical solver
    num_az,num_el,num_ch = A.shape
    num_angles = num_az*num_el
    A_flat = A.reshape(num_angles,num_ch)

    # Numerical solve of X = R^-1 @ A in MVDR demoninator
    X = np.linalg.solve(R,A_flat.T) #(num_ch,num_ch) @ (num_ch,num_angles) -> (num_ch,num_angles)

    # Solve A^H @ X
    denom = np.sum(A_flat.conj() * X.T,axis=1) #(num_angles,num_ch) * (num_angles,num_ch)


    # P = 1 / (aᴴ E_n E_nᴴ a)
    mvdr_metric = 1.0 / np.real(denom)
    return mvdr_metric.reshape(num_az,num_el)

def compute_mvdr_metric_inverse_method(signal:np.ndarray,steering_vectors:np.ndarray,frequency_index:int):
    '''
    Inputs:
    signal - (Num Antennas x Num Samples) IQ data
    steering_vectors - (Num Frequencies x Num Az x Num El x Num Antennas) array manifold at all combinations of frequencies, az, and el angles for a given array geometry
    frequency_index - Index into frequency list of steering vectors

    Returns:
    mvdr_metric - (Num Az x Num El) grid of MUSIC beamforming output
    '''
    R = general_df.compute_correlation_matrix(signal)

    A = steering_vectors[frequency_index]

    # Note - Computing explicit inverse not recommended bc of numerical instability if R is ill-conditioned (det(R) ~ 0)
    inv_R = np.linalg.inv(R)
    # MVDR denominator: a^H R^-1 a for all az/el
    denom = np.einsum("...i,ij,...j -> ...",
                      A.conj(),
                      inv_R,
                      A,
                      optimize=True)

    # P = 1 / (aᴴ E_n E_nᴴ a)
    mvdr_metric = 1.0 / np.real(denom)
    return mvdr_metric

def mvdr_power(array_manifold, cross_corr_mtx, freq_idx, reg=1e-6, eps=1e-12):
    """
    Compute MVDR power per signal and direction:
        P_s(a,e) = 1 / ( w_{f_s,a,e}^H  R_s^{-1}  w_{f_s,a,e} )

    Parameters
    ----------
    array_manifold : ndarray, shape (num_freq, num_az, num_el, num_ch)
        Steering vectors (complex).
    cross_corr_mtx : ndarray, shape (num_signals, num_ch, num_ch)
        Correlation matrices (Hermitian).
    freq_idx : array_like, shape (num_signals,)
        Frequency index for each signal (integers in [0, num_freq)).
    reg : float
        Diagonal loading (adds reg * I to each R for stability).
    eps : float
        Small floor to avoid divide-by-zero.

    Returns
    -------
    P : ndarray, shape (num_signals, num_az, num_el)
        MVDR power for each signal and direction.
    """

    num_signals = cross_corr_mtx.shape[0]
    num_freq, num_az, num_el, num_ch = array_manifold.shape

    # Select per-signal steering blocks: (S, A, E, C)
    w_sel = array_manifold[freq_idx]              # shape: (S, A, E, C)

    # Reshape to put directions into columns so we can solve R_s X = W_s:
    # W has shape (S, C, K) with K = A * E
    K = num_az * num_el
    W = w_sel.reshape(num_signals, K, num_ch).transpose(0, 2, 1)  # (S, C, K)

    # Regularize R (diagonal loading) to improve numerical stability
    if reg is not None and reg != 0:
        R_reg = cross_corr_mtx.copy().astype(np.complex128)
        R_reg += (reg * np.eye(num_ch, dtype=R_reg.dtype))[None, :, :]
    else:
        R_reg = cross_corr_mtx

    # Solve R_reg X = W  (batched solve). X has shape (S, C, K)
    # np.linalg.solve supports stacked systems: first dim is batch
    X = np.linalg.solve(R_reg, W)

    # denom_{s,k} = w^H x = sum_c conj(w_{c}) * x_{c}
    # both W and X are (S, C, K)
    denom = np.einsum('sck,sck->sk', np.conjugate(W), X)   # (S, K)

    # avoid divide-by-zero / tiny complex residues: take real part if nearly real
    denom = np.real_if_close(denom, tol=100)

    # floor denom to eps to avoid blow-ups (if numerically zero or negative small)
    denom = denom.real  # after real_if_close, keep real
    denom = np.maximum(denom, eps)

    P_flat = 1.0 / denom    # shape (S, K)
    P = P_flat.reshape(num_signals, num_az, num_el)

    return P


def mvdr_power_reference(w, R, freq_idx, reg=1e-6, eps=1e-12):
    """
    Reference (loop-based) MVDR power computation:
        P_s(a,e) = 1 / ( w_{f_s,a,e}^H R_s^{-1} w_{f_s,a,e} )

    Parameters
    ----------
    w : ndarray, (num_freq, num_az, num_el, num_ch)
        Steering vectors (complex)
    R : ndarray, (num_signals, num_ch, num_ch)
        Correlation matrices
    freq_idx : array_like, (num_signals,)
        Frequency index for each signal
    reg : float
        Diagonal loading term
    eps : float
        Small floor for numerical safety

    Returns
    -------
    P : ndarray, (num_signals, num_az, num_el)
        MVDR power (real, positive)
    """
    num_signals = R.shape[0]
    num_freq, num_az, num_el, num_ch = w.shape

    P = np.zeros((num_signals, num_az, num_el), dtype=float)

    for s in range(num_signals):
        f = freq_idx[s]

        # Regularize R to make sure it's invertible
        R_reg = R[s] + reg * np.eye(num_ch, dtype=R.dtype)

        # Compute inverse explicitly (slow but clear)
        R_inv = np.linalg.inv(R_reg)

        for a in range(num_az):
            for e in range(num_el):
                wv = w[f, a, e, :]   # steering vector, shape (num_ch,)
                denom = np.conjugate(wv).T @ (R_inv @ wv)
                denom = np.real_if_close(denom)
                denom = np.maximum(denom.real, eps)
                P[s, a, e] = 1.0 / denom

    return P