import numpy as np

def compute_cidf_metric(cross_corr_mtx,array_manifold,freq_index):
    """
    cross_corr_mtx: ndarray, shape (num_signals, num_ch, num_ch)
        Correlation matrices (complex).
    array_manifold: ndarray, shape (num_freq, num_az, num_el, num_ch)
        Steering vectors (complex).
    freq_index


    Returns
    -------
    P : ndarray, shape (num_signals, num_az, num_el)
        Scalar response for each signal, frequency, azimuth, and elevation.
    """

    num_signals, num_ch, _ = cross_corr_mtx.shape

    # Select steering vectors corresponding to each signal's frequency
    w_sel = array_manifold[freq_index, ...]   # (num_signals, num_az, num_el, num_ch)

    #w_sel dim [s: signals, a: azimuth, e: elevation, c: channel]
    #cross_corr dim [s: signals, i: channel, j: channel]
    #sajc->sae computes the dot product along the channel axis
    # Compute wᴴ R w for each signal
    P = np.einsum('saec,sij,sajc->sae',
                  np.conjugate(w_sel),
                  cross_corr_mtx,
                  w_sel,
                  optimize=True)

    return P