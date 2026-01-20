import numpy as np
from lib.util import general_df

def compute_bartlett_metric(signal:np.ndarray,steering_vectors:np.ndarray,frequency_index:int):
    '''
    Inputs:
    signal - (Num Antennas x Num Samples) IQ data
    steering_vectors - (Num Frequencies x Num Az x Num El x Num Antennas) array manifold at all combinations of frequencies, az, and el angles for a given array geometry
    frequency_index - Index into frequency list of steering vectors

    Returns:
    mvdr_metric - (Num Az x Num El) grid of MUSIC beamforming output
    '''
    R = general_df.compute_correlation_matrix_normalized(signal)

    A = steering_vectors[frequency_index]

    # MVDR denominator: a^H R a for all az/el
    bartlett_metric = np.real(np.einsum("...i,ij,...j -> ...",
                      A.conj(),
                      R,
                      A,
                      optimize=True))

    return bartlett_metric

def compute_bartlett_batch_metric(cross_corr_mtx,array_manifold,freq_index):
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