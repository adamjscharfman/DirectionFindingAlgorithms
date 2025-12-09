import numpy as np


def compute_correlation_matrix(iq_data:np.ndarray):
    '''
    Inputs:
    iq_data - (Num Signals x Num Antennas x Num Samples) batch of received signals

    Returns:
    xcorr_mtx - (Num Signals x Num Antennas x Num Antennas) batch of correlation matrices
    '''

    xcorr_mtx = iq_data @ np.conjugate(np.transpose(iq_data,(0,2,1)))

    return xcorr_mtx