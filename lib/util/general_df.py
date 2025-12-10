import numpy as np


def compute_correlation_matrix(iq_data:np.ndarray):
    '''
    Inputs:
    iq_data - (Num Antennas x Num Samples) Received signal

    Returns:
    xcorr_mtx - (Num Antennas x Num Antennas) Correlation matrices
    '''

    xcorr_mtx = iq_data @ iq_data.conj().T

    return xcorr_mtx

def compute_batch_correlation_matrix(iq_data:np.ndarray):
    '''
    Inputs:
    iq_data - (Num Signals x Num Antennas x Num Samples) batch of received signals

    Returns:
    xcorr_mtx - (Num Signals x Num Antennas x Num Antennas) batch of correlation matrices
    '''

    xcorr_mtx = iq_data @ np.conjugate(np.transpose(iq_data,(0,2,1)))

    return xcorr_mtx

def find_argmax(df_metric:np.ndarray,num_az:int,num_el:int):
    '''
    Inputs:
    df_metric - (Num Az x Num El) Grid of angle of arrival beamforming output
    num_az - Number of azimuth points in grid
    num_el - Number of elevation points in grid

    Returns:
    az_ind - Azimuth index of max df metric 
    el_ind - Elevation index of max df metric
    val - Value of df metric
    '''

    max_ind = np.argmax(df_metric)
    az_ind,el_ind = np.unravel_index(max_ind,(num_az,num_el))
    val = df_metric[az_ind,el_ind]

    return az_ind,el_ind,val