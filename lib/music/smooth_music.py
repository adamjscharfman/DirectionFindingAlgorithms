import numpy as np
from lib.util import general_df
from lib.music import music

# def compute_music_metric(signal:np.ndarray,steering_vectors:np.ndarray,frequency_index:int):
#     '''
#     Inputs:
#     signal - (Num Antennas x Num Samples) IQ data
#     steering_vectors - (Num Frequencies x Num Az x Num El x Num Antennas) array manifold at all combinations of frequencies, az, and el angles for a given array geometry
#     frequency_index - Index into frequency list of steering vectors

#     Returns:
#     music_metric - (Num Az x Num El) grid of MUSIC beamforming output
#     '''
#     R = general_df.compute_correlation_matrix(signal)

#     En_EnH = compute_noise_subspace(R)

#     A = steering_vectors[frequency_index]

#     # MUSIC denominator: aᴴ EnEnᴴ a for all az/el
#     denom = np.einsum("...i,ij,...j -> ...",
#                       A.conj(),
#                       En_EnH,
#                       A,
#                       optimize=True)

#     # P = 1 / (aᴴ E_n E_nᴴ a)
#     music_metric = 1.0 / np.real(denom)
#     return music_metric

def compute_smooth_music_metric(signal:np.ndarray,steering_vectors_subarray:np.ndarray,frequency_index:int,subarray_list:np.array,num_samples:int):
    '''
    Inputs:
    signal - (Num Antennas x Num Samples) IQ data
    steering_vectors_subarray - (Num Frequencies x Num Az x Num El x Num Antennas subarray) array manifold at all combinations of frequencies, az, and el angles for a given array geometry
    frequency_index - Index into frequency list of steering vectors
    subarray_list - (Num subarrays x num_subarray_ch List of indices to form the subarrays

    Returns:
    music_metric - (Num Az x Num El) grid of MUSIC beamforming output
    '''

    R = general_df.compute_correlation_matrix(signal)
    num_subarray = len(subarray_list)
    num_subarray_ch = len(subarray_list[0])

    # R_smooth = R[subarray_list[index,:,0],subarray_list[index,:,0]]
    R_smooth = np.zeros(
        (num_subarray_ch,num_subarray_ch),
        dtype=R.dtype
    )
    for idx in subarray_list:
        R_smooth += R[np.ix_(idx, idx)]
    R_smooth /= num_subarray

    En_EnH = music.compute_noise_subspace(R_smooth,0.01,num_samples,"aic")

    A = steering_vectors_subarray[frequency_index]

    # MUSIC denominator: aᴴ EnEnᴴ a for all az/el
    denom = np.einsum("...i,ij,...j -> ...",
                      A.conj(),
                      En_EnH,
                      A,
                      optimize=True)

    # P = 1 / (aᴴ E_n E_nᴴ a)
    music_metric = 1.0 / np.real(denom)
    return music_metric

def compute_forward_backward_smooth_music_metric(signal:np.ndarray,steering_vectors_subarray:np.ndarray,frequency_index:int,subarray_list:np.array,num_samples:int):
    '''
    Spatial smoothing filter that enforces hermitian
    Inputs:
    signal - (Num Antennas x Num Samples) IQ data
    steering_vectors_subarray - (Num Frequencies x Num Az x Num El x Num Antennas subarray) array manifold at all combinations of frequencies, az, and el angles for a given array geometry
    frequency_index - Index into frequency list of steering vectors
    subarray_list - (Num subarrays x num_subarray_ch List of indices to form the subarrays

    Returns:
    music_metric - (Num Az x Num El) grid of MUSIC beamforming output
    '''

    R = general_df.compute_correlation_matrix(signal)
    num_subarray = len(subarray_list)
    num_subarray_ch = len(subarray_list[0])

    # R_smooth = R[subarray_list[index,:,0],subarray_list[index,:,0]]
    R_smooth = np.zeros(
        (num_subarray_ch,num_subarray_ch),
        dtype=R.dtype
    )
    for idx in subarray_list:
        R_smooth += R[np.ix_(idx, idx)]
    R_smooth /= num_subarray

    # Do forward backward averaging
    J = np.fliplr(np.eye(num_subarray_ch))
    R_smooth_fb = 0.5 * (R_smooth + J@R_smooth.conj()@J)

    En_EnH = music.compute_noise_subspace(R_smooth_fb,0.01,num_samples,"aic")

    A = steering_vectors_subarray[frequency_index]

    # MUSIC denominator: aᴴ EnEnᴴ a for all az/el
    denom = np.einsum("...i,ij,...j -> ...",
                      A.conj(),
                      En_EnH,
                      A,
                      optimize=True)

    # P = 1 / (aᴴ E_n E_nᴴ a)
    music_metric = 1.0 / np.real(denom)
    return music_metric


# def compute_batch_noise_subspace(R: np.ndarray, threshold_ratio: float = 0.1):
#     '''
#     Inputs:
#     R - (Num Signals x Num Antennas x Num Antennas) batch of correlation matrices
#     threshold_ratio : eigenvalues <= threshold_ratio * max_eigenvalue 
#                       are considered noise eigenvalues

#     Returns:
#     En_EnH - (Num Signals x Num Antennas x Num Antennas) noise projectors En @ En^H
#     '''

#     num_signals,num_antennas,_ = np.shape(R)

#     # Compute eigendecomposition
#     eigvals,eigvecs = np.linalg.eigh(R)

#     # Extract the max eigvals (this should be the last eigval)
#     #max_eigvals = eigvals.max(axis=1,keepdims=True)
#     max_eigvals = eigvals[:,-1]

#     # Compute the noise threshold
#     noise_mask = eigvals <= (threshold_ratio * max_eigvals)

#     # Return noise subspace outer product for music metric
#     En_EnH = np.zeros(num_signals,num_antennas,num_antennas)
#     for s in range(num_signals):
#         En = eigvecs[s][:,noise_mask[s]] # (N x Num Noise)
#         En_EnH[s] = En @ En.conj().T

#     # Fully vectorized
#     # masked_vecs = eigvecs * noise_mask[:, None, :]
#     # En_EnH2 = np.einsum("sni,smi->snm", masked_vecs, masked_vecs.conj())
    
#     return En_EnH