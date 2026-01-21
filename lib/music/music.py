import numpy as np
from lib.util import general_df

def compute_music_metric(signal:np.ndarray,steering_vectors:np.ndarray,frequency_index:int):
    '''
    Inputs:
    signal - (Num Antennas x Num Samples) IQ data
    steering_vectors - (Num Frequencies x Num Az x Num El x Num Antennas) array manifold at all combinations of frequencies, az, and el angles for a given array geometry
    frequency_index - Index into frequency list of steering vectors

    Returns:
    music_metric - (Num Az x Num El) grid of MUSIC beamforming output
    '''
    R = general_df.compute_correlation_matrix(signal)
    En_EnH = compute_noise_subspace(R,num_samples=signal.shape[1])

    A = steering_vectors[frequency_index]

    # MUSIC denominator: aᴴ EnEnᴴ a for all az/el
    denom = np.einsum("...i,ij,...j -> ...",
                      A.conj(),
                      En_EnH,
                      A,
                      optimize=True)

    # P = 1 / (aᴴ E_n E_nᴴ a)
    music_metric = 1.0 / np.real(denom)
    return music_metric

def compute_batch_music_metric(En_EnH: np.ndarray, steering_vectors: np.ndarray, frequency_index: list[int]):
    '''
    Inputs:
    En_EnH - (Num Signals x Num Antennas x Num Antennas) Noise subspace outer product
    steering_vectors - (Num Frequencies x Num Az x Num El x Num Antennas) array manifold at all combinations of frequencies, az, and el angles for a given array geometry
    frequency_index - List of indices into frequency list of steering vectors

    Returns:
    music_metric - (Num Signals x Num Az x Num El) grid of MUSIC beamforming output
    '''

     # Select only the desired frequencies
    A = steering_vectors[frequency_index]

    music_metric_denom = np.einsum(
        'sij,faei,sjk,faek->sfae',
        En_EnH,
        A.conj(),
        En_EnH,
        A,
        optimize=True
    )

    # MUSIC spectrum is inverse of projection onto noise space
    music_metric = 1.0/np.real(music_metric_denom)
    return music_metric

def compute_noise_subspace(R:np.ndarray,threshold_ratio:float=0.01, num_samples:int=2048,method:str="mdl"):
    '''
    Inputs:
    R - ( Num Antennas x Num Antennas) Correlation matrix
    threshold_ratio : eigenvalues <= threshold_ratio * max_eigenvalue 
                      are considered noise eigenvalues

    Returns:
    En_EnH - (Num Antennas x Num Antennas) noise projectors En @ En^H
    '''
    # Compute eigendecomposition
    eigvals,eigvecs = np.linalg.eigh(R)
    M = np.shape(R)[0]
    # Extract the max eigvals (this should be the last eigval)
    #max_eigvals = eigvals.max(axis=1,keepdims=True)
    if method == "thresh":
        max_eigval = eigvals[-1]
        noise_mask = eigvals <= (threshold_ratio * max_eigval)
    elif method == "aic":
        num_sources,_ = aic_num_sources(eigvals[::-1],num_samples)
        noise_mask = np.ones(M, dtype=bool)   # start: everything is noise
        noise_mask[M-num_sources:] = False            # first k_hat are signal
    elif method == "mdl":
        num_sources,_ = mdl_num_sources(eigvals[::-1],num_samples)
        noise_mask = np.ones(M, dtype=bool)   # start: everything is noise
        noise_mask[M-num_sources:] = False            # first k_hat are signal

    # Compute the noise threshold
    En = eigvecs[:,noise_mask]
    En_EnH = En @ En.conj().T
    return En_EnH

def compute_batch_noise_subspace(R: np.ndarray, threshold_ratio: float = 0.1):
    '''
    Inputs:
    R - (Num Signals x Num Antennas x Num Antennas) batch of correlation matrices
    threshold_ratio : eigenvalues <= threshold_ratio * max_eigenvalue 
                      are considered noise eigenvalues

    Returns:
    En_EnH - (Num Signals x Num Antennas x Num Antennas) noise projectors En @ En^H
    '''

    num_signals,num_antennas,_ = np.shape(R)

    # Compute eigendecomposition
    eigvals,eigvecs = np.linalg.eigh(R)

    # Extract the max eigvals (this should be the last eigval)
    #max_eigvals = eigvals.max(axis=1,keepdims=True)
    max_eigvals = eigvals[:,-1]

    # Compute the noise threshold
    noise_mask = eigvals <= (threshold_ratio * max_eigvals)

    # Return noise subspace outer product for music metric
    En_EnH = np.zeros(num_signals,num_antennas,num_antennas)
    for s in range(num_signals):
        En = eigvecs[s][:,noise_mask[s]] # (N x Num Noise)
        En_EnH[s] = En @ En.conj().T

    # Fully vectorized
    # masked_vecs = eigvecs * noise_mask[:, None, :]
    # En_EnH2 = np.einsum("sni,smi->snm", masked_vecs, masked_vecs.conj())
    
    return En_EnH

def aic_num_sources(eigvals:np.array, N:int)->tuple[int,np.array]:
    """
    Estimate number of sources using Akaike Information Criterion (AIC).

    Inputs:
    eigvals - Eigenvalues of covariance matrix (sorted descending).
    N - Number of snapshots.
    Returns:
    k_hat - Estimated number of sources.
    aic - AIC cost for k = 0,...,M-1
    """
    eigvals = np.asarray(eigvals).real
    M = len(eigvals)

    # Small floor to avoid log(0)
    eps = 1e-12
    eigvals = np.maximum(eigvals, eps)

    aic = np.zeros(M)

    for k in range(M):
        m_k = M - k
        if m_k == 0:
            aic[k] = np.inf
            continue

        noise_eigs = eigvals[k:]

        A_k = np.mean(noise_eigs)
        G_k = np.exp(np.mean(np.log(noise_eigs)))  # geometric mean

        aic[k] = (
            -2 * N * m_k * np.log(G_k / A_k)
            + 2 * k * (2 * M - k)
        )

    k_hat = np.argmin(aic)
    return k_hat, aic

def mdl_num_sources(eigvals:np.ndarray, N:int)->tuple[int,np.array]:
    """
    Estimate number of sources using Minimum Descriptor Length (MDL).

    Inputs:
    eigvals - Eigenvalues of covariance matrix (sorted descending).
    N - Number of snapshots.
    Returns:
    k_hat - Estimated number of sources.
    mdl - mdl cost for k = 0,...,M-1
    """
    M = len(eigvals)
    mdl_vals = np.zeros(M)

    for k in range(M):
        if k == M:
            mdl_vals[k] = np.inf
            continue

        noise_eigs = eigvals[k:]

        # Avoid numerical issues
        if np.any(noise_eigs <= 0):
            mdl_vals[k] = np.inf
            continue

        # Geometric and arithmetic means
        g_mean = np.exp(np.mean(np.log(noise_eigs)))
        a_mean = np.mean(noise_eigs)

        mdl_vals[k] = (
            -N * (M - k) * np.log(g_mean / a_mean)
            + 0.5 * k * (2 * M - k) * np.log(N)
        )

    k_hat = np.argmin(mdl_vals)
    return k_hat, mdl_vals
