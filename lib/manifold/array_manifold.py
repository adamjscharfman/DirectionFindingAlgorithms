import numpy as np

speed_of_light = 299792458

def ideal_manifold_vectorized(freq_list, phi_list, theta_list, rx_coords):
    """
    Vectorized computation of the array manifold.

    Inputs:
        freq_list  - (K,) frequencies [Hz]
        phi_list   - (N,) azimuth angles [rad]
        theta_list - (M,) elevation angles [rad]
        rx_coords  - (L,3) antenna xyz coordinates [meters]

    Returns:
        A - (K, N, M, L) complex array
    """
    c = speed_of_light
    
    # Shapes
    K = len(freq_list)
    N = len(phi_list)
    M = len(theta_list)
    L = rx_coords.shape[0]

    # ---- 1. Wave numbers k = 2π f / c  ----
    k = 2 * np.pi * freq_list / c               # (K,)

    # ---- 2. Direction unit vectors u(phi,theta) ----
    cosφ = np.cos(phi_list)                     # (N,)
    sinφ = np.sin(phi_list)
    cosθ = np.cos(theta_list)                   # (M,)
    sinθ = np.sin(theta_list)

    # Create a full grid of directions:
    # u.shape = (N, M, 3)
    ux = sinθ[None, :] * cosφ[:, None]          # (N, M)
    uy = sinθ[None, :] * sinφ[:, None]          # (N, M)
    uz = cosθ[None, :] * np.ones((N, M))        # (N, M)

    u = np.stack([ux, uy, uz], axis=-1)         # (N, M, 3)

    # ---- 3. Dot product u ⋅ r_l for all antennas ----
    # rx_coords: (L,3)
    # u:         (N, M, 3)
    #
    # Result: proj = (N, M, L)
    proj = u @ rx_coords.T

    # ---- 4. Apply k for each frequency ----
    #
    # k:     (K,)
    # proj:  (N, M, L)
    #
    # k[:,None,None,None] * proj[None,:,:,:] → (K,N,M,L)
    phase = k[:, None, None, None] * proj[None, :, :, :]

    # ---- 5. Final manifold A = exp(j * phase) ----
    A = np.exp(1j * phase)

    return A


def ideal_manifold_loop(freq_list:np.array,phi_list:np.array,theta_list:np.array,rx_coords:np.ndarray):
    '''
    Inputs:
    freq_list - (K) array of frequencies
    phi_list - (N) array of azimuth angles in radians
    theta_list - (M) array of elevation angles in radians
    rx_coords - (L x 3) array of antenna positions in x,y,z

    Returns:
    A - (K x N x M x L) array of array manifold at all combinations of frequencies, az, and el angles for a given array geometry
    '''
    num_freqs = len(freq_list)
    num_phi = len(phi_list)
    num_theta = len(theta_list)
    num_antennas = np.shape(rx_coords)[0]

    # Direction cosines (unit vectors) for each (phi, theta)
    cos_phi = np.cos(phi_list)
    sin_phi = np.sin(phi_list)
    cos_theta = np.cos(theta_list)
    sin_theta = np.sin(theta_list)
    A = np.empty((num_freqs,num_phi,num_theta,num_antennas),dtype=complex)
    for f_ind,f in enumerate(freq_list):
        k_wave = 2*np.pi*f/speed_of_light
        for phi_ind,phi in enumerate(phi_list):
            for theta_ind,theta in enumerate(theta_list):
                direction_manifold = np.array([
                    sin_theta[theta_ind]*cos_phi[phi_ind],
                    sin_theta[theta_ind]*sin_phi[phi_ind],
                    cos_theta[theta_ind]])
                proj = direction_manifold @ rx_coords.T
                A[f_ind,phi_ind,theta_ind,:] = np.exp(1j * k_wave * proj)

    return A