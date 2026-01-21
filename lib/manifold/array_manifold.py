import numpy as np

speed_of_light = 299792458

def ideal_manifold_vectorized(freq_list, phi_list, theta_list, rx_coords)->np.ndarray:
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
    cosphi = np.cos(phi_list)                     # (N,)
    sinphi = np.sin(phi_list)
    costheta = np.cos(theta_list)                   # (M,)
    sintheta = np.sin(theta_list)

    # Create a full grid of directions:
    # u.shape = (N, M, 3)
    ux = sintheta[None, :] * cosphi[:, None]          # (N, M)
    uy = sintheta[None, :] * sinphi[:, None]          # (N, M)
    uz = costheta[None, :] * np.ones((N, M))        # (N, M)

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
    A = np.exp(-1j * phase)/np.sqrt(L)

    return A


def ideal_manifold_loop(freq_list:np.array,phi_list:np.array,theta_list:np.array,rx_coords:np.ndarray)->np.ndarray:
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
                A[f_ind,phi_ind,theta_ind,:] = np.exp(-1j * k_wave * proj)/np.sqrt(num_antennas)

    return A

def ideal_manifold_linear_array_loop(freq_list:np.array,theta_list:np.array,rx_coords:np.ndarray)->np.ndarray:
    '''
    Inputs:
    freq_list - (K) array of frequencies
    theta_list - (M) array of azimuth angles in radians
    rx_coords - (L) array of antenna positions in x (y=0)

    Returns:
    A - (K x M x L) array of array manifold at all combinations of frequencies, az, and for a given array geometry
    '''
    num_freqs = len(freq_list)
    num_theta = len(theta_list)
    num_antennas = np.shape(rx_coords)[0]

    # Direction cosines (unit vectors) for each (phi, theta)
    A = np.empty((num_freqs,num_theta,num_antennas),dtype=complex)
    for f_ind,f in enumerate(freq_list):
        k_wave = 2*np.pi*f/speed_of_light
        for theta_ind,theta in enumerate(theta_list):
            A[f_ind,theta_ind,:] = np.exp(-1j * k_wave * rx_coords * np.sin(theta))

    return A

def antenna_pattern_estimate_loop(freq_list:np.array,phi_list:np.array,theta_list:np.array,rx_coords:np.ndarray)->tuple[np.ndarray,np.ndarray]:
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
    antenna_pattern = np.empty((num_freqs,num_phi,num_theta),dtype=complex)
    for f_ind,f in enumerate(freq_list):
        k_wave = 2*np.pi*f/speed_of_light
        for phi_ind,phi in enumerate(phi_list):
            for theta_ind,theta in enumerate(theta_list):
                direction_manifold = np.array([
                    sin_theta[theta_ind]*cos_phi[phi_ind],
                    sin_theta[theta_ind]*sin_phi[phi_ind],
                    cos_theta[theta_ind]])
                proj = direction_manifold @ rx_coords.T
                A[f_ind,phi_ind,theta_ind,:] = np.exp(-1j * k_wave * proj)/np.sqrt(num_antennas)
    phi_ind_0 = num_phi//2
    theta_ind_0 = num_theta//2
    A_0 = A[:,phi_ind_0,theta_ind_0,:]

    for f_ind,f in enumerate(freq_list):
        for phi_ind,phi in enumerate(phi_list):
            for theta_ind,theta in enumerate(theta_list):
                antenna_pattern[f_ind,phi_ind,theta_ind] = np.dot(A_0[f_ind].conj(),A[f_ind,phi_ind,theta_ind,:])

    return A,antenna_pattern

def compute_ura_coords(Nx:int,Ny:int,lam:float) -> np.array:
    '''
    Computes zero centered uniform rectangular array coordinates with lam/2 spacing, note with even sized arrays, there is no element at 0
    Inputs:
    Nx - Number of x antenna elements
    Ny - Number of y antenna elements
    lam - Wavelength
    Returns:
    rx_coords - (L x 3) array of antenna positions in x,y,z
    '''
    num_ch = Nx*Ny
    rx_coords = np.empty((num_ch,3))
    center_correction_x = 0
    center_correction_y = 0
    if (Nx%2==0):
        center_correction_x = 0.5
    if (Ny%2==0):
        center_correction_y = 0.5
    for a in range(Nx): 
        for b in range(Ny):
            rx_coords[a*Ny+b,:] = np.array([0,a-Nx//2+center_correction_x,b-Ny//2+center_correction_y])*lam/2

    return rx_coords

def ura_subarray_indices(Nx:int, Ny:int, Sx:int, Sy:int)->list:
    """
    Compute the subarray indices for an arbitrary uniform rectangular array
    Inputs:
    Nx: Number of antennas in x dimension
    Ny: Number of antennas in y dimension
    Sx: Number of subantennas in x dimension
    Sy: Number of subantennas in y dimension
    """
    subarrays = []

    for j0 in range(Ny - Sy + 1):
        for i0 in range(Nx - Sx + 1):
            indices = []
            for j in range(Sy):
                for i in range(Sx):
                    idx = (i0 + i) + (j0 + j) * Nx
                    indices.append(idx)
            subarrays.append(indices)

    return subarrays