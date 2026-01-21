import numpy as np
from lib.bartlett import bartlett
from lib.mvdr import mvdr
from lib.music import music
from lib.music import smooth_music
from lib.manifold import array_manifold
from lib.util import general_df
import time
import matplotlib.pyplot as plt

# Define a signal
fc = 1e9 #Center Hz
lam = 3e8/fc
fs = 50e6 #Sample Rate Hz
az_ind = 20
el_ind = 10
f_ind = 1
num_samples = 2048
time_vec = np.arange(0,num_samples,1)/fs

# Define the receiver
# rx_coords = np.array([[0, 1, 0],
#              [0, -1, 0],
#              [0, 0, 1],
#              [0, 0, -1]]) * lam/2
# num_ch = np.shape(rx_coords)[0]
Nx = 4
Ny = 4
num_ch = Nx*Ny
rx_coords = array_manifold.compute_ura_coords(Nx,Ny,lam)
Nx_sub = 3
Ny_sub = 3
rx_coords_subarray = array_manifold.compute_ura_coords(Nx_sub,Ny_sub,lam)

# Initialize the manifold
f_range = np.array([fc])
az_range_deg = np.arange(-45,45,1)
el_range_deg = np.arange(90-30,90+30,1) #Using polar zenith coordinates (0 is z axis)
num_az = len(az_range_deg)
num_el = len(el_range_deg)
az_range = np.radians(az_range_deg)
el_range = np.radians(el_range_deg)
# steering_vectors = array_manifold.ideal_manifold_loop(f_range,az_range,el_range,rx_coords)
steering_vectors = array_manifold.ideal_manifold_vectorized(f_range,az_range,el_range,rx_coords)
steering_vectors_subarray = array_manifold.ideal_manifold_vectorized(f_range,az_range,el_range,rx_coords_subarray)

# Simulate a target
n_targets = 3
f_ind = 0
az_ind = [40,55,70]
el_ind = [30,30,35]
snr_dB = [30,30,30]
tone_offset = [1e5,1e5,1e5]
# signal = 10**(snr_dB/20) * 1/np.sqrt(2) * (np.random.rand(1,num_samples) + 1j*np.random.rand(1,num_samples))
signal = np.zeros((num_ch,num_samples),dtype=complex)
for tgt in range(n_targets):
    signal_tmp = 10**(snr_dB[tgt]/20) * np.exp(1j*2*np.pi * tone_offset[tgt] * time_vec)
    signal_steered = np.outer(steering_vectors[f_ind,az_ind[tgt],el_ind[tgt]], signal_tmp) # w^H @ signal
    signal += signal_steered
noise = 1/np.sqrt(2) * (np.random.rand(num_ch,num_samples) + 1j*np.random.rand(num_ch,num_samples))
rx_signal = signal + noise

# fig = plt.figure()
# plt.plot(time_vec,np.real(rx_signal[0]),label='Real')
# plt.plot(time_vec,np.imag(rx_signal[0]),label='Imag')
# plt.title('Data on Channel 0')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.legend(loc='upper left')

# Test bartlett Implementation
bartlett_metric = bartlett.compute_bartlett_metric(rx_signal,steering_vectors,f_ind)
mvdr_metric = mvdr.compute_mvdr_metric(rx_signal,steering_vectors,f_ind)
music_metric = music.compute_music_metric(rx_signal,steering_vectors,f_ind)
subarray_list = array_manifold.ura_subarray_indices(Nx,Ny,Nx_sub,Ny_sub)
smooth_music_metric = smooth_music.compute_smooth_music_metric(rx_signal,steering_vectors_subarray,f_ind,subarray_list,num_samples)
smooth_music_fb_metric = smooth_music.compute_forward_backward_smooth_music_metric(rx_signal,steering_vectors_subarray,f_ind,subarray_list,num_samples)

# az_ind_est,el_ind_est,peak_val = general_df.find_argmax(bartlett_metric,num_az,num_el)

# Plot Results
az_mesh, el_mesh = np.meshgrid(az_range, el_range, indexing="ij")
fig,ax = plt.subplots(1,3,figsize=(14,4))
ax[0].pcolormesh(np.degrees(az_mesh),np.degrees(el_mesh),20*np.log10(bartlett_metric),shading='auto')
ax[0].set_title('Bartlett Spectrum')
ax[0].set_xlabel("Azimuth (deg)")
ax[0].set_ylabel("Elevation (deg)")
ax[0].scatter(az_range_deg[az_ind],el_range_deg[el_ind])

ax[1].pcolormesh(np.degrees(az_mesh),np.degrees(el_mesh),20*np.log10(mvdr_metric),shading='auto')
ax[1].set_title('MVDR Spectrum')
ax[1].set_xlabel("Azimuth (deg)")
ax[1].set_ylabel("Elevation (deg)")
ax[1].scatter(az_range_deg[az_ind],el_range_deg[el_ind])

ax[2].pcolormesh(np.degrees(az_mesh),np.degrees(el_mesh),20*np.log10(music_metric),shading='auto')
ax[2].set_title('MUSIC Spectrum')
ax[2].set_xlabel("Azimuth (deg)")
ax[2].set_ylabel("Elevation (deg)")
ax[2].scatter(az_range_deg[az_ind],el_range_deg[el_ind])

# fig.suptitle("Multi Emitter:  deg Spacing")

# MUSIC vs SMOOTH MUSIC
az_mesh, el_mesh = np.meshgrid(az_range, el_range, indexing="ij")
fig,ax = plt.subplots(1,3,figsize=(14,4))
ax[0].pcolormesh(np.degrees(az_mesh),np.degrees(el_mesh),20*np.log10(music_metric),shading='auto')
ax[0].set_title('Music Spectrum')
ax[0].set_xlabel("Azimuth (deg)")
ax[0].set_ylabel("Elevation (deg)")
ax[0].scatter(az_range_deg[az_ind],el_range_deg[el_ind])
ax[1].pcolormesh(np.degrees(az_mesh),np.degrees(el_mesh),20*np.log10(smooth_music_metric),shading='auto')
ax[1].set_title('Smooth Music Spectrum')
ax[1].set_xlabel("Azimuth (deg)")
ax[1].set_ylabel("Elevation (deg)")
ax[1].scatter(az_range_deg[az_ind],el_range_deg[el_ind])
ax[2].pcolormesh(np.degrees(az_mesh),np.degrees(el_mesh),20*np.log10(smooth_music_fb_metric),shading='auto')
ax[2].set_title('Smooth Music Forward Backward Spectrum')
ax[2].set_xlabel("Azimuth (deg)")
ax[2].set_ylabel("Elevation (deg)")
ax[2].scatter(az_range_deg[az_ind],el_range_deg[el_ind])

# plt.scatter(az_range_deg[az_ind],el_range_deg[el_ind],marker='x',label="True")
# plt.scatter(az_range_deg[az_ind_est],el_range_deg[el_ind_est],marker='o',label="Est")

plt.show(block=False)

# Print Results
print(f"Num Subarrays: {len(subarray_list)}")
print(f"Num Subarray DOF: {Nx_sub*Ny_sub}")
print(f"True Detection at (Az,El) = ({az_range_deg[az_ind]},{el_range_deg[el_ind]}) Degrees")
# print(f"Est Detection at (Az/El) = ({az_range_deg[az_ind_est]},{el_range_deg[el_ind_est]}) Degrees")

breakpoint()
