import numpy as np
from lib.music import music
from lib.manifold import array_manifold
from lib.util import general_df
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Define a signal
fc = 1e9 #Center Hz
lam = 3e8/fc
fs = 50e6 #Sample Rate Hz
az_ind = 20
el_ind = 10
f_ind = 1
num_samples = 2048

# Define the receiver
rx_coords = np.array([[0, 1, 0],
             [0, -1, 0],
             [0, 0, 1],
             [0, 0, -1]]) * lam/2
num_ch = np.shape(rx_coords)[0]

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

# Simulate a target
n_targets = 2
f_ind = 0
az_ind = [25,65]
el_ind = [30,30]
snr_dB = [30,30]
signal = np.zeros((num_ch,num_samples),dtype=complex)
for tgt in range(n_targets):
    signal_tmp =  10**(snr_dB[tgt]/20) * 1/np.sqrt(2) * (np.random.rand(1,num_samples) + 1j*np.random.rand(1,num_samples))
    signal_steered = np.outer(steering_vectors[f_ind,az_ind[tgt],el_ind[tgt]], signal_tmp) # w^H @ signal
    signal += signal_steered
noise = 1/np.sqrt(2) * (np.random.rand(num_ch,num_samples) + 1j*np.random.rand(num_ch,num_samples))
rx_signal = signal + noise

# Test MUSIC Implementation
music_metric = music.compute_music_metric(rx_signal,steering_vectors,f_ind)
az_ind_est,el_ind_est,peak_val = general_df.find_argmax(music_metric,num_az,num_el)

# Plot Results
az_mesh, el_mesh = np.meshgrid(az_range, el_range, indexing="ij")
plt.figure()
plt.pcolormesh(np.degrees(az_mesh),np.degrees(el_mesh),20*np.log10(music_metric),shading='auto')
plt.scatter(az_range_deg[az_ind],el_range_deg[el_ind],marker='x',label="True")
plt.scatter(az_range_deg[az_ind_est],el_range_deg[el_ind_est],marker='o',label="Est")
plt.title("Music Spectrum")
plt.xlabel("Azimuth (deg)")
plt.ylabel("Elevation (deg)")
plt.legend()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(np.degrees(az_mesh), np.degrees(el_mesh), 20*np.log10(music_metric), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Azimuth (deg)')
ax.set_ylabel('Elevation (deg)')
ax.set_zlabel('Power Spectrum')
ax.set_title('MUSIC Spectrum Visualization')
plt.show(block=False)

# Print Results
print(f"True Detection at (Az,El) = ({az_range_deg[az_ind]},{el_range_deg[el_ind]}) Degrees")
print(f"Est Detection at (Az/El) = ({az_range_deg[az_ind_est]},{el_range_deg[el_ind_est]}) Degrees")

breakpoint()
