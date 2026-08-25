import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from RamanUtils.RamanHelpers import remove_laser
from RamanUtils.airPLS import airPLS
from scipy.signal import savgol_filter
laser_folder=r'C:\Users\khsat\OneDrive\Documents\python codes\Raman Factory\Raman Data\Leslie F. (40-60s)20\balackWell_background\LightNoise_65s'


output_file=r'C:\Users\khsat\OneDrive\Documents\python codes\Raman Factory\DoubleHeaded_Unet\data\noise_bank.npz'
laser_files = sorted(glob.glob(os.path.join(laser_folder, "*.asc")))
print(f"Found {len(laser_files)} laser-on files.")


laser_intensities = []
baseline_intensities = []
signal_intensities = []
lam=1e3
data=np.zeros(1)

for f in laser_files:
    data = remove_laser(np.loadtxt(f),294)
    n=data.shape[1]
    baseline=np.zeros_like(data)
    signal=np.zeros_like(data)
    for ii in np.arange(1,n):
        baseline[:,ii]=airPLS(data[:,ii],lam,2,40)
        signal[:,ii]=data[:,ii]-baseline[:,ii]

    
    
    intensities = data[:, 1:].T
    laser_intensities.append(intensities)
    baselines = baseline[:, 1:].T
    baseline_intensities.append(baselines)
    signals = signal[:, 1:].T
    signal_intensities.append(signals)
    
laser_stack = np.vstack(laser_intensities)
baseline_stack = np.vstack(baseline_intensities)
signal_stack = np.vstack(signal_intensities)
raman_shift=data[:,0]
laser_mean = np.mean(laser_stack, axis=0)
smooth_mean = np.mean(signal_stack, axis=0)
# Subtract mean to get pure random shot noise + fluctuations
laser_noise = laser_stack - laser_mean


Noise= signal_stack-smooth_mean
idx=88
plt.figure()
plt.plot(raman_shift,laser_stack[idx,:])
plt.plot(raman_shift,baseline_stack[idx,:])


plt.figure()
plt.plot(raman_shift,signal_stack[idx,:]-smooth_mean)
plt.plot(raman_shift,savgol_filter(smooth_mean, window_length=51, polyorder=2))

#np.savez(output_file,noise=Noise)
# plt.figure()
# plt.plot(raman_shift,laser_noise[83,:])
plt.show()