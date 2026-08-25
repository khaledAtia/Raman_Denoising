import numpy as np
import os
import glob
from RamanUtils.RamanHelpers import remove_laser
from RamanUtils.airPLS import airPLS,WhittakerSmooth
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- Configuration ---
folder_path =r'C:\Users\khsat\OneDrive\Documents\python codes\Raman Factory\Raman Data\Leslie F. (40-60s)20\balackWell_background\LightNoise_65s'  # Replace with your folder path
save_folder_path=r'C:\Users\khsat\OneDrive\Documents\python codes\Raman Factory\DoubleHeaded_Unet\data'
extension = '*.asc'            # The file type to look for
output_filename = 'combined_data.npz'
output_baseline_filename = 'baseline_data.npz'
output_noise_filename='noise_data.npz'
output_noise0_filename='noise_data_mean0.npz'

save_path = os.path.join(save_folder_path, output_filename)
save_baseline_path= os.path.join(save_folder_path, output_baseline_filename)
save_noise_path= os.path.join(save_folder_path, output_noise_filename)
save_noise0_path=os.path.join(save_folder_path, output_noise_filename)

# 1. Get a sorted list of all files
search_pattern = os.path.join(folder_path, extension)
files = sorted(glob.glob(search_pattern))

data_list = []
baseline_list = []
noise_list = []
w=np.ones(864)
lm=0.1e4

print(f"Found {len(files)} files. Processing...")

for i, file_path in enumerate(files): # loop over files
    # Skip the output file if it already exists in the folder to avoid errors
    if os.path.basename(file_path) == output_filename:
        continue
    
    # 2. Load the data
    # delimiter=None defaults to whitespace. Use ',' for CSVs.
    current_data = np.loadtxt(file_path, delimiter=None)
    current_data=remove_laser(current_data,294)
    num_of_spectrum= current_data.shape[1]
    baseline_spec_array=np.zeros((864,num_of_spectrum))
    baseline_spec_array[:,0]=current_data[:,0]
    for ii in range(1,num_of_spectrum): # loop over spectrums
        #smoothed_spec=WhittakerSmooth(current_data[:,ii],w,lm,2)
        smoothed_spec=airPLS(current_data[:,ii],lm,2,40)
        baseline_spec_array[:,ii]=smoothed_spec

    noise_spec_array=current_data-baseline_spec_array
    noise_spec_array[:,0]=current_data[:,0]
        

    
    # Ensure 1D arrays are treated as 2D columns if necessary
    if current_data.ndim == 1:
        current_data = current_data[:, np.newaxis]

    # 3. Logic to handle columns
    if i == 0:
        # For the FIRST file: Keep all columns (The shared first column + its data)
        data_list.append(current_data)
        baseline_list.append(baseline_spec_array)
        noise_list.append(noise_spec_array)
    else:
        # For ALL OTHER files: Slice off the first column [:, 1:] and keep the rest
        data_list.append(current_data[:, 1:])
        baseline_list.append(baseline_spec_array[:,1:])
        noise_list.append(noise_spec_array[:,1:])

cu_data=data_list[0]
cu_baseline=baseline_list[0]

plt.figure()
plt.plot(cu_data[:,0],cu_data[:,1])
plt.plot(cu_data[:,0],cu_baseline[:,1])





# 4. Join them horizontally (column-wise)
if data_list:
    final_array = np.hstack(data_list)
    final_baseline_array=np.hstack(baseline_list)
    final_noise_array=np.hstack(noise_list)
    noise_mean_zero=final_noise_array
    noise_mean=np.zeros((864,2),dtype=float)
    noise_mean[:,0]=final_noise_array[:,0]
    noise_mean[:,1]=np.mean(final_noise_array[:,1:],axis=1)
    plt.figure()
    plt.plot(noise_mean[:,0],noise_mean[:,1])
    plt.plot(noise_mean[:,0],savgol_filter(noise_mean[:,1],21,2))


    noise_mean_zero[:,1:]=final_noise_array[:,1:]-noise_mean[:,1,np.newaxis]

    plt.figure()
    plt.plot(noise_mean_zero[:,0],noise_mean_zero[:,18])
    # 5. Save the result
    
    # fmt='%.6f' prevents scientific notation if you prefer standard float
    np.savez(save_path, data=final_array)
    np.savez(save_baseline_path, data=final_baseline_array)
    #np.savez(save_noise_path, data=final_noise_array)
    np.savez(save_noise0_path, data=noise_mean_zero)
    print(f"Successfully saved merged array to: {save_path}")
    print(f"Final array shape: {final_array.shape}")
else:
    print("No files found to merge.")


plt.show()