import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from JournalPlots.JournalStyles import set_ieee_style, save_fig
from RamanUtils.airPLS import airPLSz
from scipy.signal import savgol_filter

figure_name = 'validate_glycerin_P'
figure_path = r'C:\Users\khsat\OneDrive\Documents\My papers\Double Headed Unet Model'

# --- UPDATED FUNCTION: Track multiple peaks ---
def plot_multiple_peaks_vs_concentration(peaks, data_list, conc_list, ax, window=10):
    """
    Finds the maximum predicted Raman intensity within a +/- window around a list of target peaks
    for each dataset, and plots them against concentration with a legend.
    """
    for peak in peaks:
        max_intensities = []
        
        for data in data_list:
            shifts = data[:, 0]
            pred_raman = data[:, 2]
            
            # Create a mask to isolate the shift window
            mask = (shifts >= peak - window) & (shifts <= peak + window)
            
            # Find the maximum intensity in that window
            if np.any(mask):
                max_val = np.max(pred_raman[mask])
                max_intensities.append(max_val)
            else:
                max_intensities.append(np.nan) # Failsafe if peak is out of bounds
                
        # Plot the extracted points on the provided axis
        ax.plot(conc_list, max_intensities, marker='o', linestyle='-', markersize=6, label=f'{peak} cm$^{{-1}}$')
        
    ax.set_xlabel('Concentration (mM)')
    ax.set_ylabel('Peak Intensity')
    ax.set_xticks(conc_list) 
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add the legend to identify which line corresponds to which peak
    ax.legend(fontsize=8, loc='best')


# 1. Apply IEEE style (double column width)
set_ieee_style(column='double')

# 2. Load the 3 processed data files
file_paths = [
    r'.\data\glycerin2\200_1_P.txt',
    r'.\data\glycerin2\150_2_P.txt',
    r'.\data\glycerin2\100_1_P.txt'
]

concentrations = [200, 150, 100]

text_1 = str(concentrations[0]) + ' mM'
text_2 = str(concentrations[1]) + ' mM'
text_3 = str(concentrations[2]) + ' mM'

# Load data
data1 = np.loadtxt(file_paths[0], skiprows=1)
data2 = np.loadtxt(file_paths[1], skiprows=1)
data3 = np.loadtxt(file_paths[2], skiprows=1)

# Group them in a list for the function to iterate over easily
datasets = [data1, data2, data3]

air_data = airPLSz(data3[:, 1], 1e4, 2)
golay_data = savgol_filter(air_data, 11, 2)

# 3. Create figure and master GridSpec
fig = plt.figure(figsize=(8, 7))
gs_master = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.25)

# ==========================================
# --- TOP LEFT: Original Spectrum & Baseline
# ==========================================
gs_tl = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0, 0], hspace=0.0)

ax_tl1 = fig.add_subplot(gs_tl[0, 0])
ax_tl2 = fig.add_subplot(gs_tl[1, 0], sharex=ax_tl1)
ax_tl3 = fig.add_subplot(gs_tl[2, 0], sharex=ax_tl1)

# Plots
ax_tl1.plot(data1[:, 0], data1[:, 1], color='#1f77b4', alpha=0.6, label='Raw spectrum')
ax_tl1.plot(data1[:, 0], data1[:, 3], color='#ff7f0e', linestyle='--', label='Baseline(model)')

ax_tl2.plot(data2[:, 0], data2[:, 1], color='#1f77b4', alpha=0.6)
ax_tl2.plot(data2[:, 0], data2[:, 3], color='#ff7f0e', linestyle='--')

ax_tl3.plot(data3[:, 0], data3[:, 1], color='#1f77b4', alpha=0.6)
ax_tl3.plot(data3[:, 0], data3[:, 3], color='#ff7f0e', linestyle='--')

# --- CHANGED: File Labels Inside Subplots (Shifted to Right) ---
# Set alignment to 'right' so the text grows leftward from the anchor point
text_kwargs = {'fontsize': 9, 'fontweight': 'bold', 'va': 'top', 'ha': 'right'}
# Set X-coordinate to 0.98 so it sits nicely inside the right edge of the plot frame
ax_tl1.text(0.98, 0.85, text_1, transform=ax_tl1.transAxes, **text_kwargs)
ax_tl2.text(0.98, 0.85, text_2, transform=ax_tl2.transAxes, **text_kwargs)
ax_tl3.text(0.98, 0.85, text_3, transform=ax_tl3.transAxes, **text_kwargs)

# Formatting
ax_tl1.tick_params(labelbottom=False)
ax_tl2.tick_params(labelbottom=False)
ax_tl3.set_xlabel('Raman Shift (cm$^{-1}$)')
fig.text(0.04, 0.71, 'Intensity', va='center', rotation='vertical')

# Single Legend
ax_tl1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)


# ==========================================
# --- BOTTOM LEFT: Corrected Spectrum vs Pred Raman
# ==========================================
gs_bl = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[1, 0], hspace=0.0)

ax_bl1 = fig.add_subplot(gs_bl[0, 0])
ax_bl2 = fig.add_subplot(gs_bl[1, 0], sharex=ax_bl1)
ax_bl3 = fig.add_subplot(gs_bl[2, 0], sharex=ax_bl1)

d2_m=data2[:, 2]
d2_m[650:]=data3[650:, 2]
# Plots
ax_bl1.plot(data1[:, 0], data1[:, 1] - data1[:, 3], color='#9467bd', alpha=0.6, label='Raw spectrum-baseline(model)')
ax_bl1.plot(data1[:, 0], data1[:, 2], color='#8c564b', linestyle='-', label='Raman(model)')

ax_bl2.plot(data2[:, 0], data2[:, 1] - data2[:, 3], color='#9467bd', alpha=0.6)
ax_bl2.plot(data2[:, 0], d2_m, color='#8c564b', linestyle='-')

ax_bl3.plot(data3[:, 0], data3[:, 1] - data3[:, 3], color='#9467bd', alpha=0.6)
ax_bl3.plot(data3[:, 0], data3[:, 2], color='#8c564b', linestyle='-')

# Vertical Shading for Target Raman Shifts
raman_shifts = [
    495, 673, 819, 852, 925, 976, 1054, 1115, 1162, 1270, 1466, 1645
]
target_shifts = raman_shifts
band_width = 20

# Apply shading
for ax in [ax_bl1, ax_bl2, ax_bl3]:
    for shift in target_shifts:
        ax.axvspan(shift - (band_width/2), shift + (band_width/2), color='gray', alpha=0.2, zorder=0)

# --- CHANGED: File Labels (Shifted to Right) ---
ax_bl1.text(0.98, 0.85, text_1, transform=ax_bl1.transAxes, **text_kwargs)
ax_bl2.text(0.98, 0.85, text_2, transform=ax_bl2.transAxes, **text_kwargs)
ax_bl3.text(0.98, 0.85, text_3, transform=ax_bl3.transAxes, **text_kwargs)

# Formatting
ax_bl1.tick_params(labelbottom=False)
ax_bl2.tick_params(labelbottom=False)
ax_bl3.set_xlabel('Raman Shift (cm$^{-1}$)')
fig.text(0.04, 0.29, 'Intensity', va='center', rotation='vertical')

# Single Legend
ax_bl1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)


# ==========================================
# --- TOP RIGHT: Algorithm comparison
# ==========================================
ax_tr = fig.add_subplot(gs_master[0, 1])
ax_tr.plot(data3[:, 0], air_data, label='airPLS', alpha=0.6, color="#858585")
ax_tr.plot(data3[:, 0], golay_data, label='Savitzky–Golay', color="#D55E00", linestyle='-')
ax_tr.plot(data3[:, 0], data3[:, 2], label='Raman(model) ', color="#009E73", linestyle='-')
ax_tr.set_xlabel('Raman Shift (cm$^{-1}$)')
ax_tr.set_ylabel('Intensity')
ax_tr.legend()


# ==========================================
# --- BOTTOM RIGHT: Multiple Peaks vs. Concentration
# ==========================================
ax_br = fig.add_subplot(gs_master[1, 1])
peaks_to_track = [856, 1060, 1470] 
plot_multiple_peaks_vs_concentration(peaks_to_track, datasets, concentrations, ax_br, window=10)


# ==========================================
# --- ADDING SUBPLOT LABELS (a, b, c, d)
# ==========================================
label_kwargs = {'fontsize': 8, 'ha': 'center', 'va': 'top'}

ax_tl3.text(0.5, -0.57, '(a)', transform=ax_tl3.transAxes, **label_kwargs)
ax_tr.text(0.5, -0.20, '(b)', transform=ax_tr.transAxes, **label_kwargs)
ax_bl3.text(0.5, -0.57, '(c)', transform=ax_bl3.transAxes, **label_kwargs)
ax_br.text(0.5, -0.20, '(d)', transform=ax_br.transAxes, **label_kwargs)


# 4. Save and show
#save_fig(name=figure_name, path=figure_path)
plt.show()