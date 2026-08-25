r"""Figure 10 rebuilt on the replicate measurements.

A replica of paper_validate_guanine.py in which all four panels come from new
acquisitions carrying five independent replicates per condition, so both quantitative
panels now show error bars:

  (a), (b)  concentration series, five independent dilutions at each of seven
            concentrations   (Spectrum/guanine3/guanine2_concentrations)
  (c), (d)  acquisition-time series at 100 mM, five independent measurements at each
            of six exposures  (Spectrum/Guanine/100mM_2)

Error bars are one standard deviation across the five replicates and the regression is
taken through the means.

All photon counts are hard-coded below, so the figure regenerates without re-running the
model. They were produced by report_photon_counts.py and report_concentration_counts.py
in the DoubleHeaded_Unet project, using the manuscript's readout
I_deep = (X_max - X_min)/10 * sum(deep auxiliary head).

The spectra plotted in (a) and (c) are single replicates, staged in
.\data\guanine new\ and .\data\guanine acquisition new\.

NOTE ON SPLICING: the original script overwrote parts of one plotted curve with a
different measurement (a 1700-1760 cm-1 window in panel (a), and 1590-1660, 860-940 and
800-840 cm-1 windows in panel (c)). None of that is done here: every curve is the model
output for the sample it is labelled with. SPLICE_PANEL_A is retained only so the old
behaviour can be inspected, and is off.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from JournalPlots.JournalStyles import set_ieee_style, save_fig
from RamanUtils.RamanHelpers import get_shift_indices

figure_name = 'validate_guanine_replicates'
figure_path = r'C:\Users\khsat\OneDrive\Documents\My papers\Double Headed Unet Model'

SPLICE_PANEL_A = False    # the new data is plotted exactly as the model produced it

# --- Panel (b): new concentration series, mean and SD over five independent
#     dilutions at each concentration
#     columns: concentration (mM), mean I_deep, standard deviation, n
conc_vs_photons = np.array([
    [20,   30388.8,  26368.2, 5],
    [30,  254573.1,  63601.8, 5],
    [40,  293124.4,  90882.8, 5],
    [50,  472419.4,  50531.7, 5],
    [60,  690779.6, 108121.6, 5],
    [70,  831772.7, 213738.8, 5],
    [80, 1073559.7, 102100.0, 5]
])

# --- Panel (d): new acquisition-time series, mean and SD over the five spectra --------
#     columns: acquisition time (s), mean I_deep, standard deviation, n
aqui_vs_photons = np.array([
    [10,  364579.0,  33760.2, 5],
    [20,  616906.9,  69281.6, 5],
    [30,  840602.5, 106949.8, 5],
    [40, 1017731.9,  46438.6, 5],
    [50, 1219539.9, 131055.1, 5],
    [60, 1430864.5, 108339.6, 5]
])

# 1. Apply IEEE style (double column width)
set_ieee_style(column='double')

# 2. Load the spectra
concentration_file_paths = [
    r'.\data\guanine new\50_O.txt',
    r'.\data\guanine new\40_O.txt',
    r'.\data\guanine new\30_O.txt'
]

aquisition_file_paths = [
    r'.\data\guanine acquisition new\30s_O.txt',
    r'.\data\guanine acquisition new\20s_O.txt',
    r'.\data\guanine acquisition new\10s_O.txt'
]

aquisitions = [30, 20, 10]
concentrations = [50, 40, 30]

conc_text_1 = str(concentrations[0]) + ' mM'
conc_text_2 = str(concentrations[1]) + ' mM'
conc_text_3 = str(concentrations[2]) + ' mM'

aqui_text_1 = str(aquisitions[0]) + ' s'
aqui_text_2 = str(aquisitions[1]) + ' s'
aqui_text_3 = str(aquisitions[2]) + ' s'

conc_data1 = np.loadtxt(concentration_file_paths[0], skiprows=1)
conc_data2 = np.loadtxt(concentration_file_paths[1], skiprows=1)
conc_data3 = np.loadtxt(concentration_file_paths[2], skiprows=1)

aqui_data1 = np.loadtxt(aquisition_file_paths[0], skiprows=1)
aqui_data2 = np.loadtxt(aquisition_file_paths[1], skiprows=1)
aqui_data3 = np.loadtxt(aquisition_file_paths[2], skiprows=1)

# 3. Create figure and master GridSpec
fig = plt.figure(figsize=(8, 7))
gs_master = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.25)

text_kwargs = {'fontsize': 9, 'fontweight': 'bold', 'va': 'top', 'ha': 'right'}
raman_shifts = [654, 1165, 1200, 1270, 1230, 1335, 1380, 1460, 1540]
band_width = 20

# ==========================================
# --- TOP LEFT (a): Corrected Spectrum vs Pred Raman (concentration)
# ==========================================
gs_tl = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0, 0], hspace=0.0)

ax_tl1 = fig.add_subplot(gs_tl[0, 0])
ax_tl2 = fig.add_subplot(gs_tl[1, 0], sharex=ax_tl1)
ax_tl3 = fig.add_subplot(gs_tl[2, 0], sharex=ax_tl1)

ax_tl1.plot(conc_data1[:, 0], conc_data1[:, 1] - conc_data1[:, 3], color='#9467bd', alpha=0.6, label='Raw spectrum-baseline(model)')
ax_tl1.plot(conc_data1[:, 0], conc_data1[:, 2], color='#8c564b', linestyle='-', label='Raman(model)')

alt_data2 = conc_data2[:, 2].copy()
if SPLICE_PANEL_A:
    idx = get_shift_indices(conc_data2[:, 0], 1700, 1760)
    alt_data2[idx] = conc_data1[idx, 2]
ax_tl2.plot(conc_data2[:, 0], conc_data2[:, 1] - conc_data2[:, 3], color='#9467bd', alpha=0.6)
ax_tl2.plot(conc_data2[:, 0], alt_data2, color='#8c564b', linestyle='-')

ax_tl3.plot(conc_data3[:, 0], conc_data3[:, 1] - conc_data3[:, 3], color='#9467bd', alpha=0.6)
ax_tl3.plot(conc_data3[:, 0], conc_data3[:, 2], color='#8c564b', linestyle='-')

for ax in [ax_tl1, ax_tl2, ax_tl3]:
    for shift in raman_shifts:
        ax.axvspan(shift - (band_width / 2), shift + (band_width / 2), color='gray', alpha=0.2, zorder=0)

ax_tl1.text(0.98, 0.85, conc_text_1, transform=ax_tl1.transAxes, **text_kwargs)
ax_tl2.text(0.98, 0.85, conc_text_2, transform=ax_tl2.transAxes, **text_kwargs)
ax_tl3.text(0.98, 0.85, conc_text_3, transform=ax_tl3.transAxes, **text_kwargs)

ax_tl1.tick_params(labelbottom=False)
ax_tl2.tick_params(labelbottom=False)
ax_tl3.set_xlabel('Raman Shift (cm$^{-1}$)')
fig.text(0.04, 0.71, 'Intensity', va='center', rotation='vertical')

ax_tl1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)

# ==========================================
# --- TOP RIGHT (b): Concentration vs Photon Count
# ==========================================
ax_tr = fig.add_subplot(gs_master[0, 1])

x_conc = conc_vs_photons[:, 0]
y_photons = conc_vs_photons[:, 1]
y_err_conc = conc_vs_photons[:, 2]

slope, intercept = np.polyfit(x_conc, y_photons, 1)
y_fit = slope * x_conc + intercept

ss_res = np.sum((y_photons - y_fit) ** 2)
ss_tot = np.sum((y_photons - np.mean(y_photons)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

ax_tr.errorbar(x_conc, y_photons, yerr=y_err_conc, fmt='o', color='#1f77b4',
               markersize=5, capsize=3.5, elinewidth=1.2, linestyle='none',
               label=r'Deep-layer readout $I_{\mathrm{deep}}$', zorder=5)
ax_tr.plot(x_conc, y_fit, color='#ff7f0e', linestyle='--', label='Linear Fit')

ax_tr.set_xlabel('Concentration (mM)')
ax_tr.set_ylabel(r'$I_{\mathrm{deep}}$ (counts)')
ax_tr.grid(True, linestyle='--', alpha=0.5)
ax_tr.text(0.05, 0.90, f'$R^2 = {r_squared:.4f}$', transform=ax_tr.transAxes,
           fontsize=10, fontweight='bold', va='top', ha='left')
ax_tr.legend(loc='lower right')

# ==========================================
# --- BOTTOM LEFT (c): Corrected Spectrum vs Pred Raman (new acquisition series)
# ==========================================
gs_bl = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[1, 0], hspace=0.0)

ax_bl1 = fig.add_subplot(gs_bl[0, 0])
ax_bl2 = fig.add_subplot(gs_bl[1, 0], sharex=ax_bl1)
ax_bl3 = fig.add_subplot(gs_bl[2, 0], sharex=ax_bl1)

ax_bl1.plot(aqui_data1[:, 0], aqui_data1[:, 1] - aqui_data1[:, 3], color='#9467bd', alpha=0.6, label='Raw spectrum-baseline(model)')
ax_bl1.plot(aqui_data1[:, 0], aqui_data1[:, 2], color='#8c564b', linestyle='-', label='Raman(model)')

ax_bl2.plot(aqui_data2[:, 0], aqui_data2[:, 1] - aqui_data2[:, 3], color='#9467bd', alpha=0.6)
ax_bl2.plot(aqui_data2[:, 0], aqui_data2[:, 2], color='#8c564b', linestyle='-')

ax_bl3.plot(aqui_data3[:, 0], aqui_data3[:, 1] - aqui_data3[:, 3], color='#9467bd', alpha=0.6)
ax_bl3.plot(aqui_data3[:, 0], aqui_data3[:, 2], color='#8c564b', linestyle='-')

for ax in [ax_bl1, ax_bl2, ax_bl3]:
    for shift in raman_shifts:
        ax.axvspan(shift - (band_width / 2), shift + (band_width / 2), color='gray', alpha=0.2, zorder=0)

ax_bl1.text(0.98, 0.85, aqui_text_1, transform=ax_bl1.transAxes, **text_kwargs)
ax_bl2.text(0.98, 0.85, aqui_text_2, transform=ax_bl2.transAxes, **text_kwargs)
ax_bl3.text(0.98, 0.85, aqui_text_3, transform=ax_bl3.transAxes, **text_kwargs)

ax_bl1.tick_params(labelbottom=False)
ax_bl2.tick_params(labelbottom=False)
ax_bl3.set_xlabel('Raman Shift (cm$^{-1}$)')
fig.text(0.04, 0.29, 'Intensity', va='center', rotation='vertical')

ax_bl1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False)

# ==========================================
# --- BOTTOM RIGHT (d): Acquisition Time vs Photon Count, with error bars
# ==========================================
ax_br = fig.add_subplot(gs_master[1, 1])

x_aqui = aqui_vs_photons[:, 0]
y_photons_aqui = aqui_vs_photons[:, 1]
y_err_aqui = aqui_vs_photons[:, 2]

# regression through the means
slope_aqui, intercept_aqui = np.polyfit(x_aqui, y_photons_aqui, 1)
y_fit_aqui = slope_aqui * x_aqui + intercept_aqui

ss_res_aqui = np.sum((y_photons_aqui - y_fit_aqui) ** 2)
ss_tot_aqui = np.sum((y_photons_aqui - np.mean(y_photons_aqui)) ** 2)
r_squared_aqui = 1 - (ss_res_aqui / ss_tot_aqui)

ax_br.errorbar(x_aqui, y_photons_aqui, yerr=y_err_aqui, fmt='o', color='#1f77b4',
               markersize=5, capsize=3.5, elinewidth=1.2, linestyle='none',
               label=r'Deep-layer readout $I_{\mathrm{deep}}$', zorder=5)
ax_br.plot(x_aqui, y_fit_aqui, color='#ff7f0e', linestyle='--', label='Linear Fit')

ax_br.set_xlabel('Acquisition Time (s)')
ax_br.set_ylabel(r'$I_{\mathrm{deep}}$ (counts)')
ax_br.grid(True, linestyle='--', alpha=0.5)
ax_br.text(0.05, 0.90, f'$R^2 = {r_squared_aqui:.4f}$', transform=ax_br.transAxes,
           fontsize=10, fontweight='bold', va='top', ha='left')
ax_br.legend(loc='lower right')

# ==========================================
# --- ADDING SUBPLOT LABELS (a, b, c, d)
# ==========================================
label_kwargs = {'fontsize': 8, 'ha': 'center', 'va': 'top'}

ax_tl3.text(0.5, -0.57, '(a)', transform=ax_tl3.transAxes, **label_kwargs)
ax_tr.text(0.5, -0.20, '(b)', transform=ax_tr.transAxes, **label_kwargs)
ax_bl3.text(0.5, -0.57, '(c)', transform=ax_bl3.transAxes, **label_kwargs)
ax_br.text(0.5, -0.20, '(d)', transform=ax_br.transAxes, **label_kwargs)

print(f'(b) concentration: slope {slope:,.0f} per mM, intercept {intercept:,.0f}, R2 {r_squared:.4f}')
print(f'    n = 5 independent dilutions per concentration')
print(f'(d) acquisition:   slope {slope_aqui:,.0f} per s,  intercept {intercept_aqui:,.0f}, R2 {r_squared_aqui:.4f}')

# 4. Save and show
save_fig(name=figure_name, path=figure_path)
plt.show()
