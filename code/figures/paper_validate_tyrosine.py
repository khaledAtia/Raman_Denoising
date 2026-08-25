r"""Tyrosine validation figure, in the style of paper_validate_guanine.

Same four panels and the same numbers as the version prepared for the Supporting
Information; only the styling changes, so that the figure matches the rest of the paper:
the IEEE double-column style, the shaded band columns, and the same colours and legend
conventions used for the guanine and glycerol figures.

  (a) the raw measurement at 100 mM with the predicted baseline
  (b) the recovered Raman signal against the difference between the raw spectrum and the
      predicted baseline, which is independent of the Raman branch
  (c) the recovered Raman signal at each concentration
  (d) the recovered signal integrated between 400 and 1700 cm-1, against concentration

Spectra are the processed outputs staged in .\data\tyrosine\; the integrated values are
hard-coded, as in the other paper figure scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from JournalPlots.JournalStyles import set_ieee_style, save_fig

figure_name = 'validate_tyrosine'
figure_path = r'C:\Users\khsat\OneDrive\Documents\My papers\Double Headed Unet Model'

# Recovered signal integrated between 400 and 1700 cm-1
conc_vs_integral = np.array([
    [25,  104514.7],
    [50,  233376.8],
    [75,  367008.7],
    [100, 484204.9]
])

set_ieee_style(column='double')

concentrations = [100, 75, 50, 25]
file_paths = [rf'.\data\tyrosine\{c}_O.txt' for c in concentrations]
data = [np.loadtxt(f, skiprows=1) for f in file_paths]

LO, HI = 400, 1700
raman_shifts = [649, 837, 859, 1179, 1214, 1608]
band_width = 20

def mark_bands(ax, shifts, width, label=True, fs=6.5, min_gap=60):
    """Shade the reported bands and, optionally, write their wavenumbers on top.

    Labels for bands closer together than min_gap are staggered vertically, so that
    the close pairs at 837/859 and 1179/1214 do not overprint one another.
    """
    for shift in shifts:
        ax.axvspan(shift - width / 2, shift + width / 2,
                   color='gray', alpha=0.2, zorder=0)
    if not label:
        return
    prev, low = None, False
    for shift in sorted(shifts):
        low = (prev is not None and shift - prev < min_gap) and not low
        ax.annotate(str(shift), xy=(shift, 1.0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -2 - (26 if low else 0)), textcoords='offset points',
                    ha='center', va='top', rotation=90,
                    fontsize=fs, color='0.30')
        prev = shift


fig = plt.figure(figsize=(8, 7))
gs_master = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.25)

text_kwargs = {'fontsize': 9, 'fontweight': 'bold', 'va': 'top', 'ha': 'right'}

top = data[0]
mask = (top[:, 0] >= LO) & (top[:, 0] <= HI)

# ==========================================
# --- TOP LEFT (a): raw measurement and predicted baseline
# ==========================================
ax_tl = fig.add_subplot(gs_master[0, 0])
ax_tl.plot(top[mask, 0], top[mask, 1], color='#7f7f7f', alpha=0.8, label='Raw spectrum')
ax_tl.plot(top[mask, 0], top[mask, 3], color='#2ca02c', linestyle='--',
           label='Baseline(model)')
ax_tl.text(0.98, 0.85, '100 mM', transform=ax_tl.transAxes, **text_kwargs)
ax_tl.set_xlabel('Raman Shift (cm$^{-1}$)')
ax_tl.set_ylabel('Intensity')
ax_tl.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

# ==========================================
# --- TOP RIGHT (b): recovered signal against the independent difference
# ==========================================
ax_tr = fig.add_subplot(gs_master[0, 1])
mark_bands(ax_tr, raman_shifts, band_width)
ax_tr.plot(top[mask, 0], top[mask, 1] - top[mask, 3], color='#9467bd', alpha=0.6,
           label='Raw spectrum-baseline(model)')
ax_tr.plot(top[mask, 0], top[mask, 2], color='#8c564b', linestyle='-',
           label='Raman(model)')
ax_tr.set_ylim(top=ax_tr.get_ylim()[1] * 1.34)
ax_tr.text(0.98, 0.85, '100 mM', transform=ax_tr.transAxes, **text_kwargs)
ax_tr.set_xlabel('Raman Shift (cm$^{-1}$)')
ax_tr.set_ylabel('Intensity')
ax_tr.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

# ==========================================
# --- BOTTOM LEFT (c): the concentration series
# ==========================================
ax_bl = fig.add_subplot(gs_master[1, 0])
mark_bands(ax_bl, raman_shifts, band_width)

# darkest for the highest concentration, so the ordering reads directly off the plot
shades = ['#08306b', '#2171b5', '#6baed6', '#bdd7e7']
for c, d, col in zip(concentrations, data, shades):
    m = (d[:, 0] >= LO) & (d[:, 0] <= HI)
    ax_bl.plot(d[m, 0], d[m, 2], color=col, linewidth=1.1, label=f'{c} mM')

ax_bl.set_xlim(LO, HI)
ax_bl.set_ylim(top=ax_bl.get_ylim()[1] * 1.42)
ax_bl.set_xlabel('Raman Shift (cm$^{-1}$)')
ax_bl.set_ylabel('Intensity')
ax_bl.legend(loc='upper left', bbox_to_anchor=(0.015, 0.78), frameon=False, ncol=2, fontsize=8,
             handlelength=1.4, columnspacing=1.0)

# ==========================================
# --- BOTTOM RIGHT (d): integrated response against concentration
# ==========================================
ax_br = fig.add_subplot(gs_master[1, 1])

x_conc = conc_vs_integral[:, 0]
y_int = conc_vs_integral[:, 1]

slope, intercept = np.polyfit(x_conc, y_int, 1)
y_fit = slope * x_conc + intercept
ss_res = np.sum((y_int - y_fit) ** 2)
ss_tot = np.sum((y_int - np.mean(y_int)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

ax_br.scatter(x_conc, y_int, color='#1f77b4', s=40, label='Recovered signal', zorder=5)
ax_br.plot(x_conc, y_fit, color='#ff7f0e', linestyle='--', label='Linear Fit')
ax_br.set_xlabel('Concentration (mM)')
ax_br.set_ylabel('Integrated Raman signal')
ax_br.grid(True, linestyle='--', alpha=0.5)
ax_br.text(0.05, 0.90, f'$R^2 = {r_squared:.4f}$', transform=ax_br.transAxes,
           fontsize=10, fontweight='bold', va='top', ha='left')
ax_br.legend(loc='lower right')

# ==========================================
# --- SUBPLOT LABELS
# ==========================================
label_kwargs = {'fontsize': 8, 'ha': 'center', 'va': 'top'}
ax_tl.text(0.5, -0.20, '(a)', transform=ax_tl.transAxes, **label_kwargs)
ax_tr.text(0.5, -0.20, '(b)', transform=ax_tr.transAxes, **label_kwargs)
ax_bl.text(0.5, -0.20, '(c)', transform=ax_bl.transAxes, **label_kwargs)
ax_br.text(0.5, -0.20, '(d)', transform=ax_br.transAxes, **label_kwargs)

print(f'(d) integrated response: slope {slope:,.0f} per mM, R2 {r_squared:.4f}')

save_fig(name=figure_name, path=figure_path)
plt.show()
