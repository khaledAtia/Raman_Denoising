import numpy as np
import matplotlib.pyplot as plt
from JournalPlots.JournalStyles import set_ieee_style,save_fig

set_ieee_style(column='double')
figure_name='ValidateSynthetic'
figure_path=r'C:\Users\khsat\OneDrive\Documents\My papers\Double Headed Unet Model'

# --- 1. Load Data (Using your provided paths) ---
# Note: Ensure these paths are accessible when you run the script locally.
base_path = r'C:\Users\khsat\OneDrive\Documents\python codes\Raman Factory\DoubleHeaded_Unet'

# SNR 5
d_noisy_input5 = np.loadtxt(f'{base_path}\\noisy_input1_5dB.txt') # Corrected filename based on variable name context
d_ground_truth5 = np.loadtxt(f'{base_path}\\ground_truth1_5dB.txt')
d_predicted_raman5 = np.loadtxt(f'{base_path}\\predicted_raman1_5dB.txt')
d_predicted_baseline5 = np.loadtxt(f'{base_path}\\predicted_baseline1_5dB.txt')

# SNR 15
d_noisy_input15 = np.loadtxt(f'{base_path}\\noisy_input1_15dB.txt')
d_ground_truth15 = np.loadtxt(f'{base_path}\\ground_truth1_15dB.txt')
d_predicted_raman15 = np.loadtxt(f'{base_path}\\predicted_raman1_15dB.txt')
d_predicted_baseline15 = np.loadtxt(f'{base_path}\\predicted_baseline1_15dB.txt')

# SNR 25
d_noisy_input25 = np.loadtxt(f'{base_path}\\noisy_input1_25dB.txt')
d_ground_truth25 = np.loadtxt(f'{base_path}\\ground_truth1_25dB.txt')
d_predicted_raman25 = np.loadtxt(f'{base_path}\\predicted_raman1_25dB.txt')
d_predicted_baseline25 = np.loadtxt(f'{base_path}\\predicted_baseline1_25dB.txt')

# Extract vectors (Assuming column 0 is x-axis and column 1 is intensity)
raman_shift = d_noisy_input25[:, 0]

# Organize data into a dictionary or list for easy iteration in the loop
data_map = {
    0: { # SNR 5
        "noisy": d_noisy_input5[:, 1],
        "truth": d_ground_truth5[:, 1],
        "pred_raman": d_predicted_raman5[:, 1],
        "pred_base": d_predicted_baseline5[:, 1],
        "title": "SNR = 5"
    },
    1: { # SNR 15
        "noisy": d_noisy_input15[:, 1],
        "truth": d_ground_truth15[:, 1],
        "pred_raman": d_predicted_raman15[:, 1],
        "pred_base": d_predicted_baseline15[:, 1],
        "title": "SNR = 15"
    },
    2: { # SNR 25
        "noisy": d_noisy_input25[:, 1],
        "truth": d_ground_truth25[:, 1],
        "pred_raman": d_predicted_raman25[:, 1],
        "pred_base": d_predicted_baseline25[:, 1],
        "title": "SNR = 25"
    }
}

# --- 2. Main Plotting Setup ---

# Create figure setup
# figsize=(8, 7) is generally wide enough to span two columns in a journal
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8, 7), sharex=True, sharey='row')

# Define styles
style_grey = {'color': 'grey', 'linewidth': 1.5, 'alpha': 0.7}
style_color = {'color': '#FF4B33', 'linewidth': 1.5,'linestyle': '--'} # Reddish-orange
style_blue = {'color': '#1f77b4', 'linewidth': 1.5}  # Standard matplotlib blue for contrast if needed

# --- 3. Plotting Loop ---
for col_idx in range(3):
    d = data_map[col_idx]
    
    ax_top = axes[0, col_idx]
    ax_mid = axes[1, col_idx]
    ax_bot = axes[2, col_idx]

    # --- Top Panel: Noisy Input vs Predicted Baseline ---
    ax_top.plot(raman_shift, d["noisy"], label='Noisy Input', **style_grey)
    ax_top.plot(raman_shift, d["pred_base"], label='Predicted Baseline', **style_color)
    
    # --- Middle Panel: (Noisy - Baseline) vs Ground Truth ---
    # Calculate the subtracted signal
    corrected_noisy = d["noisy"] - d["pred_base"]
    
    ax_mid.plot(raman_shift, corrected_noisy, label='Input - Baseline', **style_grey)
    ax_mid.plot(raman_shift, d["truth"], label='Ground Truth', **style_color)

    # --- Bottom Panel: Predicted Raman vs Ground Truth ---
    ax_bot.plot(raman_shift, d["truth"], label='Ground Truth', **style_grey)
    ax_bot.plot(raman_shift, d["pred_raman"], label='Predicted Raman', **style_color)

    # --- Column Formatting ---
    # Set column title only on the top row
    ax_top.set_title(d["title"], fontsize=12, fontweight='bold', pad=15)

# --- 4. General Figure Formatting ---

for r in range(3):
    for c in range(3):
        ax = axes[r, c]
        
        # Add legends with specific styling
        # We use a small font size to prevent overcrowding
        leg = ax.legend(loc='upper left', frameon=False, fontsize=8, handlelength=1.0)
        
        # Clean up axes spines (remove top and right boxes)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Remove Y-ticks and labels to keep it clean (as per original image style)
        ax.set_yticks([])
        
        # Set Y-axis label only on the left column
        if c == 0:
            ax.set_ylabel("Intensity (a.u.)", fontsize=11, labelpad=10)

    # Set X-axis label only on the bottom row
    axes[2, r].set_xlabel(r"Raman Shift (cm$^{-1}$)", fontsize=11)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.05) # Tight spacing

# Save or show
# plt.savefig("raman_model_results.png", dpi=300, bbox_inches='tight')
#save_fig(name=figure_name,path=figure_path)
plt.show()