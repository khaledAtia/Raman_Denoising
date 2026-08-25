import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import csv
import os
# Import your custom modules

from Pmodel import AUSequentialUNet,ContrastiveLoss,EarlyStopping,visualize_spatial_gates,ContrastiveLoss_nomask
# Assuming the generator class is in 'generator.py'
from RamanDataGenerator import RamanDataGenerator,FixedRamanDataset

import torch.nn.functional as F



def plot_worst_case_scenario(model, baseline_file, noise_file, device):
    """
    Forces the generator to create an SNR=5.0 spectrum, runs inference, 
    and plots the physical prediction against the ground truth.
    """
    print("Generating SNR=5.0 Worst-Case Visual...")
    model.eval()
    
    # 1. Force the generator to produce ONLY SNR=5.0
    # We set epoch_size=10 just to get a small pool to randomly pick from
    extreme_gen = RamanDataGenerator(
        BASELINE_FILE=baseline_file, 
        NOISE_FILE=noise_file, 
        epoch_size=10, 
        min_snr=5.0, 
        max_snr=5.0
    )
    
    # Batch size 1 makes it easy to grab a single spectrum
    extreme_loader = DataLoader(extreme_gen, batch_size=1, shuffle=True)
    
    # 2. Grab exactly one random highly degraded spectrum
    inputs, target_s, target_b = next(iter(extreme_loader))
    inputs = inputs.to(device)
    
    # 3. Run Inference (Remembering to safely unpack all 8 outputs!)
    with torch.no_grad():
        pred_s, pred_b, _, _, _, _, _, _ = model(inputs)
        
    # 4. Strip everything down to 1D numpy arrays for plotting
    raw_input = inputs.cpu().squeeze().numpy()
    true_sig = target_s.cpu().squeeze().numpy()
    true_base = target_b.cpu().squeeze().numpy()
    
    pred_sig = pred_s.cpu().squeeze().numpy()
    pred_base = pred_b.cpu().squeeze().numpy()
    
    # Use the generator's x-axis if available, otherwise just use pixel indices
    if hasattr(extreme_gen, 'raman_shift'):
        x_axis = extreme_gen.raman_shift
        x_label = "Raman Shift (cm⁻¹)"
    else:
        x_axis = np.arange(len(raw_input))
        x_label = "Pixel Index"

    # ==========================================
    # 5. CREATE THE PUBLICATION-READY PLOT
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=150)
    
    # Top Panel: The Macroscopic View (Raw Data vs Baselines)
    ax1.plot(x_axis, raw_input, color='lightgray', linewidth=1.5, label='Raw Input (SNR=5.0)')
    ax1.plot(x_axis, true_base, color='black', linestyle='--', linewidth=2, label='True Baseline (Ground Truth)')
    ax1.plot(x_axis, pred_base, color='dodgerblue', linewidth=2, alpha=0.8, label='Predicted Baseline')
    
    ax1.set_title("Macroscopic View: Baseline Extraction from Severe Shot Noise", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Normalized Intensity", fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Bottom Panel: The Microscopic View (Raman Signal Extraction)
    ax2.plot(x_axis, true_sig, color='black', linestyle='--', linewidth=2, label='True Raman Signal')
    ax2.plot(x_axis, pred_sig, color='crimson', linewidth=2, alpha=0.8, label='Predicted Raman Signal')
    
    # Optional: Fill the area under the predicted peak to make it visually pop
    ax2.fill_between(x_axis, 0, pred_sig, color='crimson', alpha=0.2)
    
    ax2.set_title("Microscopic View: Disentangled Raman Peak Reconstruction", fontsize=14, fontweight='bold')
    ax2.set_xlabel(x_label, fontsize=12)
    ax2.set_ylabel("Signal Intensity", fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('SNR5_Visual_Validation.png', dpi=300)
    plt.show()


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def evaluate_snr_robustness(model, baseline_file, noise_file, device, batch_size=64):
    """
    Evaluates the model across fixed SNR levels, plots the metrics (MSE, MAE, Cosine)
    with standard deviation error bars, and saves the raw data to a CSV file.
    """
    print("Starting SNR Robustness Evaluation with Variance Tracking...")
    model.eval()
    
    # Define the SNR steps: 5.0 to 25.0 inclusive, step 2.5
    snr_levels = np.arange(5.0, 27.5, 2.5)
    
    # Trackers for Mean and Standard Deviation
    avg_mse_results, std_mse_results = [], []
    avg_mae_results, std_mae_results = [], []
    avg_cos_results, std_cos_results = [], []
    
    n_samples = 500 

    with torch.no_grad():
        for current_snr in snr_levels:
            print(f"Evaluating SNR: {current_snr:.1f}...", end=" ")
            
            # Lock the generator to a single exact SNR
            val_gen = RamanDataGenerator(
                baseline_file_path=baseline_file, 
                noise_file_path=noise_file, 
                epoch_size=n_samples, 
                min_snr=current_snr, 
                max_snr=current_snr,
                blank_ratio=0.0 
            )
            val_dataloader = DataLoader(val_gen, batch_size=batch_size, shuffle=False)
            
            # Temporary lists to hold every single sample's metric for this SNR
            snr_mse_list = []
            snr_mae_list = []
            snr_cos_list = []
            
            for inputs, target_s, target_b in val_dataloader:
                inputs = inputs.to(device)
                target_s = target_s.to(device)
                
                # Unpack all 8 items, only grab pred_s
                pred_s, pred_b, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig = model(inputs)
                
                b_size = inputs.size(0)
                
                # --- PER-SAMPLE METRICS ---
                # reduction='none' keeps the batch dimension intact so we can evaluate variance
                
                # MSE: Shape [Batch, Channels, Length] -> Flatten to [Batch, -1] -> Mean per sample
                mse_per_sample = F.mse_loss(pred_s, target_s, reduction='none').view(b_size, -1).mean(dim=1)
                
                # MAE: Shape [Batch, Channels, Length] -> Flatten to [Batch, -1] -> Mean per sample
                mae_per_sample = F.l1_loss(pred_s, target_s, reduction='none').view(b_size, -1).mean(dim=1)
                
                # Cosine Similarity calculates per-sample along dim=1 automatically
                cos_per_sample = F.cosine_similarity(pred_s.flatten(1), target_s.flatten(1), dim=1)
                
                # Move to CPU and append to our SNR tracker lists
                snr_mse_list.extend(mse_per_sample.cpu().numpy())
                snr_mae_list.extend(mae_per_sample.cpu().numpy())
                snr_cos_list.extend(cos_per_sample.cpu().numpy())
                
            # Calculate Mean and STD for this SNR step
            final_mse_mean = np.mean(snr_mse_list)
            final_mse_std = np.std(snr_mse_list)
            
            final_mae_mean = np.mean(snr_mae_list)
            final_mae_std = np.std(snr_mae_list)
            
            final_cos_mean = np.mean(snr_cos_list)
            final_cos_std = np.std(snr_cos_list)
            
            # Store them globally
            avg_mse_results.append(final_mse_mean)
            std_mse_results.append(final_mse_std)
            
            avg_mae_results.append(final_mae_mean)
            std_mae_results.append(final_mae_std)
            
            avg_cos_results.append(final_cos_mean)
            std_cos_results.append(final_cos_std)
            
            print(f"MSE: {final_mse_mean:.4f}±{final_mse_std:.4f} | MAE: {final_mae_mean:.4f}±{final_mae_std:.4f} | Cosine: {final_cos_mean:.4f}±{final_cos_std:.4f}")

    # ==========================================
    # PUBLICATION PLOTTING & SAVING (WITH ERROR BARS)
    # ==========================================
    print("\nGenerating 3-Panel Benchmark Figure with Variance...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    # 1. Plot MSE (General Error)
    axes[0].errorbar(snr_levels, avg_mse_results, yerr=std_mse_results, fmt='-o', 
                     color='crimson', linewidth=2, markersize=8, capsize=5, capthick=1.5, ecolor='black')
    axes[0].set_title("General Error vs. Noise Level", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    axes[0].set_ylabel("Mean Squared Error (MSE)", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 2. Plot MAE (Quantitative Error)
    axes[1].errorbar(snr_levels, avg_mae_results, yerr=std_mae_results, fmt='-s', 
                     color='darkorange', linewidth=2, markersize=8, capsize=5, capthick=1.5, ecolor='black')
    axes[1].set_title("Quantitative Error vs. Noise Level", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    axes[1].set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 3. Plot Cosine (Qualitative Shape Error)
    axes[2].errorbar(snr_levels, avg_cos_results, yerr=std_cos_results, fmt='-^', 
                     color='dodgerblue', linewidth=2, markersize=8, capsize=5, capthick=1.5, ecolor='black')
    axes[2].set_title("Qualitative Accuracy vs. Noise Level", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    axes[2].set_ylabel("Cosine Similarity", fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("SNR_Robustness_Benchmark_Extended_STD.png", dpi=300)
    plt.show()

    # Save the raw data cleanly to an expanded CSV
    data_matrix = np.column_stack((
        snr_levels, 
        avg_mse_results, std_mse_results, 
        avg_mae_results, std_mae_results, 
        avg_cos_results, std_cos_results
    ))
    
    np.savetxt(
        "snr_robustness_withSTDbarsdata.csv", 
        data_matrix, 
        delimiter=",",
        header="SNR,MSE_Mean,MSE_Std,MAE_Mean,MAE_Std,Cosine_Mean,Cosine_Std", 
        comments=""
    )
    print("Benchmark complete. Data saved to snr_robustness_withSTDbarsdata.csv")

    return snr_levels, avg_mse_results, std_mse_results, avg_mae_results, std_mae_results, avg_cos_results, std_cos_results


def zevaluate_snr_robustness(model, baseline_file, noise_file, device, batch_size=64):
    """
    Evaluates the model across fixed SNR levels, plots the metrics (MSE, MAE, Cosine), 
    and saves the raw data to a 4-column CSV file.
    """
    print("Starting SNR Robustness Evaluation...")
    model.eval()
    
    # Define the SNR steps: 5.0 to 25.0 inclusive, step 2.5
    snr_levels = np.arange(5.0, 27.5, 2.5)
    
    avg_mse_results = []
    avg_mae_results = []  # NEW: Array to store MAE
    avg_cos_results = []
    
    n_samples = 500 

    with torch.no_grad():
        for current_snr in snr_levels:
            print(f"Evaluating SNR: {current_snr:.1f}...", end=" ")
            
            # Lock the generator to a single exact SNR
            val_gen = RamanDataGenerator(
                baseline_file_path=baseline_file, 
                noise_file_path=noise_file, 
                epoch_size=n_samples, 
                min_snr=current_snr, 
                max_snr=current_snr,
                blank_ratio=0.0 
            )
            val_dataloader = DataLoader(val_gen, batch_size=batch_size, shuffle=False)
            
            total_mse = 0.0
            total_mae = 0.0  # NEW: Tracking total batch MAE
            total_cos_sim = 0.0
            total_items = 0
            
            for inputs, target_s, target_b in val_dataloader:
                inputs = inputs.to(device)
                target_s = target_s.to(device)
                
                # Unpack all 8 items, only grab pred_s
                pred_s, pred_b, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig = model(inputs)
                
                b_size = inputs.size(0)
                
                # Metrics
                batch_mse = F.mse_loss(pred_s, target_s).item()
                batch_mae = F.l1_loss(pred_s, target_s).item()  # NEW: Calculate MAE
                batch_cos = F.cosine_similarity(pred_s.flatten(1), target_s.flatten(1)).mean().item()
                
                total_mse += batch_mse * b_size
                total_mae += batch_mae * b_size  # NEW: Accumulate MAE
                total_cos_sim += batch_cos * b_size
                total_items += b_size
                
            final_mse = total_mse / total_items
            final_mae = total_mae / total_items  # NEW: Average MAE
            final_cos = total_cos_sim / total_items
            
            avg_mse_results.append(final_mse)
            avg_mae_results.append(final_mae)    # NEW: Store MAE
            avg_cos_results.append(final_cos)
            
            # Print the updated log
            print(f"MSE: {final_mse:.4f} | MAE: {final_mae:.4f} | Cosine: {final_cos:.4f}")

    # ==========================================
    # PUBLICATION PLOTTING & SAVING
    # ==========================================
    print("\nGenerating 3-Panel Benchmark Figure...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    # 1. Plot MSE (General Error)
    axes[0].plot(snr_levels, avg_mse_results, marker='o', color='crimson', linewidth=2, markersize=8)
    axes[0].set_title("General Error vs. Noise Level", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    axes[0].set_ylabel("Mean Squared Error (MSE)", fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 2. Plot MAE (Quantitative Error)
    axes[1].plot(snr_levels, avg_mae_results, marker='s', color='darkorange', linewidth=2, markersize=8)
    axes[1].set_title("Quantitative Error vs. Noise Level", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    axes[1].set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)

    # 3. Plot Cosine (Qualitative Shape Error)
    axes[2].plot(snr_levels, avg_cos_results, marker='^', color='dodgerblue', linewidth=2, markersize=8)
    axes[2].set_title("Qualitative Accuracy vs. Noise Level", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    axes[2].set_ylabel("Cosine Similarity", fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("SNR_Robustness_Benchmark_Extended.png", dpi=300)
    plt.show()

    # Save the raw data cleanly to a CSV
    data_matrix = np.column_stack((snr_levels, avg_mse_results, avg_mae_results, avg_cos_results))
    np.savetxt(
        "snr_robustness_withSTDbarsdata.csv", 
        data_matrix, 
        delimiter=",",
        header="SNR,MSE,MAE,Cosine_Similarity", 
        comments=""
    )
    print("Benchmark complete. Data saved to snr_robustness_data.csv")

    return snr_levels, avg_mse_results, avg_mae_results, avg_cos_results

    # ==========================================
    # SAVE RAW DATA TO TXT FILE
    # ==========================================
    output_filename = 'snr_robustness_results.txt'
    print(f"\nSaving numerical results to {output_filename}...")
    
    with open(output_filename, 'w') as f:
        # Write the header row
        f.write("SNR\tAvg_MSE\tAvg_Cosine_Similarity\n")
        
        # Write the data rows
        for snr, mse, cos in zip(snr_levels, avg_mse_results, avg_cos_results):
            f.write(f"{snr:.1f}\t{mse:.6f}\t{cos:.6f}\n")

    # ==========================================
    # PLOTTING THE BENCHMARK FIGURES
    # ==========================================
    print("Plotting results...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(snr_levels, avg_mse_results, marker='o', linestyle='-', color='crimson', linewidth=2, markersize=8)
    ax1.set_title("Signal Reconstruction Error vs. Noise Level", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    ax1.set_ylabel("Average Mean Squared Error (MSE)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(snr_levels, avg_cos_results, marker='s', linestyle='-', color='dodgerblue', linewidth=2, markersize=8)
    ax2.set_title("Signal Shape Preservation vs. Noise Level", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=12)
    ax2.set_ylabel("Average Cosine Similarity", fontsize=12)
    ax2.set_ylim(min(avg_cos_results) - 0.05, 1.01) 
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('SNR_Robustness_Benchmark.png', dpi=300)
    plt.show()
    
    return snr_levels, avg_mse_results, avg_cos_results


def update_parameter(current_val, final_value, current_epoch, start_epoch, end_epoch):
    """
    Calculates the linearly decayed value iteratively based on the remaining steps.
    """
    # Phase 1: Before decay starts
    if current_epoch <= start_epoch:
        return current_val
        
    # Phase 3: After decay finishes (or is currently finishing)
    if current_epoch >= end_epoch:
        return final_value
        
    # Phase 2: Active decay phase
    # Calculate how many steps are left in the decay window (including this one)
    remaining_steps = end_epoch - current_epoch + 1
    
    # Calculate the step size needed to close the remaining gap evenly
    step_size = (current_val - final_value) / remaining_steps
    
    # Subtract the step size to get the updated value
    return current_val - step_size


def get_decayed_value(initial_value, final_value,current_epoch, start_epoch, end_epoch):
    """
    Calculates the linearly decayed value for a given epoch.
    """
    # Phase 1: Before decay starts
    if current_epoch <= start_epoch:
        return initial_value
        
    # Phase 3: After decay finishes
    if current_epoch >= end_epoch:
        return final_value
        
    # Phase 2: Active decay phase
    decay_steps = end_epoch - start_epoch
    steps_taken = current_epoch - start_epoch
    
    # Calculate what percentage of the decay phase we have completed (from 0.0 to 1.0)
    progress = steps_taken / decay_steps
    
    # Apply that percentage to the total amount we need to decay
    total_decay_amount = initial_value - final_value
    current_value = initial_value - (progress * total_decay_amount)
    
    return current_value




# --- Configuration ---
#BASELINE_FILE = r'C:\Users\khsat\OneDrive\Documents\python codes\Raman Factory\Raman Data\Leslie F. (40-60s)20\balackWell_background\baseline_combined_data.txt' # Ensure this file exists!
BASELINE_FILE = r'C:\Users\khsat\Documents\python codes\Raman Factory\DoubleHeaded_Unet\data\baseline_data.npz' # Ensure this file exists!

NOISE_FILE =r'C:\Users\khsat\Documents\python codes\Raman Factory\DoubleHeaded_Unet\data\noise_data.npz'
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zztrain_model():
    # --- 1. Setup Data ---
    print("Initializing Data Generators...")
    
    train_gen = RamanDataGenerator(BASELINE_FILE, NOISE_FILE, epoch_size=100*BATCH_SIZE, min_snr=4, max_snr=60)
    train_dataloader = DataLoader(train_gen, batch_size=BATCH_SIZE, shuffle=True)
    
    temp_val_gen = RamanDataGenerator(BASELINE_FILE, NOISE_FILE, min_snr=4, max_snr=26)
    val_dataset = FixedRamanDataset(temp_val_gen, n_samples=2000)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {DEVICE} with {train_gen.n_points} points per spectrum.")
    print(f"Split: Dynamic Training Set ({len(train_gen)}/epoch) | Fixed Validation Set ({len(val_dataset)})")

    # --- 2. Setup Model ---
    gamma = 1.0
    kernels = [1, 1, 3]
    ks = "".join(map(str, kernels))
    model = AUSequentialUNet(n_channels=1, bilinear=False, gamma=gamma, kernels=kernels, use_derivatives=False).to(DEVICE)
    
    # --- 3. Setup Loss Configurations ---
    w_signal = 20.0
    w_base = 20.0
    w_consist = 0.0   
    w_smooth = 0.2
    w_curve = 0.2
    w_shape = 10.0
    d_factor = 0.0    
    w_mid = 2.0      
    w_deep = 0.5      
    w_ortho = 0.5     # Latent Space Disentanglement Penalty
    
    val_total_weight = w_signal + w_base + w_consist + w_smooth + w_curve + w_shape + d_factor  
    train_total_weight = val_total_weight + w_mid + w_deep
    
    # Optimizer (Reverted to standard tracking)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    patience=15,    
                    factor=0.5, 
                    min_lr=1e-8     
                )
                
    early_stopping = EarlyStopping(patience=100, start_epoch=250, delta=1e-4, verbose=True, path='best_raman2_model.pt')
    model_name = f"Latent_{ks}_S{w_signal}_B{w_base}_C{w_consist}_wm{w_mid}_wd{w_deep}_ortho{w_ortho}.pth"

    # Instantiate criterion ONCE
    criterion = ContrastiveLoss(
        w_signal=w_signal,   
        w_base=w_base,      
        w_consist=w_consist,  
        w_smooth=w_smooth,    
        w_shape=w_shape,
        w_curve=w_curve,     
        dip_factor=d_factor,
        w_mid=w_mid,        
        w_deep=w_deep,
        w_ortho=w_ortho
    )

    # --- 4. Training Loop ---
    loss_history = []

    for epoch in range(EPOCHS):
        # ==========================
        # TRAINING PHASE (Dynamic)
        # ==========================
        model.train()
        train_epoch_loss = 0.0
        
        for batch_idx, (inputs, target_s, target_b) in enumerate(train_dataloader):
            inputs = inputs.to(DEVICE)
            target_s = target_s.to(DEVICE)
            target_b = target_b.to(DEVICE)

            optimizer.zero_grad()

            pred_s, pred_b, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig = model(inputs)

            # Reverted to capturing standard total_loss
            loss, t_components = criterion(
                pred_s, target_s, pred_b, target_b, inputs, 
                pred_s_mid=pred_s_mid, pred_s_deep=pred_s_deep, 
                pred_b_mid=pred_b_mid, pred_b_deep=pred_b_deep,
                x4_base=x4_base, x4_sig=x4_sig
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_dataloader)
        loss_history.append(avg_train_loss)

        # ==========================
        # VALIDATION PHASE (Fixed)
        # ==========================
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_sig_loss = 0.0
        
        with torch.no_grad():
            for inputs, target_s, target_b in val_dataloader:
                inputs = inputs.to(DEVICE)
                target_s = target_s.to(DEVICE)
                target_b = target_b.to(DEVICE)
                
                pred_s, pred_b, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig = model(inputs)

                v_loss, v_components = criterion(
                    pred_s, target_s, pred_b, target_b, inputs,
                    x4_base=x4_base, x4_sig=x4_sig
                )
                
                val_epoch_loss += v_loss.item()
                val_epoch_sig_loss += v_components[0]
        
        avg_val_loss = val_epoch_loss / len(val_dataloader)
        avg_val_sig_loss = val_epoch_sig_loss / len(val_dataloader)

        # ==========================
        # LOGGING & CHECKS
        # ==========================
        scheduler.step(avg_val_sig_loss)
        current_lr = optimizer.param_groups[0]['lr'] * 1e4
        
        # Normalize losses accurately for display (Using your original manual weight logic)
        norm_train_loss = avg_train_loss / train_total_weight
        norm_val_loss = avg_val_loss / val_total_weight

        print(f"Epoch {epoch+1}/{EPOCHS} | LR {current_lr:.4f} | AuxWt: {(w_mid+w_deep):.1f} | "
              f"Norm Train: {norm_train_loss:.4f} | Norm Val: {norm_val_loss:.4f} | "
              f"Sig: {t_components[0]:.4f} | Val Sig: {avg_val_sig_loss:.4f} | BFit: {t_components[1]:.4f} | "
              f"Rgh: {t_components[2]:.4f} | Crv: {t_components[3]:.4f} | Cst: {t_components[4]:.4f}")

        early_stopping(avg_val_sig_loss, model, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at Epoch {epoch+1}!")
            break

    # --- 5. Finish ---
    print("Training Complete.")
    model.load_state_dict(torch.load('best_raman2_model.pt'))
    print("Loaded best model weights.")
    
    model_path = './models/' + model_name
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_name}")
    
    return model, loss_history

def zzztrain_model(comment=""): # <--- NEW: Comment parameter
    # --- 1. Setup Data ---
    print("Initializing Data Generators...")
    
    train_gen = RamanDataGenerator(BASELINE_FILE, NOISE_FILE, epoch_size=100*BATCH_SIZE, min_snr=4, max_snr=60)
    train_dataloader = DataLoader(train_gen, batch_size=BATCH_SIZE, shuffle=True)
    
    temp_val_gen = RamanDataGenerator(BASELINE_FILE, NOISE_FILE, min_snr=4, max_snr=26)
    val_dataset = FixedRamanDataset(temp_val_gen, n_samples=2000)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {DEVICE} with {train_gen.n_points} points per spectrum.")
    print(f"Split: Dynamic Training Set ({len(train_gen)}/epoch) | Fixed Validation Set ({len(val_dataset)})")

    # --- 2. Setup Model ---
    gamma = 1.0
    kernels = [1, 1, 3]
    ks = "".join(map(str, kernels))
    model = AUSequentialUNet(n_channels=1, bilinear=False, gamma=gamma, kernels=kernels, use_derivatives=False).to(DEVICE)
    
    # --- 3. Setup Loss Configurations ---
    w_signal = 20.0
    w_base = 20.0
    w_consist = 0.0   
    w_smooth = 0.2
    w_curve = 0.2
    
    w_mae = 1.0
    w_cos = 1.0
    w_shape = 0.4
    
    d_factor = 0.0    
    w_mid = 2.0      
    w_deep = 0.5      
    w_ortho = 0.5     
    
    val_total_weight = w_signal + w_base + w_consist + w_smooth + w_curve + w_shape + d_factor  
    train_total_weight = val_total_weight + w_mid + w_deep
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    patience=15,    
                    factor=0.5, 
                    min_lr=1e-8     
                )
                
    early_stopping = EarlyStopping(patience=100, start_epoch=250, delta=1e-4, verbose=True, path='best_raman2_model.pt')
    model_name = f"Latent_{ks}_S{w_signal}_B{w_base}_C{w_consist}_wm{w_mid}_wd{w_deep}_ortho{w_ortho}.pth"

    criterion = ContrastiveLoss(
        w_signal=w_signal,   
        w_base=w_base,      
        w_consist=w_consist,  
        w_smooth=w_smooth,    
        w_curve=w_curve,     
        dip_factor=d_factor,
        w_mae=w_mae,       
        w_cos=w_cos,       
        w_shape=w_shape,   
        w_mid=w_mid,        
        w_deep=w_deep,
        w_ortho=w_ortho
    )

    # --- 4. Training Loop ---
    loss_history = {
        'train_loss': [], 'val_loss': [], 
        'val_mae': [], 'val_cos': [], 'val_shape': []
    }

    for epoch in range(EPOCHS):
        # ==========================
        # TRAINING PHASE 
        # ==========================
        model.train()
        train_epoch_loss = 0.0
        
        for batch_idx, (inputs, target_s, target_b) in enumerate(train_dataloader):
            inputs = inputs.to(DEVICE)
            target_s = target_s.to(DEVICE)
            target_b = target_b.to(DEVICE)

            optimizer.zero_grad()

            pred_s, pred_b, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig = model(inputs)

            loss, t_components = criterion(
                pred_s, target_s, pred_b, target_b, inputs, 
                pred_s_mid=pred_s_mid, pred_s_deep=pred_s_deep, 
                pred_b_mid=pred_b_mid, pred_b_deep=pred_b_deep,
                x4_base=x4_base, x4_sig=x4_sig
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_dataloader)
        loss_history['train_loss'].append(avg_train_loss)

        # ==========================
        # VALIDATION PHASE 
        # ==========================
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_mae = 0.0
        val_epoch_cos = 0.0
        val_epoch_shape = 0.0
        
        with torch.no_grad():
            for inputs, target_s, target_b in val_dataloader:
                inputs = inputs.to(DEVICE)
                target_s = target_s.to(DEVICE)
                target_b = target_b.to(DEVICE)
                
                pred_s, pred_b, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig = model(inputs)

                v_loss, v_components = criterion(
                    pred_s, target_s, pred_b, target_b, inputs,
                    x4_base=x4_base, x4_sig=x4_sig
                )
                
                val_epoch_loss += v_loss.item()
                val_epoch_mae += v_components[0]   
                val_epoch_cos += v_components[1]   
                val_epoch_shape += v_components[2] 
        
        avg_val_loss = val_epoch_loss / len(val_dataloader)
        avg_val_mae = val_epoch_mae / len(val_dataloader)
        avg_val_cos = val_epoch_cos / len(val_dataloader)
        avg_val_shape = val_epoch_shape / len(val_dataloader)

        loss_history['val_loss'].append(avg_val_loss)
        loss_history['val_mae'].append(avg_val_mae)
        loss_history['val_cos'].append(avg_val_cos)
        loss_history['val_shape'].append(avg_val_shape)

        # ==========================
        # LOGGING & CHECKS
        # ==========================
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr'] * 1e4
        
        print(f"Epoch {epoch+1}/{EPOCHS} | LR {current_lr:.4f} | AuxWt: {(w_mid+w_deep):.1f} | "
              f"Val MAE: {avg_val_mae:.4f} | Val Cos: {avg_val_cos:.4f} | Val Shape: {avg_val_shape:.4f} | "
              f"Trn BFit: {t_components[3]:.4f} | Trn Rgh: {t_components[4]:.4f} | Trn Crv: {t_components[5]:.4f}")

        early_stopping(avg_val_mae, model, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at Epoch {epoch+1}!")
            break

    # --- 5. Finish & CSV Logging ---
    print("Training Complete.")
    model.load_state_dict(torch.load('best_raman2_model.pt', weights_only=True))
    print("Loaded best model weights.")
    
    # Save the final model
    os.makedirs('./models', exist_ok=True)
    model_path = os.path.join('./models', model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # ==========================
    # CSV EXPERIMENT LOGGER
    # ==========================
    log_file = "experiment_tracking_log.csv"
    file_exists = os.path.isfile(log_file)
    
    # Extract the exact metrics from the epoch where the model performed best
    best_val_mae = min(loss_history['val_mae'])
    best_epoch_idx = loss_history['val_mae'].index(best_val_mae)
    
    # Gather datetime
    now = datetime.now()
    
    log_data = {
        'Date': now.strftime("%Y-%m-%d"),
        'Time': now.strftime("%H:%M:%S"),
        'Comment': comment,
        'Model_Name': model_name,
        'Best_Epoch': best_epoch_idx + 1,
        'Val_MAE': round(loss_history['val_mae'][best_epoch_idx], 6),
        'Val_Cos': round(loss_history['val_cos'][best_epoch_idx], 6),
        'Val_Shape': round(loss_history['val_shape'][best_epoch_idx], 6),
        'w_mae': w_mae,
        'w_cos': w_cos,
        'w_shape': w_shape,
        'w_signal': w_signal,
        'w_base': w_base,
        'w_consist': w_consist,
        'w_smooth': w_smooth,
        'w_curve': w_curve,
        'w_mid': w_mid,
        'w_deep': w_deep,
        'w_ortho': w_ortho,
        'Learning_Rate': LEARNING_RATE,
        'Batch_Size': BATCH_SIZE
    }
    
    # Append to the CSV file
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()  # Only write headers if file is brand new
        writer.writerow(log_data)
        
    print(f"Experiment logged successfully to {log_file}")
    
    return model, loss_history



def train_model(comment=""): 
    # --- 1. Setup Data ---
    print("Initializing Data Generators...")
    
    train_gen = RamanDataGenerator(BASELINE_FILE, NOISE_FILE, epoch_size=100*BATCH_SIZE, min_snr=4, max_snr=60)
    train_dataloader = DataLoader(train_gen, batch_size=BATCH_SIZE, shuffle=True)
    
    temp_val_gen = RamanDataGenerator(BASELINE_FILE, NOISE_FILE, min_snr=4, max_snr=26,blank_ratio=0.0)
    val_dataset = FixedRamanDataset(temp_val_gen, n_samples=2000)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {DEVICE} with {train_gen.n_points} points per spectrum.")
    print(f"Split: Dynamic Training Set ({len(train_gen)}/epoch) | Fixed Validation Set ({len(val_dataset)})")

    # --- 2. Setup Model ---
    gamma = 1.0
    kernels = [1, 1, 3]
    ks = "".join(map(str, kernels))
    model = AUSequentialUNet(n_channels=1, bilinear=False, gamma=gamma, kernels=kernels,base_latent_dim=16, use_derivatives=False).to(DEVICE)
    
    # --- 3. Setup Loss Configurations ---
    w_signal = 20.0
    w_base = 20.0
    w_consist = 0.0   
    w_smooth = 0.2
    w_curve = 0.2
    
    w_mae = 1.0
    w_cos = 1.0
    w_shape = 0.1
    
    d_factor = 0.0    
    w_mid = 2.0      
    w_deep = 1.0     
    w_ortho = 0.5    
    
    val_total_weight = w_signal + w_base + w_consist + w_smooth + w_curve + w_shape + d_factor  
    train_total_weight = val_total_weight + w_mid + w_deep
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    patience=15,    
                    factor=0.5, 
                    min_lr=1e-8     
                )
                
    early_stopping = EarlyStopping(patience=100, start_epoch=250, delta=1e-4, verbose=True, path='best_raman2_model.pt')
    model_name = f"Pmod_nosmoothBase_{ks}_S{w_signal}_B{w_base}_wm{w_mid}_wd{w_deep}_ortho{w_ortho}_wmae{w_mae}_wcos{w_cos}_wshape{w_shape}.pth"

    criterion = ContrastiveLoss(
        w_signal=w_signal,   
        w_base=w_base,      
        w_consist=w_consist,  
        w_smooth=w_smooth,    
        w_curve=w_curve,     
        dip_factor=d_factor,
        w_mae=w_mae,       
        w_cos=w_cos,       
        w_shape=w_shape,   
        w_mid=w_mid,        
        w_deep=w_deep,
        w_ortho=w_ortho
    )

    # --- 4. Training Loop ---
    # NEW: Added trackers for auxiliary and orthogonality losses
    loss_history = {
        'train_loss': [], 'val_loss': [], 
        'val_mae': [], 'val_cos': [], 'val_shape': [],
        'train_aux_s': [], 'train_aux_b': [], 'train_ortho': [] 
    }

    for epoch in range(EPOCHS):
        # ==========================
        # TRAINING PHASE 
        # ==========================
        model.train()
        train_epoch_loss = 0.0
        train_epoch_aux_s = 0.0
        train_epoch_aux_b = 0.0
        train_epoch_ortho = 0.0
        
        for batch_idx, (inputs, target_s, target_b) in enumerate(train_dataloader):
            inputs = inputs.to(DEVICE)
            target_s = target_s.to(DEVICE)
            target_b = target_b.to(DEVICE)

            optimizer.zero_grad()

            pred_s, pred_b, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig = model(inputs)

            loss, t_components = criterion(
                pred_s, target_s, pred_b, target_b, inputs, 
                pred_s_mid=pred_s_mid, pred_s_deep=pred_s_deep, 
                pred_b_mid=pred_b_mid, pred_b_deep=pred_b_deep,
                x4_base=x4_base, x4_sig=x4_sig
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_epoch_loss += loss.item()
            train_epoch_aux_s += t_components[7] # Deep Signal 
            train_epoch_aux_b += t_components[8] # Deep Baseline
            train_epoch_ortho += t_components[9] # Orthogonality

        num_batches = len(train_dataloader)
        loss_history['train_loss'].append(train_epoch_loss / num_batches)
        loss_history['train_aux_s'].append(train_epoch_aux_s / num_batches)
        loss_history['train_aux_b'].append(train_epoch_aux_b / num_batches)
        loss_history['train_ortho'].append(train_epoch_ortho / num_batches)

        # ==========================
        # VALIDATION PHASE 
        # ==========================
        model.eval()
        val_epoch_loss = 0.0
        val_epoch_mae = 0.0
        val_epoch_cos = 0.0
        val_epoch_shape = 0.0
        
        with torch.no_grad():
            for inputs, target_s, target_b in val_dataloader:
                inputs = inputs.to(DEVICE)
                target_s = target_s.to(DEVICE)
                target_b = target_b.to(DEVICE)
                
                pred_s, pred_b, pred_s_mid, pred_s_deep, pred_b_mid, pred_b_deep, x4_base, x4_sig = model(inputs)

                v_loss, v_components = criterion(
                    pred_s, target_s, pred_b, target_b, inputs,
                    x4_base=x4_base, x4_sig=x4_sig
                )
                
                val_epoch_loss += v_loss.item()
                val_epoch_mae += v_components[0]   
                val_epoch_cos += v_components[1]   
                val_epoch_shape += v_components[2] 
        
        avg_val_loss = val_epoch_loss / len(val_dataloader)
        avg_val_mae = val_epoch_mae / len(val_dataloader)
        avg_val_cos = val_epoch_cos / len(val_dataloader)
        avg_val_shape = val_epoch_shape / len(val_dataloader)

        loss_history['val_loss'].append(avg_val_loss)
        loss_history['val_mae'].append(avg_val_mae)
        loss_history['val_cos'].append(avg_val_cos)
        loss_history['val_shape'].append(avg_val_shape)

        # ==========================
        # LOGGING & CHECKS
        # ==========================
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr'] * 1e4
        
        # Updated print statement to show Orthogonality and Auxiliary Signal metrics
        print(f"Epoch {epoch+1}/{EPOCHS} | LR {current_lr:.4f} | AuxWt: {(w_mid+w_deep):.1f} | "
              f"Val MAE: {avg_val_mae:.4f} | Val Cos: {avg_val_cos:.4f} | Val Shape: {avg_val_shape:.4f} | "
              f"Ortho: {loss_history['train_ortho'][-1]:.4f} | AuxS: {loss_history['train_aux_s'][-1]:.4f}")

        early_stopping(avg_val_mae, model, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at Epoch {epoch+1}!")
            break

    # --- 5. Finish & Save Models ---
    print("Training Complete.")
    model.load_state_dict(torch.load('best_raman2_model.pt', weights_only=True))
    print("Loaded best model weights.")
    
    os.makedirs('./models', exist_ok=True)
    model_path = os.path.join('./models', model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Gather datetime for both plots and logs
    now = datetime.now()

    # ==========================
    # GENERATE DYNAMICS PLOT
    # ==========================
    print("Generating Deep Supervision & Disentanglement Plot...")
    plt.figure(figsize=(10, 6), dpi=150)
    
    trained_epochs = range(1, len(loss_history['train_ortho']) + 1)

    data_matrix = np.column_stack((trained_epochs, loss_history['train_ortho'], loss_history['train_aux_s'], loss_history['val_mae'],loss_history['val_cos'],loss_history['val_shape'],loss_history['train_loss'],loss_history['val_loss']))
    
    np.savetxt(
        "training_history.csv", 
        data_matrix, 
        delimiter=",",
        header="epoches,train_ortho,train_aux_s,val_mae,val_cos,val_shape,train_loss,val_loss", 
        comments=""
    )

    plt.plot(trained_epochs, loss_history['train_ortho'], label='Latent Orthogonality Penalty', color='purple', linewidth=2)
    plt.plot(trained_epochs, loss_history['train_aux_s'], label='Deep Supervision (Signal)', color='crimson', linewidth=2, linestyle='--')
    plt.plot(trained_epochs, loss_history['train_aux_b'], label='Deep Supervision (Baseline)', color='dodgerblue', linewidth=2, linestyle='--')
    
    plt.title("Latent Space & Deep Supervision Dynamics", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.yscale('log') 
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plot_path = f"Latent_Dynamics_{now.strftime('%H%M%S')}.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved as {plot_path}")
    plt.close() # Close plot to free memory
    
    # ==========================
    # CSV EXPERIMENT LOGGER
    # ==========================
    log_file = "experiment_tracking_log.csv"
    file_exists = os.path.isfile(log_file)
    
    best_val_mae = min(loss_history['val_mae'])
    best_epoch_idx = loss_history['val_mae'].index(best_val_mae)
    
    log_data = {
        'Date': now.strftime("%Y-%m-%d"),
        'Time': now.strftime("%H:%M:%S"),
        'Comment': comment,
        'Model_Name': model_name,
        'Best_Epoch': best_epoch_idx + 1,
        'Val_MAE': round(loss_history['val_mae'][best_epoch_idx], 6),
        'Val_Cos': round(loss_history['val_cos'][best_epoch_idx], 6),
        'Val_Shape': round(loss_history['val_shape'][best_epoch_idx], 6),
        'w_mae': w_mae,
        'w_cos': w_cos,
        'w_shape': w_shape,
        'w_signal': w_signal,
        'w_base': w_base,
        'w_consist': w_consist,
        'w_smooth': w_smooth,
        'w_curve': w_curve,
        'w_mid': w_mid,
        'w_deep': w_deep,
        'w_ortho': w_ortho,
        'Learning_Rate': LEARNING_RATE,
        'Batch_Size': BATCH_SIZE
    }
    
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()  
        writer.writerow(log_data)
        
    print(f"Experiment logged successfully to {log_file}")
    
    return model, loss_history




def visualize_prediction(model, dataset, epoch):
    """ Runs a specific defined spectrum through model and plots results """
    model.eval()
    
    # Generate a tricky defined spectrum (Weak Signal)
    # Using specific peaks to see if model keeps them
    peaks_cm = [1000, 1600] # Example positions
    inputs, t_s, t_b = dataset.generate_defined_spectrum(
        peak_locs_cm=peaks_cm, 
        widths_idx=[10, 15], 
        amplitudes=[1.0, 0.5], 
        eta=0.9, 
        snr=10
    )
    
    # Prepare for model (Normalize & Tensor)
    # Note: Generator usually returns numpy, we need to replicate normalization logic manually 
    # or rely on the generator's __getitem__ if we used that.
    # Here we do quick manual normalization for the plot
    raw_input = inputs
    min_v, max_v = np.min(raw_input), np.max(raw_input)
    norm_input = (raw_input - min_v) / (max_v - min_v)
    
    # Convert to Tensor (1, 1, 864)
    x_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        p_s, p_b = model(x_tensor)
        
    # Convert back to numpy
    pred_signal = p_s.cpu().squeeze().numpy()
    pred_baseline = p_b.cpu().squeeze().numpy()
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(dataset.raman_shift, norm_input, color='gray', alpha=0.5, label='Input (Noisy)')
    plt.plot(dataset.raman_shift, pred_signal, color='red', label='Pred: Signal')
    plt.plot(dataset.raman_shift, pred_baseline, color='blue', linestyle='--', label='Pred: Baseline')
    plt.title(f"Validation Check - Epoch {epoch}")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    #model, loss_history=train_model(comment="RK4 in encoder only, no smoothing in the baseline branch, previously we had one smoothing")

    # 2. Grab the validation dataloader logic to use for visualization
    #temp_val_gen = RamanDataGenerator(BASELINE_FILE, NOISE_FILE, min_snr=4, max_snr=26)
    #val_dataset = FixedRamanDataset(temp_val_gen, n_samples=20) # Just need a few samples
    #val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # 3. Extract and plot the heatmaps!
    #visualize_spatial_gates(model, val_dataloader, DEVICE, num_samples=3)

    gamma = 1.0
    kernels = [1, 1, 3]

    model = AUSequentialUNet(n_channels=1, bilinear=False, gamma=gamma, kernels=kernels,base_latent_dim=16, use_derivatives=False).to(DEVICE)
    
    MODEL_NAME='Pmod_nosmoothBase_113_S20.0_B20.0_wm2.0_wd1.0_ortho0.5_wmae1.0_wcos1.0_wshape0.1.pth'
    MODEL_PATH="./models/"+MODEL_NAME
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Train the model first!")
        

    model.eval()


    snr_levels, mses,maes, cosines = evaluate_snr_robustness(
    model=model, 
    baseline_file=BASELINE_FILE, 
    noise_file=NOISE_FILE, 
    device=DEVICE,
    batch_size=64
    )
