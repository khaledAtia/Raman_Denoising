import torch
import numpy as np
import matplotlib.pyplot as plt
from Pmodel import AUSequentialUNet
from RamanDataGenerator import RamanDataGenerator

# --- CONFIG ---
MODEL_NAME='Pmod_nosmoothBase_113_S20.0_B20.0_wm2.0_wd1.0_ortho0.5_wmae1.0_wcos1.0_wshape0.1.pth'
MODEL_PATH="./models/"+MODEL_NAME






BASELINE_FILE = r'C:\Users\khsat\Documents\python codes\Raman Factory\DoubleHeaded_Unet\data\baseline_data.npz' # Ensure this file exists!

NOISE_FILE =r'C:\Users\khsat\Documents\python codes\Raman Factory\DoubleHeaded_Unet\data\noise_data.npz'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def compare_results():
    SNR=25
    if SNR==5:
        peak_locs_cm=[400, 602, 810, 1300, 1400]  # Standard biological peaks
        widths_idx=[5, 10, 18, 20, 30] 
        amplitudes=[0.4, 1.0, 0.3, 0.8, 0.5] 
    elif SNR==15:
        peak_locs_cm=[380, 405,560,612,812,905,1050 ,1300, 1720]  # Standard biological peaks
        widths_idx=[4,6,15,11,25, 8,12, 18, 28] 
        amplitudes=np.random.uniform(0.2,1,9) 
    elif SNR==25:
        peak_locs_cm=[360, 405,560,612,735,805,901,1070 ,1290,1480,1610, 1730]  # Standard biological peaks
        widths_idx=[6,12,5,14,8,20, 9,20, 11, 14, 26]  
        amplitudes=np.random.uniform(0.2,1,12) 
    else:
        peak_locs_cm=[400, 602, 810, 1300, 1400]  # Standard biological peaks
        widths_idx=[5, 10, 18, 20, 30] 
        amplitudes=[0.4, 1.0, 0.3, 0.8, 0.5] 


    # 1. Load Model
    #model = TwoHeadedUNet(n_channels=1).to(DEVICE)
    model = AUSequentialUNet(n_channels=1,bilinear=False,gamma=1.0,use_derivatives=False,kernels=[1,1,3],base_latent_dim=16).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found. Train the model first!")
        return

    model.eval()

    # 2. Setup Generator
    gen = RamanDataGenerator(BASELINE_FILE,noise_file_path=NOISE_FILE)

    # 3. Generate a "Tricky" Test Spectrum
    # We purposefully add a strong baseline and noise (SNR=10)
    # We define peaks at specific locations to see if the model keeps them.
    print("Generating test spectrum...")
    
    inputs, target_s, target_b = gen.generate_defined_spectrum(
        peak_locs_cm=peak_locs_cm,  # Standard biological peaks
        widths_idx=widths_idx, 
        amplitudes=amplitudes,        # Mix of strong and weak peaks
        eta=0.9,                                     # Mostly Lorentzian
        snr=SNR                           # Moderate noise
    )

    # 4. Normalize (Same logic as training)
    # We must normalize because the model learned on 0-1 data
    min_v, max_v = np.min(inputs), np.max(inputs)
    
    # Avoid division by zero
    if max_v - min_v == 0: max_v += 1e-6
        
    norm_input = (inputs - min_v) / (max_v - min_v)
    norm_target_s = (target_s) / (max_v - min_v)  # Scale target by SAME factor
    
    # 5. Run Inference
    x_tensor = torch.tensor(norm_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        pred_s, pred_b,_,pred_s_deep,_,_,_,_ = model(x_tensor)

    # 6. Extract Data for Plotting
    pred_s_deep=pred_s_deep/10
    pred_s=pred_s/10

    pred_raman_deep=pred_s_deep.cpu().squeeze().numpy()
    prediction = pred_s.cpu().squeeze().numpy()
    pred_raman=prediction* (max_v - min_v)
    pred_deep=pred_raman_deep* (max_v - min_v)
    print('deep :',np.sum(pred_deep))
    ground_truth = norm_target_s* (max_v - min_v)
    print('truth :',np.sum(norm_target_s))
    residual = ground_truth - pred_raman
    print('prediction :',np.sum(prediction))
    pred_baseline=pred_b.cpu().squeeze().numpy()
    pred_baseline=pred_baseline* (max_v - min_v)+ min_v
    # 7. Visualization
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Panel A: The Input (What the model saw)
    ax[0].plot(gen.raman_shift, inputs, color='gray', alpha=0.6, label='Noisy Input (Raw)')
    ax[0].plot(gen.raman_shift,pred_baseline , color='blue', linestyle='--', label='Predicted Baseline')
    ax[0].set_title("Input Data & Baseline Removal")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Panel B: The Comparison (The result you asked for)
    ax[1].plot(gen.raman_shift, ground_truth, color='black', linewidth=2, label='Ground Truth (Clean)')
    ax[1].plot(gen.raman_shift, pred_raman, color='red', linestyle='--', linewidth=1.5, label='Model Prediction')
    ax[1].set_title("Denoising Performance: Truth vs. Prediction")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Panel C: The Residual (The Error)
    ax[2].plot(gen.raman_shift, residual, color='purple', label='Difference (Truth - Pred)')
    ax[2].set_title("Residual Error (Ideal = Flat Line at 0)")
    ax[2].set_ylim([-0.2, 0.2]) # Zoom in to see small errors
    ax[2].axhline(0, color='black', linestyle=':')
    ax[2].legend()
    ax[2].grid(True, alpha=0.3)
    ax[2].set_xlabel("Raman Shift (cm$^{-1}$)")

    plt.tight_layout()
    plt.show()

    # 8. Quantitative Metrics
    combined_noisyInput = np.column_stack((gen.raman_shift, inputs))
    combined_predictedBaseline = np.column_stack((gen.raman_shift, pred_baseline))
    combined_groundTruth = np.column_stack((gen.raman_shift, ground_truth))
    combined_predictedRaman = np.column_stack((gen.raman_shift, pred_raman))


    np.savetxt(f'noisy_input1_{SNR}dB.txt', combined_noisyInput)
    np.savetxt(f'predicted_baseline1_{SNR}dB.txt', combined_predictedBaseline)
    np.savetxt(f'ground_truth1_{SNR}dB.txt', combined_groundTruth)
    np.savetxt(f'predicted_raman1_{SNR}dB.txt', combined_predictedRaman)

    mse = np.mean((ground_truth - pred_raman) ** 2)

    # Calculate Cosine Similarity
    # Adding a small epsilon (1e-10) to avoid division by zero in edge cases
    dot_product = np.dot(ground_truth, pred_raman)
    norm_gt = np.linalg.norm(ground_truth)
    norm_pred = np.linalg.norm(pred_raman)
    
    if norm_gt == 0 or norm_pred == 0:
        cosine_sim = 0.0
    else:
        cosine_sim = dot_product / (norm_gt * norm_pred)





    print(f"\nModel Performance Metrics:")
    print(f"MSE Error: {mse:.6f}")
    print(f"Max Absolute Error: {np.max(np.abs(residual)):.4f}")
    print(f"Cosine Similarity: {cosine_sim:.4f}")

if __name__ == "__main__":
    compare_results()