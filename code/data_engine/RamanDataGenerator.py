import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import CubicSpline # <--- Required for the update






class FixedRamanDataset(Dataset):
    """
    Generates N samples ONCE using your RamanDataGenerator logic, 
    normalizes them exactly like your training batches, and freezes them in memory.
    """
    def __init__(self, generator, n_samples=1000):
        """
        Args:
            generator (RamanDataGenerator): An instance of your data generator class.
            n_samples (int): Number of validation samples to freeze.
        """
        self.samples = []
        print(f"Generating fixed validation set ({n_samples} samples)...")
        
        # We assume the generator is already initialized with baselines
        for _ in range(n_samples):
            # 1. Generate Raw Data (Using your method)
            # Returns numpy arrays: x (Input), y_s (Signal), y_b (Baseline)
            x, y_s, y_b = generator.generate_random_spectrum()
            
            # 2. Normalization (COPIED EXACTLY from your get_batch/getitem)
            min_val, max_val = np.min(x), np.max(x)
            
            if max_val - min_val > 1e-6:
                # Input: 0 to 1
                norm_x = (x - min_val) / (max_val - min_val)
                # Baseline: Scaled relative to input range
                norm_b = (y_b - min_val) / (max_val - min_val)
                # Signal: Scaled relative to input range AND multiplied by 10
                norm_s = (y_s / (max_val - min_val)) * 10.0 
            else:
                norm_x = np.zeros_like(x)
                norm_b = np.zeros_like(y_b)
                norm_s = np.zeros_like(y_s)
            
            # 3. Convert to Tensor & Add Channel Dimension
            # Final Shape: [1, n_points]
            t_x = torch.tensor(norm_x, dtype=torch.float32).unsqueeze(0)
            t_ys = torch.tensor(norm_s, dtype=torch.float32).unsqueeze(0)
            t_yb = torch.tensor(norm_b, dtype=torch.float32).unsqueeze(0)
            
            # Store tuple in memory
            self.samples.append((t_x, t_ys, t_yb))
            
        print("Fixed validation set created.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class RamanDataGenerator2(Dataset):
    def __init__(self, baseline_file_path, epoch_size=1000):
        """
        Args:
            baseline_file_path (str): Path to file.
            epoch_size (int): Virtual size of dataset for one training epoch.
        """
        self.epoch_size = epoch_size

        # 1. Load Data & Auto-Detect Points
        try:
            # Load all data
            raw_data = np.loadtxt(baseline_file_path)
            
            # shape[0] is the number of rows (spectral points)
            self.n_points = raw_data.shape[0] 
            
            # Split: Col 0 is Raman Shift, Cols 1:end are Baselines
            self._raman_shift = raw_data[:, 0]
            self.baselines_raw = raw_data[:, 1:].T
            
            print(f"Auto-detected {self.n_points} spectral points.")
            print(f"Loaded {self.baselines_raw.shape[0]} baselines.")
            
            # 2. Normalize Baselines Immediately
            self.baselines = []
            for b in self.baselines_raw:
                min_b = np.min(b)
                max_b = np.max(b)
                if max_b - min_b > 1e-6:
                    self.baselines.append((b - min_b) / (max_b - min_b))
                else:
                    self.baselines.append(np.zeros_like(b))
            self.baselines = np.array(self.baselines)
            
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Falling back to synthetic data.")
            self.n_points = 864
            self._raman_shift = np.linspace(200, 3500, self.n_points)
            self.baselines = self._generate_mock_baselines(50)

    @property
    def raman_shift(self):
        return self._raman_shift

    def _generate_mock_baselines(self, n=50):
        baselines = []
        x = np.linspace(0, 3.14, self.n_points)
        for _ in range(n):
            a, b = np.random.uniform(0.5, 5.0), np.random.uniform(0.5, 5.0)
            base = a * np.sin(x) + b * (x**2)
            base = (base - np.min(base)) / (np.max(base) - np.min(base))
            baselines.append(base)
        return np.array(baselines)

    def _find_nearest_idx(self, value):
        return (np.abs(self._raman_shift - value)).argmin()

    def pseudo_voigt(self, x_indices, center_idx, width_idx, eta):
        sigma = width_idx / 2.0
        L = 1 / (1 + ((x_indices - center_idx) / sigma) ** 2)
        G = np.exp(-np.log(2) * ((x_indices - center_idx) / sigma) ** 2)
        return eta * L + (1 - eta) * G

    def get_mixed_baseline(self):
        idx1, idx2 = np.random.choice(len(self.baselines), 2, replace=False)
        w = np.random.uniform(0, 1)
        return w * self.baselines[idx1] + (1 - w) * self.baselines[idx2]

    # Update arguments to accept overrides
    def generate_random_spectrum(self, force_blank=False, force_snr=None):
        # 1. Generate Signal
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        # --- LOGIC CHANGE: Check forced flag first, then random ---
        if force_blank:
            is_blank = True
        else:
            # If not forced, we assume it's a signal (since get_batch handles the ratio)
            # Or you can keep the random check as a fallback
            is_blank = False 

        if is_blank:
            n_peaks = 0
        else:
            n_peaks = np.random.randint(3, 16)
        
        # ... (Peak generation loop matches is_blank) ...
        for _ in range(n_peaks):
             # ... (Peak generation code) ...
             pass

        # 2. Baseline (Same as before)
        baseline_amp = np.random.uniform(2.0, 10.0) 
        baseline = self.get_mixed_baseline() * baseline_amp

        # 3. Add Noise
        if force_snr is not None:
            target_snr = force_snr
        else:
            # Fallback to random if not specified
            log_min = np.log(2.0)
            log_max = np.log(25.0)
            target_snr = np.exp(np.random.uniform(log_min, log_max))

        # ... (Noise addition logic same as before) ...
        max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
        noise_sigma = max_sig / target_snr
        noise = np.random.normal(0, noise_sigma, self.n_points)

        final_input = signal + noise + baseline
        return final_input, signal, baseline

    def generate_defined_spectrum(self, peak_locs_cm, widths_idx, amplitudes, eta, snr=None):
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        for loc_cm, w_idx, amp in zip(peak_locs_cm, widths_idx, amplitudes):
            center_idx = self._find_nearest_idx(loc_cm)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, w_idx, eta)

        if snr is not None:
            max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
            noise_sigma = max_sig / snr
            noise = np.random.normal(0, noise_sigma, self.n_points)
        else:
            noise = np.zeros(self.n_points)

        baseline = self.get_mixed_baseline() * 5.0 # Fixed amp for defined spec
        return signal + noise + baseline, signal, baseline

    # --- 1. PyTorch Standard Batch Loader ---
    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        x, y_s, y_b = self.generate_random_spectrum()
        
        # Normalize per sample
        min_val, max_val = np.min(x), np.max(x)
        if max_val - min_val > 1e-6:
            x = (x - min_val) / (max_val - min_val)
            y_s = (y_s / (max_val - min_val))
            y_b = (y_b - min_val) / (max_val - min_val)
            
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_b, dtype=torch.float32).unsqueeze(0)
        )

    # --- 2. Custom Batch Generator (Your Request) ---
    def get_batch(self, batch_size=32):
        batch_x = []
        batch_y_s = []
        batch_y_b = []
        
        # 1. Calculate Exact Quotas
        n_blanks = max(1, int(batch_size * 0.10)) # Ensure at least 1 blank
        n_signals = batch_size - n_blanks
        
        # Split signals into SNR buckets to guarantee coverage
        n_low = n_signals // 3
        n_med = n_signals // 3
        n_high = n_signals - n_low - n_med # Catch remainder
        
        # 2. Generate Blanks
        for _ in range(n_blanks):
            # SNR doesn't matter much for blanks (just noise level), pick random or fixed
            x, y_s, y_b = self.generate_random_spectrum(force_blank=True)
            batch_x.append(self._normalize(x))
            batch_y_s.append(self._normalize(y_s)) # Will be zeros
            batch_y_b.append(self._normalize(y_b))
            
        # 3. Generate Signals with Stratified SNR
        # Low SNR (2 - 5) - "The Hard Stuff"
        for _ in range(n_low):
            snr = np.random.uniform(2, 5)
            x, y_s, y_b = self.generate_random_spectrum(force_blank=False, force_snr=snr)
            batch_x.append(self._normalize(x))
            batch_y_s.append(self._normalize(y_s))
            batch_y_b.append(self._normalize(y_b))

        # Med SNR (5 - 15) - "The Typical Stuff"
        for _ in range(n_med):
            snr = np.random.uniform(5, 15)
            x, y_s, y_b = self.generate_random_spectrum(force_blank=False, force_snr=snr)
            batch_x.append(self._normalize(x))
            batch_y_s.append(self._normalize(y_s))
            batch_y_b.append(self._normalize(y_b))
            
        # High SNR (15 - 40) - "The Clean Stuff"
        for _ in range(n_high):
            snr = np.random.uniform(15, 40)
            x, y_s, y_b = self.generate_random_spectrum(force_blank=False, force_snr=snr)
            batch_x.append(self._normalize(x))
            batch_y_s.append(self._normalize(y_s))
            batch_y_b.append(self._normalize(y_b))

        # 4. Shuffle! (Crucial so blanks aren't always at the start)
        # We zip, shuffle, and unzip
        combined = list(zip(batch_x, batch_y_s, batch_y_b))
        np.random.shuffle(combined)
        batch_x, batch_y_s, batch_y_b = zip(*combined)

        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_s), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_b), dtype=torch.float32).unsqueeze(1)
        )

    def _normalize(self, arr):
        """Helper to keep code clean"""
        min_val, max_val = np.min(arr), np.max(arr)
        if max_val - min_val > 1e-6:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)



class zzRamanDataGenerator(Dataset):
    def __init__(self, baseline_file_path, epoch_size=1000):
        """
        Args:
            baseline_file_path (str): Path to file.
            epoch_size (int): Virtual size of dataset for one training epoch.
        """
        self.epoch_size = epoch_size

        # --- 1. Load Data & Auto-Detect Points ---
        try:
            print(f"Loading baselines from: {baseline_file_path}")
            raw_data = np.loadtxt(baseline_file_path)
            
            # shape[0] is points, Col 0 is Shift, Cols 1..N are Baselines
            self.n_points = raw_data.shape[0] 
            self._raman_shift = raw_data[:, 0]
            self.baselines_raw = raw_data[:, 1:].T
            
            print(f"Auto-detected {self.n_points} spectral points.")
            print(f"Loaded {self.baselines_raw.shape[0]} baselines.")
            
            # Normalize Baselines
            self.baselines = []
            for b in self.baselines_raw:
                min_b = np.min(b)
                max_b = np.max(b)
                if max_b - min_b > 1e-6:
                    self.baselines.append((b - min_b) / (max_b - min_b))
                else:
                    self.baselines.append(np.zeros_like(b))
            self.baselines = np.array(self.baselines)
            
        except Exception as e:
            print(f"Warning: {e}")
            print("Falling back to synthetic data.")
            self.n_points = 864
            self._raman_shift = np.linspace(200, 3500, self.n_points)
            self.baselines = self._generate_mock_baselines(50)


    def generate_defined_spectrum(self, peak_locs_cm, widths_idx, amplitudes, eta, snr=None):
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        for loc_cm, w_idx, amp in zip(peak_locs_cm, widths_idx, amplitudes):
            center_idx = self._find_nearest_idx(loc_cm)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, w_idx, eta)

        if snr is not None:
            max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
            noise_sigma = max_sig / snr
            noise = np.random.normal(0, noise_sigma, self.n_points)
        else:
            noise = np.zeros(self.n_points)

        baseline = self.get_mixed_baseline() * 5.0 # Fixed amp for defined spec
        return signal + noise + baseline, signal, baseline
    # --- Helper Methods ---
    def _generate_mock_baselines(self, n=50):
        baselines = []
        x = np.linspace(0, 3.14, self.n_points)
        for _ in range(n):
            a, b = np.random.uniform(0.5, 5.0), np.random.uniform(0.5, 5.0)
            base = a * np.sin(x) + b * (x**2)
            base = (base - np.min(base)) / (np.max(base) - np.min(base))
            baselines.append(base)
        return np.array(baselines)

    def _find_nearest_idx(self, value):
        return (np.abs(self._raman_shift - value)).argmin()

    
    def _generate_polynomial_baseline(self):
        """Generates a random slow-varying polynomial (Order 1 to 4)."""
        x = np.linspace(-1, 1, self.n_points) # Normalized domain for stability
        order = np.random.randint(1, 5)       # Order 1, 2, 3, or 4
        
        # Random coefficients (small enough to not dominate, large enough to matter)
        coeffs = np.random.uniform(-0.5, 0.5, size=order+1)
        
        poly = np.polyval(coeffs, x)
        
        # Normalize to 0-1 range so we can control its strength later
        if np.max(poly) - np.min(poly) > 1e-6:
            poly = (poly - np.min(poly)) / (np.max(poly) - np.min(poly))
        else:
            poly = np.zeros_like(poly)
            
        return poly
    
    def get_mixed_baseline(self):
        """
        Mixes:
        1. Two real experimental baselines (Texture)
        2. One random polynomial (Shape/Drift)
        """
        # A. Experimental Component
        if len(self.baselines) > 0:
            idx1, idx2 = np.random.choice(len(self.baselines), 2, replace=True)
            w = np.random.uniform(0, 1)
            real_base = w * self.baselines[idx1] + (1 - w) * self.baselines[idx2]
        else:
            real_base = np.zeros(self.n_points)
            
        # B. Polynomial Component (The new idea)
        poly_base = self._generate_polynomial_baseline()
        
        # C. Combine them
        # We want the Real Baseline to provide "Texture" and the Polynomial to provide "Trend".
        # Let's say 70% Real, 30% Poly (Randomized)
        alpha = np.random.uniform(0.2, 0.6) # Poly strength
        
        mixed = (1 - alpha) * real_base + (alpha * poly_base)
        
        # Re-normalize to ensure consistency before scaling amplitude later
        if np.max(mixed) - np.min(mixed) > 1e-6:
            mixed = (mixed - np.min(mixed)) / (np.max(mixed) - np.min(mixed))
            
        return mixed
    
    
    
    
    
    def pseudo_voigt(self, x_indices, center_idx, width_idx, eta):
        sigma = width_idx / 2.0
        if sigma < 1e-6: sigma = 1e-6
        diff = (x_indices - center_idx) / sigma
        L = 1 / (1 + diff ** 2)
        G = np.exp(-np.log(2) * diff ** 2)
        return eta * L + (1 - eta) * G

    def get_mixed_baseline(self):
        if len(self.baselines) > 0:
            idx1, idx2 = np.random.choice(len(self.baselines), 2, replace=True)
            w = np.random.uniform(0, 1)
            return w * self.baselines[idx1] + (1 - w) * self.baselines[idx2]
        return np.zeros(self.n_points)

    # --- The Original Generation Logic ---
    def generate_random_spectrum(self):
        # 1. Generate Signal or Blank
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        # PROBABILISTIC CHECK: 10% chance of blank
        if np.random.rand() < 0.1:
            n_peaks = 0 # Blank
        else:
            n_peaks = np.random.randint(3, 16) # Signal

        # Generate peaks
        for _ in range(n_peaks):
            loc_cm = np.random.uniform(self._raman_shift[20], self._raman_shift[-20])
            center_idx = self._find_nearest_idx(loc_cm)
            width_idx = np.random.uniform(5, 30)
            amp = np.random.uniform(0.1, 1.0)
            eta = np.random.uniform(0.5, 1.0)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, width_idx, eta)

        # 2. Add Baseline
        baseline_amp = np.random.uniform(2.0, 10.0) 
        baseline = self.get_mixed_baseline() * baseline_amp

        # 3. Add Noise (Random SNR 2 to 60)
        # Log-Uniform distribution so we get good mix of orders of magnitude
        log_min = np.log(1.0)
        log_max = np.log(25.0)
        target_snr = np.exp(np.random.uniform(log_min, log_max))

        max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
        noise_sigma = max_sig / target_snr
        noise = np.random.normal(0, noise_sigma, self.n_points)

        final_input = signal + noise + baseline
        return final_input, signal, baseline

    # --- Batch Generation (Random Loop) ---
    def get_batch(self, batch_size=32):
        """
        Standard random sampling. No buckets, no curriculum.
        """
        batch_x = []
        batch_y_s = []
        batch_y_b = []
        
        for _ in range(batch_size):
            x, y_s, y_b = self.generate_random_spectrum()
            
            # Normalize
            min_val, max_val = np.min(x), np.max(x)
            if max_val - min_val > 1e-6:
                norm_x = (x - min_val) / (max_val - min_val)
                norm_b = (y_b - min_val) / (max_val - min_val)
                
                # I kept the 10x scaling here because it fixed the gradient issues 
                # for weak signals. If you want pure original, remove the * 10.0
                norm_s = (y_s / (max_val - min_val)) * 10.0
            else:
                norm_x = np.zeros_like(x)
                norm_b = np.zeros_like(y_b)
                norm_s = np.zeros_like(y_s)
            
            batch_x.append(norm_x)
            batch_y_s.append(norm_s)
            batch_y_b.append(norm_b)

        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_s), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_b), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        x, y_s, y_b = self.generate_random_spectrum()
        # Normalization logic for single item
        min_val, max_val = np.min(x), np.max(x)
        if max_val - min_val > 1e-6:
            x = (x - min_val) / (max_val - min_val)
            y_s = (y_s / (max_val - min_val)) * 10.0 # Scaling
            y_b = (y_b - min_val) / (max_val - min_val)
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_b, dtype=torch.float32).unsqueeze(0)
        )
    
    @property
    def raman_shift(self):
        return self._raman_shift




class zzzRamanDataGenerator(Dataset):
    def __init__(self, baseline_file_path, epoch_size=1000):
        """
        Args:
            baseline_file_path (str): Path to file.
            epoch_size (int): Virtual size of dataset for one training epoch.
        """
        self.epoch_size = epoch_size

        # --- 1. Load Data & Auto-Detect Points ---
        try:
            print(f"Loading baselines from: {baseline_file_path}")
            raw_data = np.loadtxt(baseline_file_path)
            
            # shape[0] is points, Col 0 is Shift, Cols 1..N are Baselines
            self.n_points = raw_data.shape[0] 
            self._raman_shift = raw_data[:, 0]
            self.baselines_raw = raw_data[:, 1:].T
            
            print(f"Auto-detected {self.n_points} spectral points.")
            print(f"Loaded {self.baselines_raw.shape[0]} baselines.")
            
            # Normalize Baselines
            self.baselines = []
            for b in self.baselines_raw:
                min_b = np.min(b)
                max_b = np.max(b)
                if max_b - min_b > 1e-6:
                    self.baselines.append((b - min_b) / (max_b - min_b))
                else:
                    self.baselines.append(np.zeros_like(b))
            self.baselines = np.array(self.baselines)
            
        except Exception as e:
            print(f"Warning: {e}")
            print("Falling back to synthetic data.")
            self.n_points = 864
            self._raman_shift = np.linspace(200, 3500, self.n_points)
            self.baselines = self._generate_mock_baselines(50)


    def generate_defined_spectrum(self, peak_locs_cm, widths_idx, amplitudes, eta, snr=None):
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        for loc_cm, w_idx, amp in zip(peak_locs_cm, widths_idx, amplitudes):
            center_idx = self._find_nearest_idx(loc_cm)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, w_idx, eta)

        if snr is not None:
            max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
            noise_sigma = max_sig / snr
            noise = np.random.normal(0, noise_sigma, self.n_points)
        else:
            noise = np.zeros(self.n_points)

        baseline = self.get_mixed_baseline() * 5.0 # Fixed amp for defined spec
        return signal + noise + baseline, signal, baseline

    # --- Helper Methods ---
    def _generate_mock_baselines(self, n=50):
        baselines = []
        x = np.linspace(0, 3.14, self.n_points)
        for _ in range(n):
            a, b = np.random.uniform(0.5, 5.0), np.random.uniform(0.5, 5.0)
            base = a * np.sin(x) + b * (x**2)
            base = (base - np.min(base)) / (np.max(base) - np.min(base))
            baselines.append(base)
        return np.array(baselines)

    def _find_nearest_idx(self, value):
        return (np.abs(self._raman_shift - value)).argmin()

    def pseudo_voigt(self, x_indices, center_idx, width_idx, eta):
        sigma = width_idx / 2.0
        if sigma < 1e-6: sigma = 1e-6
        diff = (x_indices - center_idx) / sigma
        L = 1 / (1 + diff ** 2)
        G = np.exp(-np.log(2) * diff ** 2)
        return eta * L + (1 - eta) * G

    # --- NEW: Polynomial Generator ---
    def _generate_polynomial_baseline(self):
        """Generates a random slow-varying polynomial (Order 1 to 4)."""
        x = np.linspace(-1, 1, self.n_points) # Normalized domain for stability
        order = np.random.randint(1, 5)       # Order 1, 2, 3, or 4
        
        # Random coefficients (small enough to not dominate, large enough to matter)
        coeffs = np.random.uniform(-0.5, 0.5, size=order+1)
        
        poly = np.polyval(coeffs, x)
        
        # Normalize to 0-1 range so we can control its strength later
        if np.max(poly) - np.min(poly) > 1e-6:
            poly = (poly - np.min(poly)) / (np.max(poly) - np.min(poly))
        else:
            poly = np.zeros_like(poly)
            
        return poly

    # --- UPDATED: Mixed Baseline Logic ---
    def get_mixed_baseline(self):
        """
        Mixes:
        1. Two real experimental baselines (Texture)
        2. One random polynomial (Shape/Drift)
        """
        # A. Experimental Component
        if len(self.baselines) > 0:
            idx1, idx2 = np.random.choice(len(self.baselines), 2, replace=True)
            w = np.random.uniform(0, 1)
            real_base = w * self.baselines[idx1] + (1 - w) * self.baselines[idx2]
        else:
            real_base = np.zeros(self.n_points)
            
        # B. Polynomial Component (New Idea)
        
        poly_base = self._generate_polynomial_baseline()
        
        # C. Combine them
        # We want the Real Baseline to provide "Texture" and the Polynomial to provide "Trend".
        # Let's say we mix them with a random weight.
        # alpha is the strength of the polynomial (e.g., 0.2 to 0.6)
        alpha = np.random.uniform(0.2, 0.6) 
        
        mixed = (1 - alpha) * real_base + (alpha * poly_base)
        

        
        # Re-normalize to ensure consistency before scaling amplitude later
        if np.max(mixed) - np.min(mixed) > 1e-6:
            mixed = (mixed - np.min(mixed)) / (np.max(mixed) - np.min(mixed))
            
        return mixed

    # --- The Original Generation Logic ---
    def generate_random_spectrum(self):
        # 1. Generate Signal or Blank
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        # PROBABILISTIC CHECK: 10% chance of blank
        if np.random.rand() < 0.1:
            n_peaks = 0 # Blank
        else:
            n_peaks = np.random.randint(5, 25) # Signal

        # Generate peaks
        for _ in range(n_peaks):
            loc_cm = np.random.uniform(self._raman_shift[20], self._raman_shift[-20])
            center_idx = self._find_nearest_idx(loc_cm)
            width_idx = np.random.uniform(5, 30)
            amp = np.random.uniform(0.1, 1.0)
            eta = np.random.uniform(0.5, 1.0)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, width_idx, eta)

        # 2. Add Baseline (Uses the NEW mixed logic)
        baseline_amp = np.random.uniform(2.0, 10.0) 
        baseline = self.get_mixed_baseline() * baseline_amp

        # 3. Add Noise (Random SNR 2 to 60)
        # Log-Uniform distribution so we get good mix of orders of magnitude
        log_min = np.log(1.0)
        log_max = np.log(25.0)
        target_snr = np.exp(np.random.uniform(log_min, log_max))

        max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
        noise_sigma = max_sig / target_snr
        noise = np.random.normal(0, noise_sigma, self.n_points)

        final_input = signal + noise + baseline
        return final_input, signal, baseline

    # --- Batch Generation (Random Loop) ---
    def get_batch(self, batch_size=32,poly=True):
        """
        Standard random sampling. No buckets, no curriculum.
        """
        batch_x = []
        batch_y_s = []
        batch_y_b = []
        
        for _ in range(batch_size):
            x, y_s, y_b = self.generate_random_spectrum()
            
            # Normalize
            min_val, max_val = np.min(x), np.max(x)
            if max_val - min_val > 1e-6:
                norm_x = (x - min_val) / (max_val - min_val)
                norm_b = (y_b - min_val) / (max_val - min_val)
                
                # I kept the 10x scaling here because it fixed the gradient issues 
                # for weak signals. If you want pure original, remove the * 10.0
                norm_s = (y_s / (max_val - min_val)) * 10.0
            else:
                norm_x = np.zeros_like(x)
                norm_b = np.zeros_like(y_b)
                norm_s = np.zeros_like(y_s)
            
            batch_x.append(norm_x)
            batch_y_s.append(norm_s)
            batch_y_b.append(norm_b)

        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_s), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_b), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        x, y_s, y_b = self.generate_random_spectrum()
        # Normalization logic for single item
        min_val, max_val = np.min(x), np.max(x)
        if max_val - min_val > 1e-6:
            x = (x - min_val) / (max_val - min_val)
            y_s = (y_s / (max_val - min_val)) * 10.0 # Scaling
            y_b = (y_b - min_val) / (max_val - min_val)
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_b, dtype=torch.float32).unsqueeze(0)
        )
    
    @property
    def raman_shift(self):
        return self._raman_shift



class zzzzRamanDataGenerator(Dataset):
    def __init__(self, baseline_file_path, epoch_size=1000):
        """
        Args:
            baseline_file_path (str): Path to file.
            epoch_size (int): Virtual size of dataset for one training epoch.
        """
        self.epoch_size = epoch_size

        # --- 1. Load Data & Auto-Detect Points ---
        try:
            print(f"Loading baselines from: {baseline_file_path}")
            raw_data = np.loadtxt(baseline_file_path)
            
            # shape[0] is points, Col 0 is Shift, Cols 1..N are Baselines
            self.n_points = raw_data.shape[0] 
            self._raman_shift = raw_data[:, 0]
            self.baselines_raw = raw_data[:, 1:].T
            
            print(f"Auto-detected {self.n_points} spectral points.")
            print(f"Loaded {self.baselines_raw.shape[0]} baselines.")
            
            # Normalize Baselines
            self.baselines = []
            for b in self.baselines_raw:
                min_b = np.min(b)
                max_b = np.max(b)
                if max_b - min_b > 1e-6:
                    self.baselines.append((b - min_b) / (max_b - min_b))
                else:
                    self.baselines.append(np.zeros_like(b))
            self.baselines = np.array(self.baselines)
            
        except Exception as e:
            print(f"Warning: {e}")
            print("Falling back to synthetic data.")
            self.n_points = 864
            self._raman_shift = np.linspace(200, 3500, self.n_points)
            self.baselines = self._generate_mock_baselines(50)


    def generate_defined_spectrum(self, peak_locs_cm, widths_idx, amplitudes, eta, snr=None):
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        for loc_cm, w_idx, amp in zip(peak_locs_cm, widths_idx, amplitudes):
            center_idx = self._find_nearest_idx(loc_cm)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, w_idx, eta)

        if snr is not None:
            max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
            noise_sigma = max_sig / snr
            noise = np.random.normal(0, noise_sigma, self.n_points)
        else:
            noise = np.zeros(self.n_points)

        baseline = self.get_mixed_baseline() * 5.0 # Fixed amp for defined spec
        return signal + noise + baseline, signal, baseline

    # --- Helper Methods ---
    def _generate_mock_baselines(self, n=50):
        baselines = []
        x = np.linspace(0, 3.14, self.n_points)
        for _ in range(n):
            a, b = np.random.uniform(0.5, 5.0), np.random.uniform(0.5, 5.0)
            base = a * np.sin(x) + b * (x**2)
            base = (base - np.min(base)) / (np.max(base) - np.min(base))
            baselines.append(base)
        return np.array(baselines)

    def _find_nearest_idx(self, value):
        return (np.abs(self._raman_shift - value)).argmin()

    def pseudo_voigt(self, x_indices, center_idx, width_idx, eta):
        sigma = width_idx / 2.0
        if sigma < 1e-6: sigma = 1e-6
        diff = (x_indices - center_idx) / sigma
        L = 1 / (1 + diff ** 2)
        G = np.exp(-np.log(2) * diff ** 2)
        return eta * L + (1 - eta) * G

    # --- NEW: Cubic Spline Generator (Replaces Polynomial) ---
    def _generate_spline_baseline(self):
        """
        Generates a smooth random baseline using Cubic Splines.
        This creates more realistic, non-linear fluorescence shapes compared to polynomials.
        """
        x_grid = np.arange(self.n_points)
        
        # 1. Random Anchor Points (3 to 8 points)
        num_anchors = np.random.randint(3, 9) 
        
        # X-coordinates: evenly spaced anchors across the spectrum
        anchor_x = np.linspace(0, self.n_points - 1, num_anchors)
        
        # Y-coordinates: Random intensity variations
        # We use uniform random values to simulate peaks and valleys in the baseline
        anchor_y = np.random.uniform(0, 1.0, size=num_anchors)
        
        # 2. Cubic Spline Interpolation
        # Requires scipy.interpolate.CubicSpline
        cs = CubicSpline(anchor_x, anchor_y)
        spline = cs(x_grid)
        
        # 3. Normalize to 0-1 range
        if np.max(spline) - np.min(spline) > 1e-6:
            spline = (spline - np.min(spline)) / (np.max(spline) - np.min(spline))
        else:
            spline = np.zeros_like(spline)
            
        return spline

    # --- UPDATED: Mixed Baseline Logic ---
    def get_mixed_baseline(self):
        """
        Mixes:
        1. Real Experimental Baselines (Texture/Noise features)
        2. Synthetic Spline Baselines (Shape/Drift features)
        """
        # A. Experimental Component
        if len(self.baselines) > 0:
            idx1, idx2 = np.random.choice(len(self.baselines), 2, replace=True)
            w = np.random.uniform(0, 1)
            real_base = w * self.baselines[idx1] + (1 - w) * self.baselines[idx2]
        else:
            real_base = np.zeros(self.n_points)
            
        # B. Spline Component (Replaces Polynomial)
        synthetic_base = self._generate_spline_baseline()
        
        # C. Combine them
        # alpha controls the blend:
        # High alpha -> More synthetic shape (drift/curvature)
        # Low alpha -> More real texture (glass signals/pattern noise)
        alpha = np.random.uniform(0.3, 0.7) 
        
        mixed = (1 - alpha) * real_base + (alpha * synthetic_base)
        
        # Re-normalize to ensure consistency before scaling amplitude later
        if np.max(mixed) - np.min(mixed) > 1e-6:
            mixed = (mixed - np.min(mixed)) / (np.max(mixed) - np.min(mixed))
            
        return mixed

    # --- The Original Generation Logic ---
    def generate_random_spectrum(self):
        # 1. Generate Signal or Blank
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        # PROBABILISTIC CHECK: 10% chance of blank
        if np.random.rand() < 0.1:
            n_peaks = 0 # Blank
        else:
            n_peaks = np.random.randint(5, 25) # Signal

        # Generate peaks
        for _ in range(n_peaks):
            loc_cm = np.random.uniform(self._raman_shift[20], self._raman_shift[-20])
            center_idx = self._find_nearest_idx(loc_cm)
            width_idx = np.random.uniform(5, 30)
            amp = np.random.uniform(0.1, 1.0)
            eta = np.random.uniform(0.5, 1.0)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, width_idx, eta)

        # 2. Add Baseline (Uses the NEW mixed logic)
        baseline_amp = np.random.uniform(2.0, 10.0) 
        baseline = self.get_mixed_baseline() * baseline_amp

        # 3. Add Noise (Random SNR 2 to 60)
        # Log-Uniform distribution so we get good mix of orders of magnitude
        log_min = np.log(5.0)
        log_max = np.log(25.0)
        target_snr = np.exp(np.random.uniform(log_min, log_max))

        max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
        noise_sigma = max_sig / target_snr
        noise = np.random.normal(0, noise_sigma, self.n_points)

        final_input = signal + noise + baseline
        return final_input, signal, baseline

    # --- Batch Generation (Random Loop) ---
    def get_batch(self, batch_size=32,poly=True):
        """
        Standard random sampling. No buckets, no curriculum.
        """
        batch_x = []
        batch_y_s = []
        batch_y_b = []
        
        for _ in range(batch_size):
            x, y_s, y_b = self.generate_random_spectrum()
            
            # Normalize
            min_val, max_val = np.min(x), np.max(x)
            if max_val - min_val > 1e-6:
                norm_x = (x - min_val) / (max_val - min_val)
                norm_b = (y_b - min_val) / (max_val - min_val)
                
                # I kept the 10x scaling here because it fixed the gradient issues 
                # for weak signals. If you want pure original, remove the * 10.0
                norm_s = (y_s / (max_val - min_val)) * 1.0
            else:
                norm_x = np.zeros_like(x)
                norm_b = np.zeros_like(y_b)
                norm_s = np.zeros_like(y_s)
            
            batch_x.append(norm_x)
            batch_y_s.append(norm_s)
            batch_y_b.append(norm_b)

        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_s), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_b), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        x, y_s, y_b = self.generate_random_spectrum()
        # Normalization logic for single item
        min_val, max_val = np.min(x), np.max(x)
        if max_val - min_val > 1e-6:
            x = (x - min_val) / (max_val - min_val)
            y_s = (y_s / (max_val - min_val)) * 10.0 # Scaling
            y_b = (y_b - min_val) / (max_val - min_val)
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_b, dtype=torch.float32).unsqueeze(0)
        )
    
    @property
    def raman_shift(self):
        return self._raman_shift





class zzzzzRamanDataGenerator(Dataset):
    def __init__(self, baseline_file_path, epoch_size=1000,min_snr=5,max_snr=25):
        """
        Args:
            baseline_file_path (str): Path to file.
            epoch_size (int): Virtual size of dataset for one training epoch.
        """
        self.epoch_size = epoch_size
        self.min_snr=min_snr
        self.max_snr=max_snr

        # --- 1. Load Data & Auto-Detect Points ---
        try:
            print(f"Loading baselines from: {baseline_file_path}")
            raw_data = np.loadtxt(baseline_file_path)
            
            # shape[0] is points, Col 0 is Shift, Cols 1..N are Baselines
            self.n_points = raw_data.shape[0] 
            self._raman_shift = raw_data[:, 0]
            self.baselines_raw = raw_data[:, 1:].T
            
            print(f"Auto-detected {self.n_points} spectral points.")
            print(f"Loaded {self.baselines_raw.shape[0]} baselines.")
            
            # Normalize Baselines
            self.baselines = []
            for b in self.baselines_raw:
                min_b = np.min(b)
                max_b = np.max(b)
                if max_b - min_b > 1e-6:
                    self.baselines.append((b - min_b) / (max_b - min_b))
                else:
                    self.baselines.append(np.zeros_like(b))
            self.baselines = np.array(self.baselines)
            
        except Exception as e:
            print(f"Warning: {e}")
            print("Falling back to synthetic data.")
            self.n_points = 864
            self._raman_shift = np.linspace(200, 3500, self.n_points)
            self.baselines = self._generate_mock_baselines(50)


    def generate_defined_spectrum(self, peak_locs_cm, widths_idx, amplitudes, eta, snr=None):
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        for loc_cm, w_idx, amp in zip(peak_locs_cm, widths_idx, amplitudes):
            center_idx = self._find_nearest_idx(loc_cm)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, w_idx, eta)

        if snr is not None:
            max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
            noise_sigma = max_sig / snr
            noise = np.random.normal(0, noise_sigma, self.n_points)
        else:
            noise = np.zeros(self.n_points)

        baseline = self.get_mixed_baseline() * 5.0 # Fixed amp for defined spec
        return signal + noise + baseline, signal, baseline

    # --- Helper Methods ---
    def _generate_mock_baselines(self, n=50):
        baselines = []
        x = np.linspace(0, 3.14, self.n_points)
        for _ in range(n):
            a, b = np.random.uniform(0.5, 5.0), np.random.uniform(0.5, 5.0)
            base = a * np.sin(x) + b * (x**2)
            base = (base - np.min(base)) / (np.max(base) - np.min(base))
            baselines.append(base)
        return np.array(baselines)

    def _find_nearest_idx(self, value):
        return (np.abs(self._raman_shift - value)).argmin()

    def pseudo_voigt(self, x_indices, center_idx, width_idx, eta):
        sigma = width_idx / 2.0
        if sigma < 1e-6: sigma = 1e-6
        diff = (x_indices - center_idx) / sigma
        L = 1 / (1 + diff ** 2)
        G = np.exp(-np.log(2) * diff ** 2)
        return eta * L + (1 - eta) * G

    # --- Cubic Spline Generator (Replaces Polynomial) ---
    def _generate_spline_baseline(self):
        """Generates a smooth random baseline using Cubic Splines."""
        x_grid = np.arange(self.n_points)
        num_anchors = np.random.randint(3, 11) 
        #anchor_x = np.linspace(0, self.n_points - 1, num_anchors)
        
        # We force the first and last points to be at edges (0 and N-1) to cover the full spectrum.
        middle_anchors = np.random.uniform(0, self.n_points - 1, num_anchors - 2)
        anchor_x = np.array([0] + sorted(middle_anchors) + [self.n_points - 1])
        
        
        
        anchor_y = np.random.uniform(0, 1.0, size=num_anchors)
        
        cs = CubicSpline(anchor_x, anchor_y)
        spline = cs(x_grid)
        
        if np.max(spline) - np.min(spline) > 1e-6:
            spline = (spline - np.min(spline)) / (np.max(spline) - np.min(spline))
        else:
            spline = np.zeros_like(spline)
            
        return spline

    def get_mixed_baseline(self):
        """Mixes Real Experimental Baselines with Synthetic Spline Baselines"""
        # A. Experimental Component
        if len(self.baselines) > 0:
            idx1, idx2 = np.random.choice(len(self.baselines), 2, replace=True)
            w = np.random.uniform(0, 1)
            real_base = w * self.baselines[idx1] + (1 - w) * self.baselines[idx2]
        else:
            real_base = np.zeros(self.n_points)
            
        # B. Spline Component
        synthetic_base = self._generate_spline_baseline()
        
        # C. Combine
        alpha = np.random.uniform(0.1, 0.5) 
        mixed = (1 - alpha) * real_base + (alpha * synthetic_base)
        
        if np.max(mixed) - np.min(mixed) > 1e-6:
            mixed = (mixed - np.min(mixed)) / (np.max(mixed) - np.min(mixed))
            
        return mixed

    # --- NEW: Beta Noise Generator (From OASIS Paper) ---
    def _add_beta_noise(self, current_spectrum):
        """
        Adds non-Gaussian artifacts ('dents' and 'spikes') using a Beta(1, 2) distribution.
        This forces the model to be robust against sensor glitches.
        """
        # 1. Generate Beta Noise (0 to 1 range, skewed towards 0)
        # alpha=1, beta=2 means most values are small, but some are large spikes.
        beta_raw = np.random.beta(a=1, b=2, size=self.n_points)
        
        # 2. Randomly decide Magnitude (Small vs Large Artifacts)
        # 50% chance of small dents, 50% chance of large spikes
        if np.random.rand() < 0.5:
            mag = np.random.uniform(0.001, 0.004) # Small
        else:
            mag = np.random.uniform(0.005, 0.025) # Large
            
        # 3. Scale and Apply Direction
        # We subtract 0.5 and multiply by 2 to center it around 0, then scale by magnitude.
        # Or, to strictly mimic 'dents', we can make it subtractive. 
        # Here we randomly flip signs to get both dents and spikes.
        signs = np.random.choice([-1, 1], size=self.n_points)
        beta_noise = beta_raw * mag * signs
        
        return current_spectrum + beta_noise

    # --- UPDATED: Random Spectrum Generation ---
    def generate_random_spectrum(self):
        # 1. Generate Signal or Blank
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        if np.random.rand() < 0.1:
            n_peaks = 0 # Blank
        else:
            n_peaks = np.random.randint(5, 25) # Signal

        for _ in range(n_peaks):
            loc_cm = np.random.uniform(self._raman_shift[20], self._raman_shift[-20])
            center_idx = self._find_nearest_idx(loc_cm)
            width_idx = np.random.uniform(5, 30)
            amp = np.random.uniform(0.1, 1.0)
            eta = np.random.uniform(0.5, 1.0)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, width_idx, eta)

        # 2. Add Baseline
        baseline_amp = np.random.uniform(2.0, 10.0) 
        baseline = self.get_mixed_baseline() * baseline_amp

        # 3. Add Standard Gaussian Noise (Sensor Thermal Noise)
        log_min = np.log(self.min_snr)
        log_max = np.log(self.max_snr)
        target_snr = np.exp(np.random.uniform(log_min, log_max))

        max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
        noise_sigma = max_sig / target_snr
        gaussian_noise = np.random.normal(0, noise_sigma, self.n_points)

        # 4. Add Beta Noise (Artifacts/Dents) - OASIS Upgrade
        # 50% chance to add this extra layer of difficulty
        noisy_spectrum = signal + gaussian_noise + baseline
        if np.random.rand() < 0.5:
            noisy_spectrum = self._add_beta_noise(noisy_spectrum)

        final_input = noisy_spectrum
        return final_input, signal, baseline

    def get_batch(self, batch_size=32, poly=True):
        batch_x, batch_y_s, batch_y_b = [], [], []
        
        for _ in range(batch_size):
            x, y_s, y_b = self.generate_random_spectrum()
            
            # Normalize
            min_val, max_val = np.min(x), np.max(x)
            if max_val - min_val > 1e-6:
                norm_x = (x - min_val) / (max_val - min_val)
                norm_b = (y_b - min_val) / (max_val - min_val)
                norm_s = (y_s / (max_val - min_val)) * 10.0 # 10x Scaling
            else:
                norm_x = np.zeros_like(x)
                norm_b = np.zeros_like(y_b)
                norm_s = np.zeros_like(y_s)
            
            batch_x.append(norm_x)
            batch_y_s.append(norm_s)
            batch_y_b.append(norm_b)

        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_s), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_b), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        x, y_s, y_b = self.generate_random_spectrum()
        min_val, max_val = np.min(x), np.max(x)
        if max_val - min_val > 1e-6:
            x = (x - min_val) / (max_val - min_val)
            y_s = (y_s / (max_val - min_val)) * 10.0
            y_b = (y_b - min_val) / (max_val - min_val)
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_b, dtype=torch.float32).unsqueeze(0)
        )
    
    @property
    def raman_shift(self):
        return self._raman_shift



class zzzzzRamanDataGenerator(Dataset):
    def __init__(self, baseline_file_path,noise_file_path=None, epoch_size=1000, min_snr=5, max_snr=25):
        """
        Args:
            baseline_file_path (str): Path to file.
            epoch_size (int): Virtual size of dataset for one training epoch.
        """
        self.epoch_size = epoch_size
        self.min_snr = min_snr
        self.max_snr = max_snr

        data = np.load(noise_file_path)
        self.noise_bank = data['noise']
        self.noise_rms = np.std(self.noise_bank, axis=1)

        # --- 1. Load Data & Auto-Detect Points ---
        try:
            print(f"Loading baselines from: {baseline_file_path}")
            raw_data = np.loadtxt(baseline_file_path)
            
            # shape[0] is points, Col 0 is Shift, Cols 1..N are Baselines
            self.n_points = raw_data.shape[0] 
            self._raman_shift = raw_data[:, 0]
            self.baselines_raw = raw_data[:, 1:].T
            
            print(f"Auto-detected {self.n_points} spectral points.")
            print(f"Loaded {self.baselines_raw.shape[0]} baselines.")
            
            # Normalize Baselines
            self.baselines = []
            for b in self.baselines_raw:
                min_b = np.min(b)
                max_b = np.max(b)
                if max_b - min_b > 1e-6:
                    self.baselines.append((b - min_b) / (max_b - min_b))
                else:
                    self.baselines.append(np.zeros_like(b))
            self.baselines = np.array(self.baselines)
            
        except Exception as e:
            print(f"Warning: {e}")
            print("Falling back to synthetic data.")
            self.n_points = 864
            self._raman_shift = np.linspace(200, 3500, self.n_points)
            self.baselines = self._generate_mock_baselines(50)

    # --- UPDATED: Defined Spectrum Generation ---
    def generate_defined_spectrum(self, peak_locs_cm, widths_idx, amplitudes, eta, snr=None):
        """
        Generates a spectrum with specific peak parameters.
        NOW INCLUDES: Beta-distributed artifacts to match training data.
        """
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        for loc_cm, w_idx, amp in zip(peak_locs_cm, widths_idx, amplitudes):
            center_idx = self._find_nearest_idx(loc_cm)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, w_idx, eta)

        # 1. Add Gaussian Noise (Thermal)
        if snr is not None:
            max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
            noise_sigma = max_sig / snr
            gaussian_noise = np.random.normal(0, noise_sigma, self.n_points)
        else:
            gaussian_noise = np.zeros(self.n_points)

        # 2. Add Baseline
        baseline = self.get_mixed_baseline() * 5.0 
        
        # 3. Combine initial noisy spectrum
        noisy_spectrum = signal + gaussian_noise + baseline

        # 4. Add Beta Noise (Artifacts) - FIX: Added to match training data
        # We apply the same 50% probability check to keep statistics consistent.
        if np.random.rand() < 0.5:
             noisy_spectrum = self._add_beta_noise(noisy_spectrum)

        return noisy_spectrum, signal, baseline

    # --- Helper Methods ---
    def _generate_mock_baselines(self, n=50):
        baselines = []
        x = np.linspace(0, 3.14, self.n_points)
        for _ in range(n):
            a, b = np.random.uniform(0.5, 5.0), np.random.uniform(0.5, 5.0)
            base = a * np.sin(x) + b * (x**2)
            base = (base - np.min(base)) / (np.max(base) - np.min(base))
            baselines.append(base)
        return np.array(baselines)

    def _find_nearest_idx(self, value):
        return (np.abs(self._raman_shift - value)).argmin()

    def pseudo_voigt(self, x_indices, center_idx, width_idx, eta):
        sigma = width_idx / 2.0
        if sigma < 1e-6: sigma = 1e-6
        diff = (x_indices - center_idx) / sigma
        L = 1 / (1 + diff ** 2)
        G = np.exp(-np.log(2) * diff ** 2)
        return eta * L + (1 - eta) * G

    # --- Cubic Spline Generator ---
    def _generate_spline_baseline(self):
        """Generates a smooth random baseline using Cubic Splines."""
        x_grid = np.arange(self.n_points)
        num_anchors = np.random.randint(3, 9) 
        
        # Random X-Coordinates (The Upgrade)
        middle_anchors = np.random.uniform(0, self.n_points - 1, num_anchors - 2)
        anchor_x = np.array([0] + sorted(middle_anchors) + [self.n_points - 1])
        
        anchor_y = np.random.uniform(0, 1.0, size=num_anchors)
        
        cs = CubicSpline(anchor_x, anchor_y)
        spline = cs(x_grid)
        
        if np.max(spline) - np.min(spline) > 1e-6:
            spline = (spline - np.min(spline)) / (np.max(spline) - np.min(spline))
        else:
            spline = np.zeros_like(spline)
            
        return spline

    def get_mixed_baseline(self):
        """Mixes Real Experimental Baselines with Synthetic Spline Baselines"""
        # A. Experimental Component
        if len(self.baselines) > 0:
            idx1, idx2 = np.random.choice(len(self.baselines), 2, replace=True)
            w = np.random.uniform(0, 1)
            real_base = w * self.baselines[idx1] + (1 - w) * self.baselines[idx2]
        else:
            real_base = np.zeros(self.n_points)
            
        # B. Spline Component
        synthetic_base = self._generate_spline_baseline()
        
        # C. Combine
        alpha = np.random.uniform(0.0, 0.3) 
        mixed = (1 - alpha) * real_base + (alpha * synthetic_base)
        
        if np.max(mixed) - np.min(mixed) > 1e-6:
            mixed = (mixed - np.min(mixed)) / (np.max(mixed) - np.min(mixed))
            
        return mixed

    # --- Beta Noise Generator (From OASIS Paper) ---
    def _add_beta_noise(self, current_spectrum):
        """
        Adds non-Gaussian artifacts ('dents' and 'spikes') using a Beta(1, 2) distribution.
        """
        # 1. Generate Beta Noise (0 to 1 range, skewed towards 0)
        beta_raw = np.random.beta(a=1, b=2, size=self.n_points)
        
        # 2. Randomly decide Magnitude
        if np.random.rand() < 0.5:
            mag = np.random.uniform(0.001, 0.004) # Small
        else:
            mag = np.random.uniform(0.005, 0.025) # Large
            
        # 3. Scale and Apply Direction
        signs = np.random.choice([-1, 1], size=self.n_points)
        beta_noise = beta_raw * mag * signs
        
        return current_spectrum + beta_noise

    # --- Random Spectrum Generation ---
    def generate_random_spectrum(self):
        # 1. Generate Signal or Blank

        idx = np.random.randint(0, len(self.noise_bank))
        real_noise_vector = self.noise_bank[idx]
        real_noise_rms = self.noise_rms[idx]



        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        if np.random.rand() < 0.1:
            n_peaks = 0 # Blank
        else:
            n_peaks = np.random.randint(5, 15) # Signal

        for _ in range(n_peaks):
            loc_cm = np.random.uniform(self._raman_shift[20], self._raman_shift[-20])
            center_idx = self._find_nearest_idx(loc_cm)
            log_min = np.log(3.0)  # Allow even sharper peaks (3 pixels)
            log_max = np.log(30.0) # Allow slightly broader peaks (40 pixels)
            width_idx = np.exp(np.random.uniform(log_min, log_max))
            
            
            #width_idx = np.random.uniform(3, 30)
            amp = np.random.uniform(0.1, 1.0)
            eta = np.random.uniform(0.5, 1.0)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, width_idx, eta)

        # 2. Add Baseline
        baseline_amp = np.random.uniform(2.0, 10.0) 
        baseline = self.get_mixed_baseline() * baseline_amp

        # 3. Add Standard Gaussian Noise (Sensor Thermal Noise)
        log_min = np.log(self.min_snr)
        log_max = np.log(self.max_snr)
        target_snr = np.exp(np.random.uniform(log_min, log_max))

        max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
        noise_sigma = max_sig / target_snr
        gaussian_noise = np.random.normal(0, noise_sigma, self.n_points)

        # 4. Add Beta Noise (Artifacts/Dents)
        noisy_spectrum = signal + gaussian_noise + baseline
        if np.random.rand() < 0.5:
            noisy_spectrum = self._add_beta_noise(noisy_spectrum)

        final_input = noisy_spectrum
        return final_input, signal, baseline

    def get_batch(self, batch_size=32, poly=True):
        batch_x, batch_y_s, batch_y_b = [], [], []
        
        for _ in range(batch_size):
            x, y_s, y_b = self.generate_random_spectrum()
            
            # Normalize
            min_val, max_val = np.min(x), np.max(x)
            if max_val - min_val > 1e-6:
                norm_x = (x - min_val) / (max_val - min_val)
                norm_b = (y_b - min_val) / (max_val - min_val)
                norm_s = (y_s / (max_val - min_val)) * 10.0 
            else:
                norm_x = np.zeros_like(x)
                norm_b = np.zeros_like(y_b)
                norm_s = np.zeros_like(y_s)
            
            batch_x.append(norm_x)
            batch_y_s.append(norm_s)
            batch_y_b.append(norm_b)

        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_s), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_b), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        x, y_s, y_b = self.generate_random_spectrum()
        min_val, max_val = np.min(x), np.max(x)
        if max_val - min_val > 1e-6:
            x = (x - min_val) / (max_val - min_val)
            y_s = (y_s / (max_val - min_val)) * 10.0
            y_b = (y_b - min_val) / (max_val - min_val)
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_b, dtype=torch.float32).unsqueeze(0)
        )
    
    @property
    def raman_shift(self):
        return self._raman_shift


import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import CubicSpline

class RamanDataGenerator(Dataset):
    def __init__(self, baseline_file_path, noise_file_path=None, epoch_size=1000, min_snr=5, max_snr=25,blank_ratio=0.05):
        """
        Args:
            baseline_file_path (str): Path to baseline .npz file.
            noise_file_path (str): Path to real noise .npz file (optional).
            epoch_size (int): Virtual size of dataset for one training epoch.
        """
        self.epoch_size = epoch_size
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.blank_ratio=blank_ratio
        # --- 1. Load Noise Bank (Dual Mode) ---
        self.noise_bank = None
        self.noise_rms = None
        
        if noise_file_path is not None:
            try:
                print(f"Loading real noise from: {noise_file_path}")
                data = np.load(noise_file_path)
                noise_data = data['data']
                self.noise_bank=noise_data[:,1:].T   
                # Pre-calculate RMS for faster scaling later
                self.noise_rms = np.std(self.noise_bank, axis=1)
                print(f"Loaded {len(self.noise_bank)} real noise vectors.")
                
            except Exception as e:
                print(f"Error loading noise file: {e}")
                print("Falling back to synthetic Beta/Gaussian noise.")
                self.noise_bank = None

        # --- 2. Load Baselines & Auto-Detect Points ---
        try:
            print(f"Loading baselines from: {baseline_file_path}")
            data = np.load(baseline_file_path)
            raw_data=data['data']
            # shape[0] is points, Col 0 is Shift, Cols 1..N are Baselines
            self.n_points = raw_data.shape[0] 
            self._raman_shift = raw_data[:, 0]
            self.baselines_raw = raw_data[:, 1:].T
            
            print(f"Auto-detected {self.n_points} spectral points.")
            print(f"Loaded {self.baselines_raw.shape[0]} baselines.")
            
            # Normalize Baselines
            self.baselines = []
            for b in self.baselines_raw:
                min_b = np.min(b)
                max_b = np.max(b)
                if max_b - min_b > 1e-6:
                    self.baselines.append((b - min_b) / (max_b - min_b))
                else:
                    self.baselines.append(np.zeros_like(b))
            self.baselines = np.array(self.baselines)
            
        except Exception as e:
            print(f"Warning: {e}")
            print("Falling back to synthetic data.")
            self.n_points = 864
            self._raman_shift = np.linspace(200, 3500, self.n_points)
            self.baselines = self._generate_mock_baselines(50)

    # --- HELPER: Unified Noise Generation ---
    def _get_noise_vector(self, target_snr, max_signal_amp):
        """
        Returns a noise vector based on availability of real noise bank.
        """
        # A. USE REAL NOISE (If available)
        if self.noise_bank is not None:
            # 1. Pick random real noise
            idx = np.random.randint(0, len(self.noise_bank))
            real_noise = self.noise_bank[idx]
            real_rms = self.noise_rms[idx]
            
            # 2. Scale to match Target SNR
            # Desired Noise RMS = Max_Signal / Target_SNR
            target_noise_rms = max_signal_amp / target_snr
            
            # Avoid division by zero
            if real_rms < 1e-9: real_rms = 1.0 
            
            scale_factor = target_noise_rms / real_rms
            return real_noise * scale_factor

        # B. USE SYNTHETIC NOISE (Fallback)
        else:
            # 1. Gaussian Thermal Noise
            noise_sigma = max_signal_amp / target_snr
            gaussian = np.random.normal(0, noise_sigma, self.n_points)
            
            # 2. Add Beta Noise (Artifacts)
            # Apply 50% chance of having beta artifacts
            final_noise = gaussian
            if np.random.rand() < 0.5:
                final_noise = self._add_beta_noise(final_noise)
                
            return final_noise

    # --- UPDATED: Defined Spectrum Generation ---
    def generate_defined_spectrum(self, peak_locs_cm, widths_idx, amplitudes, eta, snr=None,signal_scale=1.0):
        """
        Generates a spectrum with specific parameters using the unified noise logic.
        """
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        for loc_cm, w_idx, amp in zip(peak_locs_cm, widths_idx, amplitudes):
            center_idx = self._find_nearest_idx(loc_cm)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, w_idx, eta)

        # 1. Determine SNR
        # If user provided a specific SNR, use it. Otherwise pick random.
        if snr is None:
            if (self.min_snr==0) and (self.max_snr==0):
                snr=0.0
            else:
            
                log_min, log_max = np.log(self.min_snr), np.log(self.max_snr)
                snr = np.exp(np.random.uniform(log_min, log_max))

        signal=signal_scale*signal
        max_sig = np.max(signal) if np.max(signal) > 0 else 1.0

        # 2. Get Noise (Real or Synthetic)
        noise_vector = self._get_noise_vector(snr, max_sig)

        # 3. Add Baseline
        baseline = self.get_mixed_baseline() * 5.0 
        
        # 4. Combine
        noisy_spectrum = signal + noise_vector + baseline

        return noisy_spectrum, signal, baseline

    # --- UPDATED: Random Spectrum Generation ---
    def generate_random_spectrum(self):
        # 1. Generate Signal (Peaks)
        signal = np.zeros(self.n_points)
        x_indices = np.arange(self.n_points)
        
        if np.random.rand() < self.blank_ratio: # typically 10%
            n_peaks = 0 # Blank
        else:
            n_peaks = np.random.randint(5, 15) # Signal

        for _ in range(n_peaks):
            loc_cm = np.random.uniform(self._raman_shift[20], self._raman_shift[-20])
            center_idx = self._find_nearest_idx(loc_cm)
            
            # Log-uniform width sampling (sharp peaks preferred)
            log_min = np.log(3.0) 
            log_max = np.log(30.0)
            width_idx = np.exp(np.random.uniform(log_min, log_max))
            
            amp = np.random.uniform(0.1, 1.0)
            eta = np.random.uniform(0.5, 1.0)
            signal += amp * self.pseudo_voigt(x_indices, center_idx, width_idx, eta)

        # 2. Generate Baseline
        baseline_amp = np.random.uniform(2.0, 10.0) 
        baseline = self.get_mixed_baseline() * baseline_amp

        # 3. Generate Noise (Real or Synthetic)
        # Determine target SNR for this sample
        if (self.min_snr==0) and (self.max_snr==0):
            target_snr =0
        else:
            log_min = np.log(self.min_snr)
            log_max = np.log(self.max_snr)
            target_snr = np.exp(np.random.uniform(log_min, log_max))
        
        max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
        
        # Call the helper to get the noise vector
        noise_vector = self._get_noise_vector(target_snr, max_sig)

        # 4. Combine
        final_input = signal + baseline + noise_vector
        
        return final_input, signal, baseline

    # --- Helper Methods (Unchanged) ---
    def _generate_mock_baselines(self, n=50):
        baselines = []
        x = np.linspace(0, 3.14, self.n_points)
        for _ in range(n):
            a, b = np.random.uniform(0.5, 5.0), np.random.uniform(0.5, 5.0)
            base = a * np.sin(x) + b * (x**2)
            base = (base - np.min(base)) / (np.max(base) - np.min(base))
            baselines.append(base)
        return np.array(baselines)

    def _find_nearest_idx(self, value):
        return (np.abs(self._raman_shift - value)).argmin()

    def pseudo_voigt(self, x_indices, center_idx, width_idx, eta):
        sigma = width_idx / 2.0
        if sigma < 1e-6: sigma = 1e-6
        diff = (x_indices - center_idx) / sigma
        L = 1 / (1 + diff ** 2)
        G = np.exp(-np.log(2) * diff ** 2)
        return eta * L + (1 - eta) * G

    def _generate_spline_baseline(self):
        x_grid = np.arange(self.n_points)
        num_anchors = np.random.randint(3, 9) 
        middle_anchors = np.random.uniform(0, self.n_points - 1, num_anchors - 2)
        anchor_x = np.array([0] + sorted(middle_anchors) + [self.n_points - 1])
        anchor_y = np.random.uniform(0, 1.0, size=num_anchors)
        cs = CubicSpline(anchor_x, anchor_y)
        spline = cs(x_grid)
        if np.max(spline) - np.min(spline) > 1e-6:
            spline = (spline - np.min(spline)) / (np.max(spline) - np.min(spline))
        else:
            spline = np.zeros_like(spline)
        return spline

    def get_mixed_baseline(self):
        if len(self.baselines) > 0:
            idx1, idx2 = np.random.choice(len(self.baselines), 2, replace=True)
            w = np.random.uniform(0, 1)
            real_base = w * self.baselines[idx1] + (1 - w) * self.baselines[idx2]
        else:
            real_base = np.zeros(self.n_points)
        synthetic_base = self._generate_spline_baseline()
        alpha = np.random.uniform(0.0, 0.3) 
        mixed = (1 - alpha) * real_base + (alpha * synthetic_base)
        if np.max(mixed) - np.min(mixed) > 1e-6:
            mixed = (mixed - np.min(mixed)) / (np.max(mixed) - np.min(mixed))
        return mixed

    def _add_beta_noise(self, current_spectrum):
        beta_raw = np.random.beta(a=1, b=2, size=self.n_points)
        if np.random.rand() < 0.5:
            mag = np.random.uniform(0.001, 0.004)
        else:
            mag = np.random.uniform(0.005, 0.025)
        signs = np.random.choice([-1, 1], size=self.n_points)
        beta_noise = beta_raw * mag * signs
        return current_spectrum + beta_noise

    def get_batch(self, batch_size=32, poly=True):
        batch_x, batch_y_s, batch_y_b = [], [], []
        for _ in range(batch_size):
            x, y_s, y_b = self.generate_random_spectrum()
            min_val, max_val = np.min(x), np.max(x)
            if max_val - min_val > 1e-6:
                norm_x = (x - min_val) / (max_val - min_val)
                norm_b = (y_b - min_val) / (max_val - min_val)
                norm_s = (y_s / (max_val - min_val)) * 10.0 
            else:
                norm_x = np.zeros_like(x)
                norm_b = np.zeros_like(y_b)
                norm_s = np.zeros_like(y_s)
            batch_x.append(norm_x)
            batch_y_s.append(norm_s)
            batch_y_b.append(norm_b)
        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_s), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_b), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        x, y_s, y_b = self.generate_random_spectrum()
        min_val, max_val = np.min(x), np.max(x)
        if max_val - min_val > 1e-6:
            x = (x - min_val) / (max_val - min_val)
            y_s = (y_s / (max_val - min_val)) * 10.0
            y_b = (y_b - min_val) / (max_val - min_val)
        return (
            torch.tensor(x, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_s, dtype=torch.float32).unsqueeze(0),
            torch.tensor(y_b, dtype=torch.float32).unsqueeze(0)
        )
    
    @property
    def raman_shift(self):
        return self._raman_shift


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. Initialize (Make sure 'baselines.txt' exists!)
    # Format of file: Col 0 = Raman Shift, Cols 1..N = Intensity
    path = r'C:\Users\khsat\OneDrive\Documents\python codes\Raman Factory\Raman Data\Leslie F. (40-60s)20\balackWell_background\baseline_combined_data.txt'
    
    # NOTE: If the path is wrong, it will print a warning and use synthetic data automatically.
    gen = RamanDataGenerator(path)

    # 2. Define Peaks in cm^-1 (e.g., Phenylalanine ring breathing ~1000)
    my_peaks = [1002.5, 1450.0, 1600.0]
    
    spec, sig, base = gen.generate_defined_spectrum(
        peak_locs_cm=my_peaks, 
        widths_idx=[10, 15, 12], 
        amplitudes=[1.0, 0.5, 0.8], 
        eta=0.9, 
        snr=15
    )

    # 3. Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(gen.raman_shift, spec, color='black', alpha=0.5, label='Noisy Input')
    plt.plot(gen.raman_shift, sig, color='red', label='Target: Signal')
    plt.plot(gen.raman_shift, base, color='green', linestyle='--', label='Target: Mixed Baseline')
    
    # Mark where we asked for peaks
    for p in my_peaks:
        plt.axvline(x=p, color='blue', linestyle=':', alpha=0.5)

    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.title("Defined Spectrum (Peaks matched to Raman Shift)")
    plt.legend()
    plt.show()

    # 2. Get a batch using the CUSTOM function
    inputs, t_sig, t_base = gen.get_batch(batch_size=16)
    print(f"Batch Shape: {inputs.shape}") # Should be (16, 1, N_POINTS)







class CurriculumRamanGenerator(Dataset):
    def __init__(self, baseline_file_path, epoch_size=1000):
        """
        Args:
            baseline_file_path (str): Path to file with real baselines.
            epoch_size (int): Virtual size of dataset for one training epoch.
        """
        self.epoch_size = epoch_size
        self.current_epoch = 0  # Tracks training progress for curriculum
        self.baseline_file_path = baseline_file_path

        # --- 1. Load Data & Auto-Detect Points ---
        try:
            print(f"Loading baselines from: {baseline_file_path}")
            # Load all data (Assuming Col 0 = Raman Shift, Cols 1..N = Baselines)
            raw_data = np.loadtxt(baseline_file_path)
            
            # shape[0] is the number of rows (spectral points)
            self.n_points = raw_data.shape[0] 
            
            # Split: Col 0 is Raman Shift, Cols 1:end are Baselines
            self._raman_shift = raw_data[:, 0]
            self.baselines_raw = raw_data[:, 1:].T
            
            print(f"Auto-detected {self.n_points} spectral points.")
            print(f"Loaded {self.baselines_raw.shape[0]} real baselines.")
            
            # Normalize Baselines Immediately
            self.baselines = []
            for b in self.baselines_raw:
                min_b = np.min(b)
                max_b = np.max(b)
                if max_b - min_b > 1e-6:
                    self.baselines.append((b - min_b) / (max_b - min_b))
                else:
                    self.baselines.append(np.zeros_like(b))
            self.baselines = np.array(self.baselines)
            
        except Exception as e:
            print(f"Warning: Could not load baseline file. Error: {e}")
            print("Falling back to synthetic mock baselines.")
            self.n_points = 864
            self._raman_shift = np.linspace(200, 3500, self.n_points)
            self.baselines = self._generate_mock_baselines(50)

    # --- 2. Helper Methods (These were missing!) ---
    def _generate_mock_baselines(self, n=50):
        """Generates synthetic polynomial/sine baselines if file load fails."""
        baselines = []
        x = np.linspace(0, 3.14, self.n_points)
        for _ in range(n):
            a = np.random.uniform(0.5, 5.0)
            b = np.random.uniform(0.5, 5.0)
            # Mix of sine wave and quadratic curve
            base = a * np.sin(x) + b * (x**2)
            # Normalize 0-1
            base = (base - np.min(base)) / (np.max(base) - np.min(base))
            baselines.append(base)
        return np.array(baselines)

    def _find_nearest_idx(self, value):
        """Finds the index of the closest Raman shift value."""
        return (np.abs(self._raman_shift - value)).argmin()

    def pseudo_voigt(self, x_indices, center_idx, width_idx, eta):
        """Generates a single Raman peak shape."""
        sigma = width_idx / 2.0
        # Avoid division by zero
        if sigma < 1e-6: sigma = 1e-6
            
        diff = (x_indices - center_idx) / sigma
        L = 1 / (1 + diff ** 2)
        G = np.exp(-np.log(2) * diff ** 2)
        return eta * L + (1 - eta) * G

    def get_mixed_baseline(self):
        """Randomly mixes two real baselines to create variety."""
        if len(self.baselines) > 0:
            idx1, idx2 = np.random.choice(len(self.baselines), 2, replace=True)
            w = np.random.uniform(0, 1)
            return w * self.baselines[idx1] + (1 - w) * self.baselines[idx2]
        else:
            return np.zeros(self.n_points)

    # --- 3. Curriculum Logic ---
    def set_epoch(self, epoch):
        """Call this from train.py at the start of every epoch."""
        self.current_epoch = epoch

    def get_difficulty_params(self):
        """Returns SNR range and Blank Probability based on epoch."""
        if self.current_epoch < 30:
            # PHASE 1: WARMUP (Epoch 0-5)
            # Only Strong Signals. No Blanks.
            return {'blank_prob': 0.0, 'snr_range': (20.0, 60.0)}
            
        elif self.current_epoch < 60:
            # PHASE 2: REALITY (Epoch 6-20)
            # Standard Signals. Rare Blanks.
            return {'blank_prob': 0.05, 'snr_range': (8.0, 40.0)}
            
        else:
            # PHASE 3: MASTERY (Epoch 20+)
            # Full difficulty: Trace elements & Blanks.
            return {'blank_prob': 0.20, 'snr_range': (2.0, 40.0)}

    # --- 4. Main Generation Logic ---
    def generate_random_spectrum(self):
        params = self.get_difficulty_params()
        
        # A. Determine if Blank
        if np.random.rand() < params['blank_prob']:
            n_peaks = 0
            signal = np.zeros(self.n_points)
        else:
            n_peaks = np.random.randint(3, 16)
            signal = np.zeros(self.n_points)
            x_indices = np.arange(self.n_points)
            
            # Generate Peaks
            for _ in range(n_peaks):
                # Random location within range (avoiding extreme edges)
                loc_cm = np.random.uniform(self._raman_shift[20], self._raman_shift[-20])
                center_idx = self._find_nearest_idx(loc_cm)
                width_idx = np.random.uniform(5, 30)
                amp = np.random.uniform(0.1, 1.0)
                eta = np.random.uniform(0.5, 1.0)
                
                signal += amp * self.pseudo_voigt(x_indices, center_idx, width_idx, eta)

        # B. Generate Baseline
        baseline_amp = np.random.uniform(2.0, 10.0)
        baseline = self.get_mixed_baseline() * baseline_amp

        # C. Add Noise (Based on Curriculum SNR)
        min_snr, max_snr = params['snr_range']
        log_min = np.log(min_snr)
        log_max = np.log(max_snr)
        target_snr = np.exp(np.random.uniform(log_min, log_max))

        # Noise calculation
        max_sig = np.max(signal) if np.max(signal) > 0 else 1.0
        noise_sigma = max_sig / target_snr
        noise = np.random.normal(0, noise_sigma, self.n_points)

        final_input = signal + noise + baseline
        
        return final_input, signal, baseline

    # --- 5. Batch Generation ---
    def get_batch(self, batch_size=32):
        """Generates a processed batch of tensors."""
        batch_x, batch_y_s, batch_y_b = [], [], []
        
        for _ in range(batch_size):
            x, y_s, y_b = self.generate_random_spectrum()
            
            # Normalize Inputs
            min_val, max_val = np.min(x), np.max(x)
            
            # Avoid division by zero if input is flat (rare but possible)
            if max_val - min_val > 1e-6:
                norm_x = (x - min_val) / (max_val - min_val)
                norm_b = (y_b - min_val) / (max_val - min_val)
                
                # CRITICAL: Scale signal up by 10x for gradient stability
                norm_s = (y_s / (max_val - min_val)) * 10.0
            else:
                norm_x = np.zeros_like(x)
                norm_b = np.zeros_like(y_b)
                norm_s = np.zeros_like(y_s)

            batch_x.append(norm_x)
            batch_y_s.append(norm_s)
            batch_y_b.append(norm_b)

        # Convert to PyTorch Tensors (Batch, Channel, Points)
        return (
            torch.tensor(np.array(batch_x), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_s), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(batch_y_b), dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        """Standard DataLoader interface (optional usage)."""
        # Wraps get_batch logic for single item if needed
        x, y_s, y_b = self.get_batch(batch_size=1)
        return x[0], y_s[0], y_b[0]

    @property
    def raman_shift(self):
        return self._raman_shift