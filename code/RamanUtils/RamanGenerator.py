
import numpy as np
from RamanUtils.RamanHelpers import normalize



def lorentzian(x, x0, gamma, A):
    return A * gamma**2 / ((x - x0)**2 + gamma**2)

def generate_clean_spectrum(length=1024, peaks=[1],raman_shifts=[200],widths=[0.5],max_shift=1800):
    x = max_shift*np.linspace(0, 1, length)
    spectrum = np.zeros_like(x)
    for idx,p in enumerate(peaks):         # amplitude
        spectrum += lorentzian(x, raman_shifts[idx], widths[idx], p)
    normalize(spectrum)
    return x,spectrum

def generate_noise(length=1024,power_db=60,power_ref=1):

    # --- Convert dBm to Watts ---
    power_watts = power_ref* (10 ** ((power_db) / 10))

    # --- Compute power per sample ---
    power_per_sample = power_watts / length
    std_dev = np.sqrt(power_per_sample)

    # --- Generate Gaussian noise ---
    noise = np.random.normal(loc=0.0, scale=std_dev, size=length)  # reduced sample size for performance
    return noise

