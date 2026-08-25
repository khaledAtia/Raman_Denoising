# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 08:03:38 2022

@author: ARC338
"""

import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import statistics
from scipy.special import wofz
import os
import random
import string

def generate_random_code():
  """Generates a random 6-character code with 3 letters and 3 digits."""
  
  # Get 3 random uppercase letters
  letters = random.choices(string.ascii_uppercase, k=3)
  
  # Get 3 random digits
  digits = random.choices(string.digits, k=3)
  
  # Combine the letters and digits into one list
  combined_list = letters + digits
  
  # Shuffle the list to mix the letters and digits randomly
  random.shuffle(combined_list)
  
  # Join the list elements into a single string and return it
  return "".join(combined_list)




def find_peak_index(raman_shift:np.ndarray,spectrum,peak:float,win_size:float=5):
    indces=np.arange(raman_shift.shape[0])
    max_lim=peak+win_size/2
    min_lim=peak-win_size/2
    W=(raman_shift<max_lim) & (raman_shift>min_lim)
    ram_win=raman_shift[W]
    spec_win=spectrum[W]
    q=np.max(spec_win)==spectrum
    idx=indces[q]
    return idx

def extract_number(s):
    matches = re.findall(r'\d+', s)  # Find all numbers
    return int(matches[-1]) if matches else None  # Return the last number found

def ramanshift_to_wavelength(ramanshift_cm, laser_wavelength_nm=785):
    # Convert Raman shift from cm^-1 to nm^-1
    ramanshift_nm = ramanshift_cm * 1e-7
    
    # Apply the formula to get the scattered wavelength in nm
    scattered_wavelength_nm = 1 / ((1 / laser_wavelength_nm) - ramanshift_nm)
    
    return scattered_wavelength_nm

def find_raman_resolution(ramanshift1_cm,ramanshift2_cm,laser_wavelength_nm=785):
    lam1=ramanshift_to_wavelength(ramanshift1_cm, laser_wavelength_nm)
    lam2=ramanshift_to_wavelength(ramanshift2_cm, laser_wavelength_nm)
    return np.abs(lam1-lam2)


def listOfFiles(dir,f="*.txt"):
    txtfiles = []
    for file in glob.glob(dir+f):
        txtfiles.append(file)
    return txtfiles



def Xrelate(a,b):
    sa=np.correlate(a,a)
    an=a/np.sqrt(sa)
    sb=np.correlate(b,b)
    bn=b/np.sqrt(sb)
    return np.correlate(an,bn)

def normalize(spectrum,ax=None):
    """
    normalize a raman spectrum
    """
    
    Smin=np.min(spectrum,axis=ax)
    spectrum=(spectrum-Smin)
    Smax=np.max(spectrum,axis=ax)
    return((spectrum)/(Smax-0))


def denoise(spectrum,width=100,typ='h'):
    Yfft=np.fft.fft(spectrum)
    Yfft_shifted=np.fft.fftshift(Yfft)
    K=len(Yfft)
    C=int(np.floor(K/2))
    if typ=='l':
        Wind=np.zeros(K)
        Wind[C-width:C+width]=1
    elif typ=='h':
        Wind=np.ones(K)
        Wind[C-width:C+width]=0
    filtered=Yfft_shifted*Wind
    Yfft=np.fft.ifftshift(filtered)
    spectrum=np.fft.ifft(Yfft)
    return spectrum.real

def processBG(background,cutoff=20):
    X=preprocess(background)
    return(denoise(X,cutoff))
    

def preprocess(spectrum):
    A=normalize(spectrum)
    return A

def G_shape(x,xo=0, alpha=0.1):
    """ Return Gaussian line shape at x with HWHM alpha """
    return np.sqrt(np.log(2) / np.pi) / alpha\
                             * np.exp(-((x-xo) / alpha)**2 * np.log(2))

def L_shape(x,xo=0, gamma=0.1):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / ((x-xo)**2 + gamma**2)

def V_shape(x,xo=0, alpha=0.1, gamma=0.1):
    """
    Return the Voigt line shape at x with Lorentzian component HWHM gamma
    and Gaussian component HWHM alpha.

    """
    sigma = alpha / np.sqrt(2 * np.log(2))

    return np.real(wofz(((x-xo) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)

def loadDataFolder(folderPath):
    dataList=os.listdir(folderPath)
    dataset=[]
    names=[]
    for ii,fileName in enumerate(dataList):
        A=np.loadtxt(folderPath+fileName)
        dataset.append(A)
        names.append(fileName)
    
    return dataset,names



def remove_laser(data,spectrum_cut=250.0)->np.ndarray:
    w=data[:,0]>spectrum_cut
    data=data[w,:]
    return data



def kalmanFilter(z,Q=1e-5,R=0.001):
    # intial parameters
    n_iter=z.shape[0]
    sz = (n_iter,) # size of array

    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    #R = 0.1**2 # estimate of measurement variance, change to see effect

    # intial guesses
    xhat[0] = 0.0
    P[0] = 0.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q

        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
        
    return xhat

def loadData(zstr_dataPath):
    data=[]
    if (os.path.isfile(zstr_dataPath)):
        a=(np.loadtxt(zstr_dataPath))
        data.append(a)

    elif (os.path.isdir(zstr_dataPath)):
        files = [f for f in os.listdir(zstr_dataPath) if os.path.isfile(f)]
        data=[]
        for file in files:
            a=(np.loadtxt(zstr_dataPath+'/'+file))
            data.append(a)
    else:
        print('empty directory !!')
    return data


def averageData(data):
    N=len(data)
    av_data=np.zeros(data[0].shape)
    av_data[:,0]=data[0][:,0]
    for ii in np.arange(N):
        av_data[:,1]=av_data[:,1]+normalize(data[ii][:,1])
    av_data[:,1]=av_data[:,1]/N
    return av_data

def lens_na(focal:float,diameter:float,refractive_index=1):
    """_summary_

    Args:
        focal (float): _description_
        diameter (float): _description_
        refractive_index (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    na=refractive_index/np.sqrt(1+(2*focal/diameter)^2)
    return na


def find_Raman_peak():
    pass


def get_shift_indices(raman_shifts, start_shift, stop_shift):
    """
    Finds the array indices corresponding to a specific Raman shift range.
    
    Parameters:
    raman_shifts (numpy array): The full array of Raman shifts (e.g., 250 to 1800).
    start_shift (float/int): The beginning of the desired range.
    stop_shift (float/int): The end of the desired range.
    
    Returns:
    numpy array: An array of integer indices.
    """
    # Create a condition that checks if values are within the range (inclusive)
    condition = (raman_shifts >= start_shift) & (raman_shifts <= stop_shift)
    
    # Extract and return the indices where the condition is True
    indices = np.where(condition)[0]
    
    return indices


    