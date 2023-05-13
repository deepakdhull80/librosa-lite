import numpy as np

#TODO : hard coded number should provide justification in docstring
def mel_filter(
        n_mel, sr, nfft, minf=None, maxf=None
        ):
    '''
    ## Mel filterbank
    The Mel-scale aims to mimic the non-linear human ear perception of sound, by being more discriminative at lower frequencies
    and less discriminative at higher frequencies.


    #### formula for mel filter cofficient:
    H(i,k) = 0,                              if k < f(i-1)
            (k - f(i-1))/(f(i) - f(i-1)),    if f(i-1) <= k < f(i)
            (f(i+1) - k)/(f(i+1) - f(i)),    if f(i) <= k < f(i+1)
            0,                               if k >= f(i+1)

    '''

    if minf is None:
        minf = 0
    if maxf is None:
        maxf = 2595 * np.log10(1 + (sr / 2) / 700)    
    
    mel_points = np.linspace(minf, maxf, n_mel + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((nfft + 1) * hz_points / sr)
    
    filters = np.zeros((n_mel, int(np.floor(nfft/2 + 1)))) # nfft/2 because of fft.rfft()
    for m in range(1, n_mel + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])

        for k in range(f_m_minus, f_m):
            filters[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            filters[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    return filters