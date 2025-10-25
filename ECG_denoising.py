from scipy import signal

def ECG_deno(ECG, fs):
    hpf = 0.9
    lpf = 40

    b, a = signal.butter(2, lpf / (fs / 2), 'lowpass')
    ECG = signal.filtfilt(b, a, ECG)

    b, a = signal.butter(2, hpf / (fs / 2), 'highpass')
    ECG = signal.filtfilt(b, a, ECG)

    b, a = signal.butter(2, [59 / (fs / 2), 61 / (fs / 2)], 'bandstop')
    ECG = signal.filtfilt(b, a, ECG)  # lfilter

    b, a = signal.butter(2, [49 / (fs / 2), 51 / (fs / 2)], 'bandstop')
    ECG = signal.filtfilt(b, a, ECG)  # lfilter

    return ECG

