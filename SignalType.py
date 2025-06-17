import numpy as np
from scipy.signal import butter, filtfilt

class SignalType:
    def __init__(self, name, unit, ylim, sampling_rate, transfer_function=None):
        self.name = name
        self.unit = unit
        self.ylim = ylim
        self.sampling_rate = sampling_rate
        self.transfer_function = transfer_function or (lambda x: x)

    def apply_transfer(self, adc_data):
        return self.transfer_function(adc_data)

# transfer functions
def ecg_transfer(adc_data, adc_bits=8, vcc=3.0, gain=1900):#1019):
    adc_data = np.asarray(adc_data, dtype=np.float32)
    adc_max = 2**adc_bits - 1
    ecg_v = ((adc_data / adc_max) - 0.5) * vcc / gain
    return ecg_v * 1000  # mV
    #print(adc_data)
    #return adc_data

def eeg_transfer(adc_data, n=10, vcc=3.3, g_eeg=41782):
    #adc_data = np.asarray(adc_data, dtype=np.float32)  ##
    adc_data = np.asarray(adc_data).flatten().astype(np.float32) ## flatten
    eeg_v = ((adc_data / (2**n-1)) - 0.5) * vcc / g_eeg
    return eeg_v * 1e6  # μV

# The number of bits for each channel depends on the resolution of the Analog-to-Digital Converter (ADC); in BITalino the first four channels are sampled using 10-bit resolution (𝑛 = 10), while the last two may be sampled using 6-bit (𝑛 = 6).
def emg_transfer(adc_data, n=6, vcc=3.3, g_emg=1009):
    adc_data = np.asarray(adc_data, dtype=np.float32)
    emg_v = ((adc_data / (2**n-1)) - 0.5) * vcc / g_emg
    return emg_v * 1000  # mV

def acc_transfer(adc_data, c_min=2, c_max=4, scale=6.0):
    adc_data = np.asarray(adc_data, dtype=np.float32)
    return ((adc_data - c_min) / (c_max - c_min)) * scale - (scale / 2)

signal_types = {
    #'ecg': SignalType('ecg', 'mV', (-1.5, 1.5), 1000, transfer_function=ecg_transfer),
    'ecg': SignalType('ecg', 'mV', (-200, 200), 1000, transfer_function=ecg_transfer),
    'eeg': SignalType('eeg', 'μV', (-40, 40), 100, transfer_function=eeg_transfer),
    'emg': SignalType('emg', 'mV', (-20, 20), 1000, transfer_function=emg_transfer), # lim -1.64mV,+1.64mV
    'acc': SignalType('acc', 'g', (-3, 3), 100, transfer_function=acc_transfer),
    'None': SignalType('raw', None, (-45, 45), 100, None)
}
