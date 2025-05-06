# transfer_functions.py
import numpy as np

def biosignal_transfer(datatype='eeg'):
    """
    Returns transfer function for given type.
    Usage:
        ecg_func = biosignal_transfer('ecg')
        ecg_data = ecg_func(raw_data)
    """
    def ecg_transfer(adc_data, adc_bits=16, vcc=3.0, gain=1019):
        adc_data = np.asarray(adc_data, dtype=np.float32)
        adc_max = 2**adc_bits - 1
        ecg_v = ((adc_data / adc_max) - 0.5) * vcc / gain
        return ecg_v * 1000  # mV
    ''' n = number of bits for each channel, depends on the resolution of the Analog-to-Digital
    Converter (ADC); in biosignalsplux the default is 16-bit resolution (n = 16), although 12-bit
    (n = 12) and 8-bit (n = 8) may also be found on older devices '''


    '''
    [-39.49𝜇V, 39.49𝜇V]
    EEG(V) = (((ADC/2**n)-0.5)*VCC)/G_eeg
    EEG(𝜇V) = EEG(V)*1*10^6
    VCC = 3.3V (operating voltage)
    G_eeg = 41782 (sensor gain)
    EEG (V) = EEG value in Volt (V)
    EEG(𝜇V) = EEG value in microvolt (𝜇𝑉)
    ADC = Value sampled from the channel
    n = Number of bits of the channel, depends on the resolution of the Analog-to-Digital Converter (ADC); in BITalino the first four channels are sampled using 10-bit resolution (n = 10), while the last two may be sampled using 6-bit (n = 6).
    '''
    def eeg_transfer(adc_data, n=10, vcc=3.3, g_eeg=41782):
        """
        Convert raw ADC data to EEG in microvolts.
        Parameters:
            adc_data : raw ADC values
            n       : int, number of ADC bits (default 10)
            vcc     : float, operating voltage (default 3.3V)
            g_eeg   : float, sensor gain (default 41782)
        Returns:
            eeg_uv  : numpy array or float, EEG in microvolts (μV)
        """
        import numpy as np
        adc_data = np.asarray(adc_data, dtype=np.float32)
        eeg_v = ((adc_data / (2**n)) - 0.5) * vcc / g_eeg
        eeg_uv = eeg_v * 1e6
        return eeg_uv
    

    '''
    [-3g, +3g] range
    x = 2
    ACC(g) = ((ADC-C_min)/C_max -C_min)*x -(x/2)
    ACC(g) = ACC value in g-force
    ADC = value sampled from the channel
    C_min = minimum calibration value
    C_max = maximum calibration value
    '''
    def acc_transfer(adc_data, c_min=2, c_max=4, scale=6.0):
        """
        Convert raw ADC data to acceleration in g-force.
        Parameters:
            adc_data : raw ADC values
            c_min   : float, calibration minimum (ADC value at -3g)
            c_max   : float, calibration maximum (ADC value at +3g)
            scale   : float, full scale range (default 6g, when [-3g, +3g])
        Returns:
            acc_g   : numpy array or float, acceleration in g
        """
        import numpy as np
        adc_data = np.asarray(adc_data, dtype=np.float32)
        acc_g = ((adc_data - c_min) / (c_max-c_min)) * scale - (scale/2)
        return acc_g

    if datatype.lower() == 'ecg':
        return ecg_transfer
    elif datatype.lower() == 'eeg':
        return eeg_transfer
    elif datatype.lower() == 'acc':
        return acc_transfer
    else:
        raise ValueError("Unknown datatype. Use 'ecg', 'eeg' or 'acc'.")
