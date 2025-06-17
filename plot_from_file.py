import pandas as pd
import matplotlib.pyplot as plt
import mne
#from scipy.signal import butter, filtfilt, welch
#import scipy.signal as signal
from scipy import signal
#import seaborn as sns
#import neurokit2 as nk
import numpy as np

from matplotlib.widgets import CheckButtons
from SignalType import signal_types

#print(f"mne version {mne.__version__}")

def read_csv_file(filename):
    #df = pd.read_csv(f'data_files/data_recording_{filename}.csv')
    df = pd.read_csv(f'data_files/{filename}.csv')
    datatype = df.columns[1]  # 'eeg (μV)' | 'emg (mV)'
    data = df[datatype]
    return data, datatype, df

def plot_eeg_from_file(filename):
    data, datatype, df = read_csv_file(filename)
    
    df['Time (s)'] = pd.to_datetime(df['Time (s)'], unit='s')
    df.set_index('Time (s)', inplace=True)

    data = df[datatype].resample('3.90625ms').mean()  # resample to 256 Hz (3.90625 ms intervals)
    #data = data.resample('4ms').mean()  # resample to 250 Hz (4 ms intervals)   

    # interpolate EEG signal to uniform time grid
    data = data.interpolate(method='linear')  # linear interpolation  

    raw = mne.io.RawArray(data.values.reshape(1, -1), mne.create_info(ch_names=['EEG'], sfreq=250, ch_types='eeg'))

    # apply high-pass and notch filters with MNE
    raw.filter(l_freq=0.5, h_freq=None, fir_design='firwin')  # high-pass filter
    raw.notch_filter(freqs=50, fir_design='firwin')  # notch filter

    # downsampling
    #raw.resample(sfreq=246, npad='auto')  # to 246 Hz

    # band-bass filter for alpha and theta bands
    #raw.filter(l_freq=4, h_freq=12, fir_design='firwin')

    # ICA # TODO: needs multichannel data
    #raw_copy = raw.copy()
    #ica = mne.preprocessing.ICA(n_components=5, random_state=97, max_iter='auto')
    #ica.fit(raw)
    #ica.plot_components()
    #ica.plot_sources(raw)
    #ica.exclude = [0,1] # excludes components 0 and 1
    #raw = ica.apply(raw_copy, exclude=ica.exclude)

    # compute power spectral density (PSD) using Welch's method
    psd = raw.compute_psd(fmin=0.5, fmax=100, n_fft=1024) # n_fft 256, 512, 1024
    psd_data = psd.get_data()
    freqs = psd.freqs

    # plot PSD
    plt.figure(figsize=(15, 5))
    plt.semilogy(freqs, psd_data[0], label='EEG PSD', color='blue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (μV²/Hz)')
    plt.title(f'EEG Power Spectral Density from {filename}')
    plt.xlim(0.5, 70)
    plt.grid()
    # different background colors to bands
    plt.axvspan(0, 4, color='blue', alpha=0.3, label='delta (1-4 Hz)')
    plt.axvspan(4, 8, color='green', alpha=0.3, label='theta (4-8 Hz)')
    plt.axvspan(8, 12, color='lightgreen', alpha=0.3, label='alpha (8-12 Hz)')
    plt.axvspan(12, 30, color='yellow', alpha=0.3, label='beta (13-30 Hz)')
    plt.axvspan(30, 100, color='orange', alpha=0.3, label='gamma (30-100 Hz)') 
    plt.legend(['psd','delta','theta','alpha','beta', 'gamma'], loc='upper right')
    plt.tight_layout()
    #plt.show()

    plt.savefig(f'plots/{filename}_psd_plot.png')
    print(f"plot saved as plots/{filename}_psd_plot.png")

    return psd_data, freqs


    # extract alpha and theta bands
    # calculate alpha/theta ratio

    # Fourier transform to second segments of data
    # detect alfa and theta peaks
    # calculate alfa/theta ratio

def calculate_ratios(psd_data, freqs):

    #print(f"psd_data shape: {psd_data.shape}, freqs shape: {freqs.shape}")
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)
     
    # calculate band power
    def band_power(psd_data, freqs, band):
        band_indices = np.logical_and(freqs >= band[0], freqs < band[1])
        return np.sum(psd_data[:,band_indices])
    
    theta_power = band_power(psd_data, freqs, theta_band)
    alpha_power = band_power(psd_data, freqs, alpha_band)
    beta_power = band_power(psd_data, freqs, beta_band)

    return theta_power/beta_power, theta_power/alpha_power, alpha_power/beta_power
    
def get_phase_ratios(base_filename):
    phases = ['rest', 'stress', 'relax']
    ratios_and_power = {'theta_beta': [], 'theta_alpha': [], 'alpha_beta': [] ,
                        'theta_power': [], 'alpha_power': [], 'beta_power': []}
    #{'theta_beta': [], 'theta_alpha': [], 'alpha_beta': []}

    print(f"Plotting phase ratios for {base_filename}")

    # process each phase
    for phase in phases:
        #psd_data, freqs = plot_eeg_from_file(f'{filename}_{phase}')
        full_filename = f'{base_filename}_{phase}'
        psd_data, freqs = plot_eeg_from_file(f'{full_filename}')
        theta_beta, theta_alpha, alpha_beta = calculate_ratios(psd_data, freqs)
        ratios_and_power['theta_beta'].append(theta_beta)
        ratios_and_power['theta_alpha'].append(theta_alpha)
        ratios_and_power['alpha_beta'].append(alpha_beta)
        ratios_and_power['theta_power'].append(np.sum(psd_data[:, np.logical_and(freqs >= 4, freqs < 8)]))
        ratios_and_power['alpha_power'].append(np.sum(psd_data[:, np.logical_and(freqs >= 8, freqs < 13)]))
        ratios_and_power['beta_power'].append(np.sum(psd_data[:, np.logical_and(freqs >= 13, freqs < 30)]))

    return ratios_and_power

def plot_phases_from_file(filename):
    
    ratios = get_phase_ratios(filename)
    phases = ['rest', 'stress', 'relax']
    
    # Plotting
    x = np.arange(len(phases))
    width = 0.2

    fig, ax = plt.subplots()
    ax.bar(x - width, ratios['theta_beta'], width, label='theta/beta')
    ax.bar(x, ratios['theta_alpha'], width, label='theta/alpha')
    ax.bar(x + width, ratios['alpha_beta'], width, label='alpha/beta')

    ax.set_xlabel('phase')
    ax.set_ylabel('ratio')
    ax.set_title('EEG band power ratios')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{filename}_phase_ratios.png')
    print(f"Phase ratios plot saved as plots/{filename}_phase_ratios.png")
    #plt.show()
    return ratios


def plot_eeg_frequency_bands(filename):

    signal_type_key = filename[17:20]
    print(f"Signal type: {signal_type_key}")
    eeg_signal = signal_types.get(signal_type_key, signal_types['None'])
    sampling_rate = eeg_signal.sampling_rate
    #print(sampling_rate)
    #print(signal_type_key)
    df = pd.read_csv(f'data_files/data_recording_{filename}.csv')
    #df['eeg (μV)'] = df['eeg (μV)'].str[0]
    #df['eeg (μV)'] = df['eeg (μV)'].apply(lambda x: x[0])
    datatype = df.columns[1] # 'eeg (μV)' | 'emg (mV)'
    data = df[datatype] 

    #print(df.head())

    def bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        #print(f"b: {b}, a: {a}")
        #print(f"{type(b)}, {type(a)}")
        y = signal.filtfilt(b, a, data)
        return y

    def eeg_bandpass(data, band, fs, order=4):
        bands = {
            'alpha': (8, 12),
            'beta': (13, 30),
            'delta': (1, 4), 
            'gamma': (30, fs/2 - 1),  #  ylä fs/2-1 tai välillä 70-100 Hz
            'theta': (4, 8),
        }
        if band not in bands:
            raise ValueError(f"Unknown band: {band}. Choose from {list(bands.keys())}")
        lowcut, highcut = bands[band]
        return bandpass_filter(data, lowcut, highcut, fs, order)

    if datatype == 'eeg (μV)':
        alpha_signal = eeg_bandpass(data, 'alpha', sampling_rate)
        beta_signal = eeg_bandpass(data, 'beta', sampling_rate)
        delta_signal = eeg_bandpass(data, 'delta', sampling_rate)
        gamma_signal = eeg_bandpass(data, 'gamma', sampling_rate)
        theta_signal = eeg_bandpass(data, 'theta', sampling_rate)

        fig, ax = plt.subplots(figsize=(15, 5))
        lines = [] # store lines to list
        #lines.append(ax.plot(df['Time (s)'], df[datatype], label='raw EEG', color='#add8e6')[0])
        lines.append(ax.plot(df['Time (s)'], gamma_signal, label='gamma (30-50 Hz)', color='#deb887')[0])
        lines.append(ax.plot(df['Time (s)'], beta_signal, label='beta (13-30 Hz)', color='red')[0])
        lines.append(ax.plot(df['Time (s)'], alpha_signal, label='alpha (8-12 Hz)', color='green')[0])
        lines.append(ax.plot(df['Time (s)'], theta_signal, label='theta (4-8 Hz)', color='orange')[0])
        lines.append(ax.plot(df['Time (s)'], delta_signal, label='delta (1-4 Hz)', color='purple')[0])

        # checkButtons
        labels = [line.get_label() for line in lines]
        visibility = [line.get_visible() for line in lines]
        colors = [line.get_color() for line in lines]

        # inset axes for checkboxes
        rax = ax.inset_axes([0.01, 0.75, 0.14, 0.25])  # [left, bottom, width, height]
        check = CheckButtons(
            rax, labels, visibility,
            label_props={'color': colors},
            check_props={'facecolor': colors},
            frame_props={'edgecolor': colors}
        )

        def func(label):
            index = labels.index(label)
            lines[index].set_visible(not lines[index].get_visible())
            plt.draw()

        check.on_clicked(func)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'EEG ({sampling_rate} Hz) frequency bands')
        plt.tight_layout()
        plt.show()

    elif datatype == 'emg (mV)' or datatype == 'EMG (mV)':
        emg_rectified = np.abs(df[datatype])
        window_size = int(0.05 * sampling_rate)  # 50 ms window
        emg_envelope = np.convolve(emg_rectified, np.ones(window_size)/window_size, mode='same')

        # frequency analysis with Welch's method
        frequencies, power_spectrum = signal.welch(df[datatype], fs=sampling_rate, nperseg=1024)
        mean_freq = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum)
        cumulative_power = np.cumsum(power_spectrum)
        median_freq = frequencies[np.where(cumulative_power >= cumulative_power[-1]/2)[0][0]]

        # plot raw and envelope
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(df['Time (s)'], df[datatype], label='Raw EMG', color='#add8e6')
        ax.plot(df['Time (s)'], emg_envelope, label='EMG Envelope', color='orange')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title('Raw EMG and Envelope')

        # add frequency info
        legend_text = f'Mean Freq: {mean_freq:.1f} Hz\nMedian Freq: {median_freq:.1f} Hz'
        ax.legend(loc='upper right')
        ax.text(0.98, 0.02, legend_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

        plt.tight_layout()
        plt.show()
    
    '''

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['Time (s)'], df[datatype], label='Raw EEG', color='#add8e6')
    ax.plot(df['Time (s)'], gamma_signal, label='Gamma (30-70 Hz)', color='tan')
    ax.plot(df['Time (s)'], beta_signal, label='Beta (13-30 Hz)', color='red')
    ax.plot(df['Time (s)'], delta_signal, label='Delta (1-4 Hz)', color='purple')
    ax.plot(df['Time (s)'], theta_signal, label='Theta (4-8 Hz)', color='orange')
    ax.plot(df['Time (s)'], alpha_signal, label='Alpha (8-12 Hz)', color='green')



    ax.legend(loc='upper right')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Raw EEG and Frequency Bands')
    plt.tight_layout()
    plt.show()
    '''

    #alpha_wave = nk.signal_filter(data, sampling_rate=250, lowcut=8, highcut=12, method="butterworth", order=2)
    # Plot the original and filtered signals
    #nk.signal_plot([data, alpha_wave], labels=["Raw Signal", "Alpha Waves"])
    '''
    alpha_signal = bandpass_filter(data, 8, 12, sampling_rate)
    plt.plot(alpha_signal)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df['Time (s)'], df[datatype])
    plt.show()
    '''


if __name__ == "__main__":


    #filename = '2025-05-23_09-59_eeg' #09-39
    filename = 'data_recording_2025-05-23_09-59_eeg' #09-39
    #filename = '2025-06-11_11-18_eeg' # vaihiet
    #filename = 'data_recording_2025-06-11_11-34_eeg' # vaihiet

    #get_phase_ratios(filename)
    plot_phases_from_file(filename)

    #plot_eeg_from_file(filename)


    #plot_eeg_frequency_bands(filename)
