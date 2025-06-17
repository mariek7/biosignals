import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json
from datetime import datetime
from SignalType import signal_types,eeg_transfer, ecg_transfer, emg_transfer, acc_transfer # SignalType.py

from dotenv import load_dotenv
import os

def realtime_plot_and_save(phase=None):

    load_dotenv()
    date_and_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    mac_address = os.getenv('MAC_ADDRESS')
    signal_type_key = os.getenv('signal_type','None') #  ecg|eeg|emg|acc|raw, if none then raw
    print(f"Signal type: {signal_type_key}")

    signal = signal_types.get(signal_type_key, signal_types['None'])
    sampling_rate = signal.sampling_rate
    signal_unit= signal.unit
    transfer_func = signal.transfer_function

    print(f"Timestamp: {date_and_time}")
    print(f"MAC Address: {mac_address}")
    print(f"Signal: {signal.name} (sampling rate {sampling_rate}, unit {signal.unit})")
    print(f"Transfer function: {transfer_func.__name__}")

    # TODO: multichannel
    channel_to_plot = 'A1' # A1|A2|A3|A4|A5|A6

    data_buffer = []
    time_buffer = []
    all_data = [] # for saving to file
    all_time = []
    t = 0
    dt =1.0/sampling_rate

    fig, ax = plt.subplots(figsize=(10, 5))
    line, = ax.plot([], [], lw=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(signal_unit)
    ax.set_ylim(signal.ylim) # from SignalType.py
    #ax.set_ylim(-5,5)

    time_buffer = []
    data_buffer = []
    window_seconds = 2

    stop = False
    def on_key(event):
        global stop
        if event.key == ' ':  # spacebar
            print("Stopping --")
            stop = True


    def animate(frame):
        global stop
        try:
            response = requests.get(f"http://localhost:8000/bitalino-get/?macAddress={mac_address}&samplingRate={sampling_rate}&recordingTime=1")
            #response = requests.get(f"http://localhost:8000/bitalino-get/?recordingTime=1")
            
            #all = response.json()
            #data_got = all['data']
            #print(f"Sampling rate {all['samplingRate']}")

            all = json.loads(response.text)
            if "error" in all:
                print("API error:", all["error"])
                print("Details:", all.get("detail", "No details provided"))
                return

            data_got = all.get("data", [])
            channel_data= np.array(data_got[5])

            if not response.text.strip():   
                print("Empty response received from server.")
                return
            
            #if not data_got.any():
            #    print("No 'data' field in response or it's empty.")
            #    return

            #column_names = ['seqN', 'D0', 'D1', 'D2', 'D3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']
            # Check if data_got has the expected number of columns
            #if len(data_got) == 0 or len(data_got[0]) != len(column_names):
            #    print(f"Data format error: expected columns do not match, data_got has {len(data_got[0])} columns")
            #    return
            
            #df = pd.DataFrame(list(zip(*data_got)), columns=column_names)
            #df = pd.DataFrame(channel_data)
            channels_to_plot = ['A1','A2']
            print(data_got.shape)
            all_df = pd.DataFrame(data_got, columns=['seqN', 'D0', 'D1', 'D2', 'D3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6'])
            df = all_df[channels_to_plot]
            print(df.head())
            #data_from_channel = df[channel_to_plot]
            #transfered_data = transfer_func(data_from_channel))
            transfered_data = transfer_func(df) 
            #print(transfered_data[10:20])
            #transfered_data = ecg_transfer(df)
            #transfered_data = df
            n_samples = len(transfered_data)
            #n_samples = len(df)
            global t
            times = np.arange(t, t + n_samples * dt, dt)
            t += n_samples * dt

            #if any(val > threshold for val in transfered_data):
            #    print('-- beep --')

            # update buffers for saving
            all_data.extend(transfered_data)
            #all_data.extend(df)
            all_time.extend(times)

            # update buffers
            time_buffer.extend(times)
            data_buffer.extend(transfered_data)
            #data_buffer.extend(df)
            if len(time_buffer) > sampling_rate * window_seconds:
                time_buffer[:] = time_buffer[-sampling_rate*window_seconds:]
                data_buffer[:] = data_buffer[-sampling_rate*window_seconds:]

            # update plot
            line.set_data(time_buffer, data_buffer)
            ax.relim()
            ax.autoscale_view()

            if stop:
                anim.event_source.stop()  # end animation loop
            return line,
        except Exception as e:
            print("Error fetching or processing data:", e)

    fig.canvas.mpl_connect('key_press_event', on_key)
    anim = animation.FuncAnimation(fig, animate, interval=200, cache_frame_data=False)
    plt.show()

    #df = pd.DataFrame({'Time (s)': all_time, f'{signal.name} ({signal_unit})': all_data})
    #df.to_csv(f'data_files/data_recording_{signal.name}_{date_and_time}.csv', index=False)

    filename = f'data_recording_{date_and_time}_{signal.name}'
    df = pd.DataFrame({'Time (s)': all_time, f'{signal.name} ({signal_unit})': all_data})
    df.to_csv(f'data_files/{filename}_{phase}.csv', index=False)

    return filename

if __name__ == "__main__":
    filename = realtime_plot_and_save()
    print(f"Data saved to file")