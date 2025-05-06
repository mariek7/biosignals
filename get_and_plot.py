import requests
import pandas as pd
import matplotlib.pyplot as plt
from transfer_functions import biosignal_transfer # transfer_functions.py

response = requests.get("http://127.0.0.1:8000/bitalino-get/?macAddress=98:D3:C1:FE:04:08&samplingRate=100&recordingTime=5")
all = response.json()

#print(data.keys()) # dict_keys(['macAddress', 'samplingRate', 'recordingTime', 'data'])
data = all['data']
sampling_rate = all['samplingRate']
column_names = ['seqN', 'D0', 'D1', 'D2', 'D3', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6']

df = pd.DataFrame(list(zip(*data)), columns=column_names)
a2 = df['A2']
print(a2)
ecg_func = biosignal_transfer('ecg')
ecg_data = ecg_func(a2)

plt.plot(ecg_data)
plt.show()