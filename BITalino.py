import json
import numpy as np
import bluetooth # pip3 install git+https://github.com/pybluez/pybluez.git
import serial
from serial.tools import list_ports
import time
import math

class BITalino:
    def __init__(self):
        self.socket = None
        self.analogChannels = []
        self.number_bytes = None
        self.macAddress = None
        self.serial = False
        self.returndata = None

    def find(self, serial=False):
        try:
            if serial:
                nearby_devices = list(port[0] for port in list_ports.comports() if 'bitalino' or 'COM' in port[0])
            else:
                nearby_devices = bluetooth.discover_devices(lookup_names=True)
            return nearby_devices
        except Exception as e:
            print(f"Error finding devices: {e}")
            return -1
    
    def open(self, macAddress=None, SamplingRate=1000):
        try:
            if macAddress:
                if ":" in macAddress and len(macAddress) == 17:
                    self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                    self.socket.connect((macAddress, 1))
                else:
                    self.socket = serial.Serial(macAddress, 115200)
                    self.serial = True
                time.sleep(2)

                # Configure sampling rate
                variableToSend = {1000: 0x03, 100: 0x02, 10: 0x01, 1: 0x00}.get(SamplingRate, None)
                if variableToSend is None:
                    raise ValueError(f"Invalid sampling rate {SamplingRate}")

                variableToSend = int((variableToSend << 6) | 0x03)
                self.write(variableToSend)
                return True
            else:
                raise TypeError("MAC address or serial port is needed to connect")
        except Exception as e:
            print(f"Error opening connection: {e}")
            return -1

    def start(self, analogChannels=[0, 1, 2, 3, 4, 5]):
        self.analogChannels = list(set(analogChannels))
        if len(self.analogChannels) == 0 or len(self.analogChannels) > 6:
            raise ValueError("Invalid analog channels")
        
        bit = 1
        for i in analogChannels:
            bit |= 1 << (2 + i)
        self.write(bit)
        return True

    def stop(self):
        self.write(0)
        return True

    def close(self):
        self.socket.close()
        return True

    def write(self, data=0):
        if self.serial:
            self.socket.write(bytes([data]))
        else:
            self.socket.send(bytes([data]))
        return True

    def read(self, nSamples=100):
        if self.socket is None:
            raise TypeError("An input connection is needed.")

        # Check if analogChannels is initialized and set to a valid list
        if not self.analogChannels:
            raise ValueError("Analog channels must be specified before reading.")

        nChannels = len(self.analogChannels)
        if nChannels <= 4:
            self.number_bytes = int(math.ceil((12 + 10 * nChannels) / 8))
        else:
            self.number_bytes = int(math.ceil((52 + 6 * (nChannels - 4)) / 8))

        if self.number_bytes is None:
            raise ValueError("Number of bytes ('self.number_bytes') has not been initialized.")

        dataAcquired = np.zeros((5 + nChannels, nSamples))  # Prepare a matrix to hold data

        # reading method chosen based on connection type
        if self.serial:
            print("Reading from serial...")
            reader = self.socket.read
        else:
            print("Reading from Bluetooth...")
            reader = self.socket.recv

        Data = b''
        sampleIndex = 0
        while sampleIndex < nSamples:
            while len(Data) < self.number_bytes:
                Data += reader(1)
                #print(Data)
            decoded = self.decode(Data) # decode the collected data
            if len(decoded) != 0:
                dataAcquired[:, sampleIndex] = decoded.T
                Data = b''
                sampleIndex += 1
            else:
                Data += reader(1)  # Continue reading until data is ready
                Data = Data[1:]  # remove the first byte if it's invalid, check if this is correct!
                print("ERROR DECODING")

        return dataAcquired


    def decode(self, data, nAnalog=None):
        if nAnalog == None: nAnalog = len(self.analogChannels)
        if nAnalog <= 4:
            number_bytes = int(math.ceil((12. + 10. * nAnalog) / 8.))
        else:
            number_bytes = int(math.ceil((52. + 6. * (nAnalog - 4)) / 8.))
        
        nSamples = len(data) // number_bytes
        res = np.zeros(((nAnalog + 5), nSamples))
        print(res.shape)
        print(res)
        
        j, x0, x1, x2, x3, out, inp, col, line = 0, 0, 0, 0, 0, 0, 0, 0, 0
        encode01 = 0x01
        encode03 = 0x03
        encodeFC = 0xFC
        encodeFF = 0xFF
        encodeC0 = 0xC0
        encode3F = 0x3F
        encodeF0 = 0xF0
        encode0F = 0x0F
        
        #CRC check
        CRC = data[j + number_bytes - 1] & encode0F
        for byte in range(number_bytes):
            for bit in range(7, -1, -1):
                inp = data[byte] >> bit & encode01
                if byte == (number_bytes - 1) and bit < 4:
                    inp = 0
                out = x3
                x3 = x2
                x2 = x1
                x1 = out^x0
                x0 = inp^out
 
        if CRC == ((x3<<3)|(x2<<2)|(x1<<1)|x0):
            try:
                def store(value): # function to write to result and increment line
                    nonlocal line
                    res[line, col] = value
                    line += 1

                # Sequence number
                store((data[j + number_bytes - 1] >> 4) & 0x0F)

                # Digital channels D0 to D3 from a single byte
                digital_byte = data[j + number_bytes - 2]
                for bit in range(7, 3, -1):  # bits 7 to 4
                    store((digital_byte >> bit) & 0x01)

                # Analog channel decoding rules
                analog_rules = [
                    [(-2, 0x0F, 6), (-3, 0xFC, -2)],    # A0
                    [(-3, 0x03, 8), (-4, 0xFF,  0)],    # A1
                    [(-5, 0xFF, 2), (-6, 0xC0, -6)],    # A2
                    [(-6, 0x3F, 4), (-7, 0xF0, -4)],    # A3
                    [(-7, 0x0F, 2), (-8, 0xC0, -6)],    # A4
                    [(-8, 0x3F, 0)]                     # A5 (only if 11 channels)
                ]

                #max_channels = 6 if res.shape[0] == 11 else 5
                max_channels = res.shape[0] - 5
                for i in range(max_channels):
                    value = 0
                    for byte_offset, mask, shift in analog_rules[i]:
                        part = data[j + number_bytes + byte_offset] & mask
                        if shift >= 0:
                            value |= part << shift
                        else:
                            value |= part >> -shift
                    store(value)

            except Exception as e:
                print(f"Exception decoding frame: {e}")
            return res
        # CRC check failed
        else:
            return []


