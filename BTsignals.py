'''
$ uvicorn BTsignals.py:app --reload port --8000
$ sudo /home/m/anaconda3/envs/biosig/bin/uvicorn BTsignals:app --reload --port 8000
'''
from fastapi import FastAPI # $ pip install fastapi uvicorn
from pydantic import BaseModel
from BITalino import BITalino  # custom BITalino class from BITalino.py

app = FastAPI()

class BITalinoRequest(BaseModel):
    macAddress: str
    samplingRate: int
    recordingTime: int


@app.post("/bitalino-data/")
async def get_bitalino_data(request: BITalinoRequest):
    device = BITalino()
    
    try:
        # Open the device with parameters
        device.open(request.macAddress, request.samplingRate)
        print(f"BITalino: {device}")
        # Start the device
        device.start([0, 1, 2, 3, 4, 5])  # Adjust according to actual channels needed
        
        # Read the data
        nSamples = request.samplingRate * request.recordingTime
        dataAcquired = device.read(nSamples)
        
        # Stop the device after reading data
        device.stop()
        device.close()

        # data to JSON
        print(dataAcquired)
        
        data = {
            "Version": "1.0",
            "Digital Channels": {
                "D0": dataAcquired[1].tolist(),
                "D1": dataAcquired[2].tolist(),
                "D2": dataAcquired[3].tolist(),
                "D3": dataAcquired[4].tolist()
                },
            "Analog Channels": {
                "A1": dataAcquired[5].tolist(),
                "A2": dataAcquired[6].tolist(),
                "A3": dataAcquired[7].tolist(),
                "A4": dataAcquired[8].tolist(),
                "A5": dataAcquired[9].tolist(),
                "A6": dataAcquired[10].tolist()
                }
            }            

        return data
    except Exception as e:
        return {"error": str(e)}

# Endpoint to start Bitalino device and fetch data
# GET http://127.0.0.1:8000/bitalino-get/?macAddress=<mac-address>&samplingRate=100&recordingTime=5

@app.get("/bitalino-get/")
#async def get_bitalino_data(params: BITalinoRequest):
async def bitalino_data(macAddress: str, samplingRate: int, recordingTime: int):
    # Initialize BITalino instance
    device = BITalino()

    # Set analog channels, here as an example (you can modify as needed)
    device.analogChannels = [0, 1, 2]  # Set channels you want to read (A0, A1, A2)
    
    # Open the device with the parameters
    device.open(macAddress=macAddress, SamplingRate=samplingRate)

    # Start the device with the analog channels
    device.start([0, 1, 2])

    # Read the data for the specified recording time
    nSamples = samplingRate * recordingTime  # Total samples to read based on time and rate
    data = device.read(nSamples=nSamples)

    # Return the data in JSON format
    return {
        "macAddress": macAddress,
        "samplingRate": samplingRate,
        "recordingTime": recordingTime,
        "data": data.tolist()  # Convert the NumPy array to a list for JSON serialization
    }
