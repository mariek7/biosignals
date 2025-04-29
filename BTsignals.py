from fastapi import FastAPI # $ pip install fastapi uvicorn
from pydantic import BaseModel
from BITalino import BITalino  # custom BITalino class from BITalino.py

app = FastAPI()

class BITalinoRequest(BaseModel):
    macAddress: str
    samplingRate: int
    recordingTime: int

# GET from /bitalino-get/?macAddress=<mac-address>&samplingRate=<sr>&recordingTime=<rt>
@app.get("/bitalino-get/")
#async def get_bitalino_data(params: BITalinoRequest):
async def bitalino_data(macAddress: str, samplingRate: int, recordingTime: int):
    device = BITalino() # initialize BITalino instance
    device.analogChannels = [0, 1, 2]  # Set channels you want to read (A0, A1, A2)
    device.open(macAddress=macAddress, SamplingRate=samplingRate)
    device.start([0, 1, 2]) # start the device with the analog channels
    nSamples = samplingRate * recordingTime  # total samples to read based on time and rate
    data = device.read(nSamples=nSamples) # read data for given time

    return {
        "macAddress": macAddress,
        "samplingRate": samplingRate,
        "recordingTime": recordingTime,
        "data": data.tolist()  # Convert the NumPy array to a list for JSON serialization
    }

# POST to get data from /bitalino-data/
@app.post("/bitalino-data/")
async def get_bitalino_data(request: BITalinoRequest):
    device = BITalino()
    
    try:
        device.open(request.macAddress, request.samplingRate)
        print(f"BITalino: {device}")
        # start device
        device.start([0, 1, 2, 3, 4, 5])  # adjust channels needed
        
        # read data
        nSamples = request.samplingRate * request.recordingTime
        dataAcquired = device.read(nSamples)
        print(f"Data acquired: {dataAcquired}")
        print(dataAcquired.shape)
        print(dataAcquired.type)
        
        # stop device after data read
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

