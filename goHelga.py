import sys
sys.path.append('.')

import time
import signal

from multiprocessing import Pipe, Process, Event 

from hardware.nucleo.movementControl                import MovementControl
from hardware.nucleo.serialhandler.serialhandler    import SerialHandler
from hardware.camera.cameraprocess                  import CameraProcess
from imageprocessing.lateralcontrol.laneKeeping     import LaneKeeping
from hardware.camera.camerastreamer                 import CameraStreamer
from imageprocessing.signdetection.signDetection    import SignDetection


# Config
enableStream            =   False
enableLateralControl    =   False
enableSignDetection     =   True

# Initilize processes
allProcesses = list()


# Pipes:
# Camera process -> Image streamer
camStR, camStS = Pipe(duplex = False)

# Camera process -> Lane keeping
lkR, lkS = Pipe(duplex = False)

# Lane keeping -> Movement control
lcR, lcS = Pipe(duplex = False)

# Movement control -> Serial handler
cfR, cfS = Pipe(duplex = False)

# Camera process -> Sign detection
sdR, sdS = Pipe(duplex = False)

# Pipe collections
movementControlR = []
camOutPs = []

# Processes:
if enableStream:
    camOutPs.append(camStS)

    streamProc = CameraStreamer([camStR], [])
    allProcesses.append(streamProc)

if enableLateralControl:
    camOutPs.append(lkS)
    movementControlR.append(lcR)

    lkProc = LaneKeeping([lkR], [lcS])
    allProcesses.append(lkProc)

# Sign detection
if enableSignDetection:
    camOutPs.append(sdS)

    sdProc = SignDetection([sdR], [])
    allProcesses.append(sdProc)

# Movement control
cfProc = MovementControl(movementControlR, [cfS])
allProcesses.append(cfProc)

# Serial handler
shProc = SerialHandler([cfR], [])
allProcesses.append(shProc)

# Camera process
if enableStream or enableSignDetection or enableLateralControl:
    camProc = CameraProcess([],camOutPs)
    allProcesses.append(camProc)


# Start processes
print("Starting the processes!",allProcesses)
for proc in allProcesses:
    proc.daemon = True
    proc.start()


# Wait for keyboard interruption
blocker = Event()

try:
    blocker.wait()
except KeyboardInterrupt:
    print("\nCatching a KeyboardInterruption exception! Shutdown all processes.\n")
    for proc in allProcesses:
        if hasattr(proc,'stop') and callable(getattr(proc,'stop')):
            print("Process with stop",proc)
            proc.stop()
            proc.join()
        else:
            print("Process witouth stop",proc)
            proc.terminate()
            proc.join()
