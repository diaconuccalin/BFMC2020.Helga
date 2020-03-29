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


# Config
enableLateralControl    =   True
enableStream            =   False


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


# Processes:
# Movement control
cfProc = MovementControl([lcR], [cfS])
allProcesses.append(cfProc)

# Serial handler
shProc = SerialHandler([cfR], [])
allProcesses.append(shProc)

if enableLateralControl:
    camOutPs = []

    if enableStream:
        camOutPs = [lkS, camStS]
        # Camera streamer
        streamProc = CameraStreamer([camStR], [])
        allProcesses.append(streamProc)
    else:
        camOutPs = [lkS]

    # Camera process
    camProc = CameraProcess([],camOutPs)
    allProcesses.append(camProc)

    # Lane keeping
    lkProc = LaneKeeping([lkR], [lcS])
    allProcesses.append(lkProc)


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
