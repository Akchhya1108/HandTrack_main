import cv2
import numpy as np
import time
import math
from HandTrackModule import handDetector

# Platform-specific volume control
try:
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    WINDOWS_AUDIO = True
except:
    WINDOWS_AUDIO = False
    print("⚠️ Windows audio libraries not available. Using simulation mode.")


class VolumeControlSystem:
    """
    Gesture-based volume control system demonstrating HCI application.
    Uses pinch gesture (thumb-index distance) to control system volume.
    """
    
    def __init__(self):
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.wCam, self.hCam = 640, 480
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        
        # Hand detector
        self.detector = handDetector(maxHands=1, detectionCon=0.7)
        
        # Volume control setup
        self.setupVolumeControl()
        
        # Performance tracking
        self.pTime = 0
        
        # Volume bar visualization
        self.volBar = 400
        self.volPer = 0
        
        # Color scheme
        self.colorInactive = (255, 0, 0)
        self.colorActive = (0, 255, 0)
        
    def setupVolumeControl(self):
        """Initialize platform-specific volume control"""
        if WINDOWS_AUDIO:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
                
                # Get volume range
                volRange = self.volume.GetVolumeRange()
                self.minVol = volRange[0]
                self.maxVol = volRange[1]
                
                print("✓ Windows audio control initialized")
                self.simulationMode = False
            except Exception as e:
                print(f"⚠️ Audio control error: {e}")
                print("Running in simulation mode")
                self.setupSimulationMode()
        else:
            self.setupSimulationMode()
    
    def setupSimulationMode(self):
        """Setup simulation mode for non-Windows platforms"""
        self.simulationMode = True
        self.minVol = -65
        self.maxVol = 0
        self.currentVol = -20
        print("✓ Simulation mode initialized")
    
    def setVolume(self, volumeLevel):
        """Set system volume or simulate"""
        if self.simulationMode:
            self.currentVol = volumeLevel
        else:
            try:
                self.volume.SetMasterVolumeLevel(volumeLevel, None)
            except:
                self.currentVol = volumeLevel
    
    def getVolume(self):
        """Get current volume level"""
        if self.simulationMode:
            return self.currentVol
        else:
            try:
                return self.volume.GetMasterVolumeLevel()
            except:
                return self.currentVol
    
    def calculateDistance(self, lmList):
        """Calculate pinch distance between thumb and index finger"""
        if len(lmList) < 9:
            return 0, 0, 0, 0, 0, 0, 0
        
        # Thumb tip (4) and Index finger tip (8)
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        length = math.hypot(x2 - x1, y2 - y1)
        
        return x1, y1, x2, y2, cx, cy, length
    
    def drawVolumeBar(self, img):
        """Draw volume visualization bar"""
        # Volume bar background
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 3)
        
        # Volume bar fill
        cv2.rectangle(img, (50, int(self.volBar)), (85, 400), self.colorActive, cv2.FILLED)
        
        # Volume percentage circle
        color = self.colorActive if self.volPer > 0 else self.colorInactive
        cv2.circle(img, (67, int(self.volBar)), 15, color, cv2.FILLED)
        
        # Volume percentage text
        cv2.putText(img, f'{int(self.volPer)}%', (40, 450), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        # Labels
        cv2.putText(img, 'VOL', (45, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def run(self):
        """Main control loop"""
        print("\n🔊 Volume Control System Started!")
        print("📋 Controls:")
        print("  - Pinch thumb and index finger to control volume")
        print("  - Closer = Lower volume")
        print("  - Further = Higher volume")
        print("  - Press 'q' to quit")
        if self.simulationMode:
            print("  ⚠️ Running in SIMULATION mode (no actual volume change)")
        print("-" * 50)
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("❌ Failed to capture frame")
                break
            
            img = cv2.flip(img, 1)
            
            # Detect hands
            img = self.detector.findHands(img, draw=True)
            lmList = self.detector.findPosition(img, draw=False)
            
            if len(lmList) != 0:
                # Get pinch distance
                x1, y1, x2, y2, cx, cy, length = self.calculateDistance(lmList)
                
                # Visual feedback
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # Color based on distance
                if length < 50:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
                else:
                    cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
                
                # Map hand distance to volume range
                # Hand range: 30 to 200 pixels
                # Volume range: minVol to maxVol
                vol = np.interp(length, [30, 200], [self.minVol, self.maxVol])
                self.volBar = np.interp(length, [30, 200], [400, 150])
                self.volPer = np.interp(length, [30, 200], [0, 100])
                
                # Set volume
                self.setVolume(vol)
                
                # Display distance
                cv2.putText(img, f'Distance: {int(length)}px', (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw volume bar
            self.drawVolumeBar(img)
            
            # Calculate FPS
            cTime = time.time()
            fps = 1 / (cTime - self.pTime) if (cTime - self.pTime) > 0 else 0
            self.pTime = cTime
            
            # Display information
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            mode_text = "Mode: SIMULATION" if self.simulationMode else "Mode: ACTIVE"
            cv2.putText(img, mode_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(img, "Pinch to control volume", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            cv2.imshow("Volume Control System", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Volume Control System stopped")


def main():
    controller = VolumeControlSystem()
    controller.run()


if __name__ == "__main__":
    main()