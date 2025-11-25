import cv2
import numpy as np
import pyautogui
import time
from HandTrackModule import handDetector

class VirtualMouseController:
    """
    Human-Computer Interaction system using hand gestures to control mouse.
    Demonstrates scalable CV application for real-world use cases.
    """
    
    def __init__(self, frameReduction=100, smoothening=7):
        self.frameReduction = frameReduction
        self.smoothening = smoothening
        
        # Screen dimensions
        self.wScr, self.hScr = pyautogui.size()
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.wCam, self.hCam = 640, 480
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        
        # Hand detector
        self.detector = handDetector(maxHands=1, detectionCon=0.7)
        
        # Smoothening variables
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        
        # Control state
        self.clickState = False
        self.lastClickTime = 0
        self.clickCooldown = 0.5  # seconds
        
        # Performance tracking
        self.pTime = 0
        
        # PyAutoGUI settings
        pyautogui.FAILSAFE = False
        
    def smoothenMovement(self, x, y):
        """Apply smoothening to reduce jitter"""
        self.clocX = self.plocX + (x - self.plocX) / self.smoothening
        self.clocY = self.plocY + (y - self.plocY) / self.smoothening
        self.plocX, self.plocY = self.clocX, self.clocY
        return int(self.clocX), int(self.clocY)
    
    def mapToScreen(self, x, y):
        """Map camera coordinates to screen coordinates"""
        # Only use the inner frame (excluding borders)
        x = np.interp(x, (self.frameReduction, self.wCam - self.frameReduction), 
                      (0, self.wScr))
        y = np.interp(y, (self.frameReduction, self.hCam - self.frameReduction), 
                      (0, self.hScr))
        return x, y
    
    def detectGesture(self, fingers):
        """
        Detect control gestures:
        - Index up only: Move cursor
        - Index + Middle up: Click
        """
        gesture = "None"
        action = None
        
        if fingers[1] == 1 and fingers[2] == 0:  # Only index up
            gesture = "Move"
            action = "move"
        elif fingers[1] == 1 and fingers[2] == 1:  # Index + Middle up
            gesture = "Click"
            action = "click"
        
        return gesture, action
    
    def run(self):
        """Main control loop"""
        print("🖱️ Virtual Mouse Controller Started!")
        print("📋 Controls:")
        print("  - Index finger up: Move cursor")
        print("  - Index + Middle fingers up: Click")
        print("  - Press 'q' to quit")
        print("-" * 50)
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("❌ Failed to capture frame")
                break
            
            img = cv2.flip(img, 1)  # Mirror for natural interaction
            
            # Detect hands
            img = self.detector.findHands(img, draw=True)
            lmList = self.detector.findPosition(img, draw=False)
            
            # Initialize display variables
            gesture = "No Hand"
            status = "Waiting..."
            
            if len(lmList) != 0:
                fingers = self.detector.fingersUp()
                gesture, action = self.detectGesture(fingers)
                
                # Get index finger tip position (landmark 8)
                x1, y1 = lmList[8][1], lmList[8][2]
                
                if action == "move":
                    # Convert to screen coordinates
                    x3, y3 = self.mapToScreen(x1, y1)
                    
                    # Apply smoothening
                    x3, y3 = self.smoothenMovement(x3, y3)
                    
                    # Move mouse
                    pyautogui.moveTo(self.wScr - x3, y3)
                    
                    status = f"Moving cursor to ({int(x3)}, {int(y3)})"
                    cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                
                elif action == "click":
                    # Get middle finger tip position (landmark 12)
                    x2, y2 = lmList[12][1], lmList[12][2]
                    
                    # Calculate distance between index and middle finger
                    length = np.hypot(x2 - x1, y2 - y1)
                    
                    # If fingers are close together, perform click
                    currentTime = time.time()
                    if length < 60 and (currentTime - self.lastClickTime) > self.clickCooldown:
                        pyautogui.click()
                        self.lastClickTime = currentTime
                        status = "✓ CLICKED!"
                        cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                    else:
                        status = "Click gesture ready"
                        cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
                    
                    # Draw line between fingers
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Draw control zone
            cv2.rectangle(img, (self.frameReduction, self.frameReduction),
                         (self.wCam - self.frameReduction, self.hCam - self.frameReduction),
                         (255, 0, 255), 2)
            
            # Calculate FPS
            cTime = time.time()
            fps = 1 / (cTime - self.pTime) if (cTime - self.pTime) > 0 else 0
            self.pTime = cTime
            
            # Display information
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Gesture: {gesture}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(img, f"Status: {status}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Instructions
            cv2.putText(img, "Use control zone for better accuracy", (10, self.hCam - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Virtual Mouse Controller", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Virtual Mouse Controller stopped")


def main():
    controller = VirtualMouseController(frameReduction=100, smoothening=10)
    controller.run()


if __name__ == "__main__":
    main()