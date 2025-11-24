import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = float(detectionCon)
        self.trackCon = float(trackCon)
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]  # Fingertip landmark IDs

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=False):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        """Returns list of which fingers are up [thumb, index, middle, ring, pinky]"""
        fingers = []
        if len(self.lmList) == 0:
            return fingers

        # Thumb (check x-coordinate for left/right hand)
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 Fingers (check y-coordinate)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img=None, draw=True):
        """Calculate distance between two landmarks"""
        if len(self.lmList) == 0:
            return 0, img, []

        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)

        if img is not None and draw:
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

        return length, img, [x1, y1, x2, y2, cx, cy]

    def recognizeGesture(self):
        """Recognize hand gestures based on finger positions"""
        if len(self.lmList) == 0:
            return "No Hand Detected"

        fingers = self.fingersUp()

        # Thumbs Up: Only thumb up
        if fingers == [1, 0, 0, 0, 0]:
            return "Thumbs Up 👍"

        # Peace Sign: Index and middle fingers up
        if fingers == [0, 1, 1, 0, 0]:
            return "Peace Sign ✌️"

        # Pointing: Only index finger up
        if fingers == [0, 1, 0, 0, 0]:
            return "Pointing 👆"

        # All fingers up
        if fingers == [1, 1, 1, 1, 1]:
            return "Open Hand 🖐️"

        # Fist: No fingers up
        if fingers == [0, 0, 0, 0, 0]:
            return "Fist ✊"

        # OK Sign: Check if thumb and index form circle
        if len(self.lmList) >= 9:
            distance, _, _ = self.findDistance(4, 8, draw=False)
            if distance < 40 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
                return "OK Sign 👌"

        # Rock Sign: Index and pinky up
        if fingers == [0, 1, 0, 0, 1]:
            return "Rock On 🤘"

        return "Unknown Gesture"


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)

    # Set camera resolution for better performance
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Recognize gesture
        gesture = detector.recognizeGesture()

        # Display information
        cv2.putText(img, f"FPS: {int(fps)}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(img, f"Gesture: {gesture}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

        # Show finger count
        if len(lmList) != 0:
            fingers = detector.fingersUp()
            fingerCount = fingers.count(1)
            cv2.putText(img, f"Fingers Up: {fingerCount}", (10, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Gesture Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()