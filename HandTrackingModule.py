# we are going to use MediaPipe(developed by google)
import cv2 as cv
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpdraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, frame, draw=True):
        imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(frame)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(
                        frame, handLms, self.mphands.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            # for only one hand
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, ch = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv.circle(frame, (cx, cy), 10, (0, 0, 255), cv.FILLED)

        return self.lmList

    def fingersUp(self):
        fingers = []

        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #4 Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    pTime = 0
    cTime = 0

    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        success, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv.putText(frame, f'{int(fps)}', (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                   (255, 0, 255), thickness=3)

        cv.imshow("Image", frame)
        if cv.waitKey(1) & 0xFF == ord('d'):
            break


# if __name__ == "__main__":
#     main()
