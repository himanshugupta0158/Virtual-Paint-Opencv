from turtle import color
import cv2 as cv
import mediapipe as mp
import HandTrackingModule as htm
import os
import numpy as np
import time

eraserthickness = 100
brushthickness = 15
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

folderPath = "header"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

# print(len(overlayList))

header = overlayList[0]
cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.5)
drawColor = (255, 0, 255)

xp, yp = 0, 0

while True:
    #1.> Import image
    success, img = cap.read()
    img = cv.flip(img, 1)  # flipping image

    # 2.> Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        # print(lmList)

        # tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3.> Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4.> If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")
            # checking for click
            if y1 < 125:
                if 30 < x1 < 70:
                    cap.release()
                    break
                elif 210 < x1 < 340:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                elif 480 < x1 < 570:
                    header = overlayList[1]
                    drawColor = (255, 0, 0)
                elif 680 < x1 < 810:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 900 < x1 < 1020:
                    header = overlayList[3]
                    drawColor = (255, 0, 255)
                elif 1150 < x1 < 1280:
                    header = overlayList[4]
                    drawColor = (0, 0, 0)

            cv.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv.FILLED)

        # 5.> if Drawing Mode - Index finger is up
        if fingers[1] and fingers[2] == False:
            print("Drawing Mode")
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)

            # drawing lines but differently
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv.line(img, (xp, yp), (x1, y1), drawColor, eraserthickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1),
                        drawColor, eraserthickness)
            else:
                cv.line(img, (xp, yp), (x1, y1), drawColor, brushthickness)
                cv.line(imgCanvas, (xp, yp), (x1, y1),
                        drawColor, brushthickness)
            xp, yp = x1, y1

    imgGray = cv.cvtColor(imgCanvas, cv.COLOR_BGR2GRAY)
    _, imgInv = cv.threshold(imgGray, 50, 255, cv.THRESH_BINARY_INV)
    imgInv = cv.cvtColor(imgInv, cv.COLOR_GRAY2BGR)
    # all color draw -> black draw
    # black -> white
    # below will just add drawn one onthing else
    img = cv.bitwise_and(img, imgInv)
    # adding/replacing black to color draw
    img = cv.bitwise_or(img, imgCanvas)

    # setting the header image
    img[0:125, 0:1280] = header
    # this will add and blend two different images.
    # img = cv.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv.putText(img, "Touch Logo to exit", (30, 690),
               cv.FONT_HERSHEY_PLAIN, 2, (100, 100, 100), 2)
    cv.imshow("Virtual Paint", img)
    # cv.imshow("Image Canvas",imgCanvas)
    cv.waitKey(1)
