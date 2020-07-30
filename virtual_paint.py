import numpy as np
import cv2
from collections import deque


blueLower = np.array([105, 50, 50])
blueUpper = np.array([125, 255, 255])


kernel = np.ones((5, 5), np.uint8)


bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

bindex = 0
gindex = 0
rindex = 0
yindex = 0


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0


Paint = np.zeros((471,636,3)) + 255
Paint = cv2.circle(Paint, (90,61), 48, (0,0,0), 2)
Paint = cv2.circle(Paint, (200,61), 48, colors[0], 2)
Paint = cv2.circle(Paint, (315,61), 48, colors[1], 2)
Paint = cv2.circle(Paint, (430,61), 48, colors[2], 2)


cv2.putText(Paint, "CLEAR ALL", (49, 73), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(Paint, "BLUE", (185, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2, cv2.LINE_AA)
cv2.putText(Paint, "GREEN", (298, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1], 2, cv2.LINE_AA)
cv2.putText(Paint, "RED", (420, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[2], 2, cv2.LINE_AA)


cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


camera = cv2.VideoCapture(0)

while True:

    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    frame = cv2.circle(frame, (90, 61), 50, (0, 0, 0), 2)
    frame = cv2.circle(frame, (200, 61), 50, colors[0], 2)
    frame = cv2.circle(frame, (315, 61), 50, colors[1], 2)
    frame = cv2.circle(frame, (430, 61), 50, colors[2], 2)


    cv2.putText(frame, "CLEAR ALL", (49, 73), cv2.FONT_ITALIC, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[0], 2, cv2.LINE_AA)
    cv2.putText(frame, "GREEN", (298, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[1], 2, cv2.LINE_AA)
    cv2.putText(frame, "RED", (420, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[2], 2, cv2.LINE_AA)



    if not grabbed:
        break


    blueMask = cv2.inRange(hsv, blueLower, blueUpper)


    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)


    (cnts, _) = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None


    if len(cnts) > 0:

        cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] <= 105:
            if 40 <= center[0] <= 140: # Clear All
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                bindex = 0
                gindex = 0
                rindex = 0
                yindex = 0

                Paint[107:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 # Blue
            elif 275 <= center[0] <= 370:
                    colorIndex = 1 # Green
            elif 390 <= center[0] <= 485:
                    colorIndex = 2 # Red

        else :
            if colorIndex == 0:
                bpoints[bindex].appendleft(center)
            elif colorIndex == 1:
                gpoints[gindex].appendleft(center)
            elif colorIndex == 2:
                rpoints[rindex].appendleft(center)

    else:
        bpoints.append(deque(maxlen=512))
        bindex += 1
        gpoints.append(deque(maxlen=512))
        gindex += 1
        rpoints.append(deque(maxlen=512))
        rindex += 1
        

    point = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(point)):
        for j in range(len(point[i])):
            for k in range(1, len(point[i][j])):
                if point[i][j][k - 1] is None or point[i][j][k] is None:
                    continue
                cv2.line(frame, point[i][j][k - 1], point[i][j][k], colors[i], 2)
                cv2.line(Paint, point[i][j][k - 1], point[i][j][k], colors[i], 2)


    cv2.imshow("Photo Tracker", frame)
    cv2.imshow("Paint image", Paint)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break