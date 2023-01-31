import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
previousTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for HandLms in results.multi_hand_landmarks:
            for id, lm in enumerate(HandLms.landmark):
                #print(id, lm)  # NOTE that this will print the decimel values
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)   # NOTE We used this operation to convert the decimil values into pixels to get more clear information about landmarks
                if id==0:           # Here we specified that if the id of the circle is 0 then we do some specific operation on it like it's size or color and so on
                    cv2.circle(img, (cx, cy), 15, (255, 240, 51), cv2.FILLED)
            mpDraw.draw_landmarks(img, HandLms, mpHands.HAND_CONNECTIONS)         # We used this mpDraw.draw_landmarks(img, HandLms) to draw our hands with circles and we wil use


    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
                                                                            # this mpDraw.draw_landmarks(img, HandLms, mpHands.HAND_CONNECTIONS) to add connections to circles


    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)              # We use these lines of code by default to run the webcam
    #print(results.multi_hand_landmarks)         # We will print the results to see if there's hands or not
