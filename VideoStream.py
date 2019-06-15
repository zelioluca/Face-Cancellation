import numpy as np
import cv2 as cv
import cvlib as c


def simpleFaceCancellation():
    """This function cancel the face of people"""
    cap = cv.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        faces, confidences = c.detect_face(frame)
        print("Score: " + str(confidences))
        for(x, y, w, h) in faces:
            sub_face = frame[y:y + h, x:x + w]
            sub_face = cv.blur(sub_face, (100,100), cv.BORDER_DEFAULT)
            frame[y: y + sub_face.shape[0], x: x + sub_face.shape[1]] = sub_face
            cv.putText(frame, "Score: " + str(confidences), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break 
    cap.release()
    cv.destroyAllWindows()

def simpleFaceBoxing():
    """This function try to draw a box around faces"""
    cap = cv.VideoCapture(0)

    while(cap.isOpened()):
        rect, frame = cap.read()
        faces, confidences = c.detect_face(frame)
        print("Score is: " + str(confidences))
        for(startX, startY, endX, endY) in faces:
            cv.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 5)
            cv.putText(frame, "Score: " + str(confidences), (startX, startY), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break 

    cap.release()
    cv.destroyAllWindows()


def simpleFaceCancellationII():
    """This function cancel the face of people"""
    cap = cv.VideoCapture(0)

    while(cap.isOpened()):
        ret, frame = cap.read()
        faces, confidences = c.detect_face(frame)
        print("Score: " + str(confidences))
        for(x, y, w, h) in faces:
            frame[y:h, x:w] =cv.blur(frame[y:h, x:w], (100,100), cv.BORDER_DEFAULT)
            cv.putText(frame, "Score: " + str(confidences), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break 
    cap.release()
    cv.destroyAllWindows()

