#---1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

img = cv.imread('image.jpg')
#img = cv.imread('CasinoRoyale.jpg')
#img = cv.imread('Quantum_of_Solace.jpg')
#img = cv.imread('SkyFall.jpg')
#img = cv.imread('Spectre.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5) # default setting
#faces = face_cascade.detectMultiScale(gray, 1.8, 5) # Casino Royale
#faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Quantum of Solace
#faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Skyfall
#faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Spectre
print(len(faces))

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    eye_gray = gray[y:y+h, x:x+w]
    eye_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(eye_gray)
    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(eye_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
