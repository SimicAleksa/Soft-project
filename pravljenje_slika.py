import cv2
import os
import uuid  # za kreiranje jedinstvenih imena
from skimage import io, transform

# VideoCapture(<camNUM>) kod camNUM-a moze doci do greske jer gadjamo pogresnu kameru, on se resava promenom vrednosti camNUM-a
videoPath = "minusR.mp4"
cap = cv2.VideoCapture(videoPath)  # dobijamo pristup nasoj web kameri
frame_num = 0
cap.set(1, frame_num)
while True:  # imamo loop frejmova dobijenih od web kamere
    frame_num += 1
    grabbed, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame = cap.read()  # citamo frejm
    # frejmovi nisu formata 250x250px, zato ih rucno podesavamo
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.resize(frame_gray, (128, 128))

    cv2.imshow('Web cam', frame_gray)  # prikaz prozora live feed-as
    # za kreiranje positive i anchor slika koristimo lib uuid - koja obezbedjuje jedinstvene nazive
    if cv2.waitKey(0) & 0XFF == ord('a'):  # za pravljenje chor slika
        # kreiramo jedinstvenu putanju na kojoj cemo da cuvamo sliku
        imgname = os.path.join("napravljeneSlike", '{}_minusR.jpg'.format(uuid.uuid1()))
        # samo cuvanje slike
        cv2.imwrite(imgname, frame_gray)
    if cv2.waitKey(0) & 0XFF == ord('b'):  # za pravljenje anchor slika
        pass
    if cv2.waitKey(0) & 0XFF == ord('x'):  # za pravljenje anchor slika
        cap.release()  # prekidamo konekciju sa web kamerom
        cv2.destroyAllWindows()  # zatvaramo prozor live feed-a




