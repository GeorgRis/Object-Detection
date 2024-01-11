import cv2
import uuid
import os 
import time


labels = ['tommel_opp','tommel_ned']
number_imgs = 8

IMAGE_P = os.path.join('Tensorflow','workspace','images','collectimages')
print(IMAGE_P)
if not os.path.exists(IMAGE_P):
    if os.name == 'nt':
        os.makedirs(IMAGE_P)
        
for label in labels:
    path = os.path.join(IMAGE_P, label)
    if not os.path.exists(path):
        os.makedirs(path)
        
for label in labels:
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGE_P,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


for label in labels:
    path = os.path.join(IMAGE_P, label)
    if not os.path.exists(path):
        os.makedirs(path)
        
#Må labele de til xml
#også legge til i train og test(kan eventuelt legge til python skript for det)
        
TRAIN_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'train')
TEST_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'test')
ARCHIVE_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'archive.tar.gz')
