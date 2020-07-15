

from keras.models import load_model
import cv2
import numpy as np
import time 



model = load_model('model-017 (1).model')

haar_face_cls = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
#address = "https://##########/video"
#cap.open(address)

label_dict = {0:'MASK',1:'NO MASK'}
color_dict = {0:(255,255,0),1:((0,0,255))}
frame_id = 0
time_now = time.time()



while True:
  frame , image = cap.read()
  frame_id +=1
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  face = haar_face_cls.detectMultiScale(gray,1.3,5)

  

  for x,y,w,h in face:

    img_f = gray[y:y+w,x:x+w]
    resize_img = cv2.resize(img_f,(100,100))
    norm = resize_img / 255.0
    reshape_img = np.reshape(norm,(1,100,100,1))
    result = model.predict(reshape_img)


    label = np.argmax(result,axis=1)[0]

    cv2.rectangle(image,(x,y),(x+w,y+h),color_dict[label],2)
    cv2.rectangle(image,(x,y-40),(x+w,y),color_dict[label],-1)
    cv2.putText(image,label_dict[label],(x,y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8,(255,255,255),2)

  elapsed_time = time.time() - time_now
  fps = frame_id / elapsed_time
  cv2.putText(image,"FPS: " + str(round(fps,2)),(10,20),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),2)

  cv2.imshow('Live Webcam',image)
  key = cv2.waitKey(1)

  if (key == 27):
    break

cv2.destroyAllWindows()
cap.release()
