import cv2
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

model = load_model('model/model_last.h5')
img_width, img_height = 200, 200

# Load the cascade face classifier
face_cascade = cv2.CascadeClassifier(r"C:\Users\Acer\anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
# cap = cv2.VideoCapture("face-mask-12k-images-dataset\4216631-uhd_3840_2160_30fps.mp4")
cap = cv2.VideoCapture(0)
img_count_full = 0

# Parameters for text
font = cv2.FONT_HERSHEY_SIMPLEX
Class_label = ' '
fontscale = 0.5
color = (255, 0, 0)
thickness = 1

while cap.isOpened():
    img_count_full += 1
    ret, frame = cap.read()

    if ret:

        # scale = 50
        # width = int(frame.shape[1] * scale / 100)
        # height = int(frame.shape[0] * scale / 100)
        # dim = (width, height)
        # frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, 1.1, 6)

        img_count = 0
        for (x, y, w, h) in faces:
            org = (x - 10, y - 10)
            img_count += 1
            color_face = frame[y:y+h, x:x+w]
            
            face_dir = 'face/input'
            with_mask_dir = 'face/with_mask'
            without_mask_dir = 'face/without_mask'
            
            os.makedirs(face_dir, exist_ok=True)
            os.makedirs(with_mask_dir, exist_ok=True)
            os.makedirs(without_mask_dir, exist_ok=True)

            face_path = os.path.join(face_dir, f'{img_count_full}{img_count}face.jpg')
            cv2.imwrite(face_path, color_face)

            img = load_img(face_path, target_size=(img_width, img_height))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)
            pred_prob = model.predict(img)
            pred = np.argmax(pred_prob)

            if pred == 0:
                print(f"user with mask - predict = {pred_prob[0][0]}")
                Class_label = "Mask"
                color = (255, 0, 0)
                # mask_path = os.path.join(with_mask_dir, f'{img_count_full}{img_count}face.jpg')
                # cv2.imwrite(mask_path, color_face)
            else:
                print(f"user not wearing mask - predict = {pred_prob[0][1]}")
                Class_label = "No Mask"
                color = (0, 255, 0)
                # no_mask_path = os.path.join(without_mask_dir, f'{img_count_full}{img_count}face.jpg')
                # cv2.imwrite(no_mask_path, color_face)
            if os.path.exists(face_path):
                os.remove(face_path)
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, Class_label, org, font, fontscale, color, thickness, cv2.LINE_AA)

        cv2.imshow("opencv", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
