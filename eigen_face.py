import os

import cv2 as cv
import numpy as np

face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
recognizer = cv.face.EigenFaceRecognizer_create()


def record_train_data():
    capture = cv.VideoCapture(0)
    count = 0
    face_id = input('输入保存数据的名称')
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(frame, 1.3, 5)
        if len(faces) == 1:
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0))
                count += 1
                cv.imwrite("data/User-" + str(face_id) + '-' + str(count) + '.jpg', gray[y:y + h, x:x + w])
        # 显示图片
        cv.imshow('image', frame)
        k = cv.waitKey(1)
        if k == 27:
            break
        elif count >= 800:
            break
    # 关闭摄像头，释放资源
    capture.release()
    cv.destroyAllWindows()


def create_dataset():
    imgs = []
    ids = []
    root = './data'
    names = os.listdir(root)
    for name in names:
        suffix = name.split('.')[-1]
        info = name.split('.')[0]
        user_id = int(info.split('-')[1])
        img = cv.imread(f'{root}/{name}', 0)
        img = cv.resize(img, (128, 128))
        ids.append(user_id)
        imgs.append(img)
    return imgs, ids


def train():
    faces, ids = create_dataset()
    print('training data......')
    recognizer.train(faces, np.array(ids))


def predict(threshold=5000):
    train()
    capture = cv.VideoCapture(0)
    # capture = cv.VideoCapture('./test_demo.mp4')
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + w), (255, 0, 0))
            user_id, score = recognizer.predict(cv.resize(gray[y:y + h, x:x + w], (128, 128)))
            print(f'user{user_id} --- score: {score}')
            if score < threshold:
                cv.putText(frame, f'{user_id}', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            else:
                cv.putText(frame, 'unkonw', (x + 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        # 显示图片
        cv.imshow('image', frame)
        k = cv.waitKey(24)
        if k == 27:
            break
    # 关闭摄像头，释放资源
    capture.release()
    cv.destroyAllWindows()