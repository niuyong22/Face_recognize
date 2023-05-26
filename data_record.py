import os
import shutil

import cv2 as cv
import numpy as np

face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
CAMERA = 0
VIDEO = 1


def img_resize(image, width_new, height_new):
    height, width = image.shape[0], image.shape[1]
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv.resize(image, (int(width * height_new / height), height_new))
    return img_new


def record_train_data(option):
    if option == CAMERA:
        capture = cv.VideoCapture(0)
    elif option == VIDEO:
        print('ur recording data from an video...')
        file_name = input('please input the location of video: ')
        capture = cv.VideoCapture(file_name)

    face_id = input('输入保存数据的名称:')
    user_dict = {}
    user_exist = False
    # 遍历user.txt文件，查找是否存在用户信息,并构建用户信息字典
    with open('./user.txt') as f:
        for line in f:
            user_id = line.split(':')[0]
            if user_id == int(face_id):
                user_exist = True
            user_dict[line.split(':')[0]] = line.split(':')[1]
    count = 0
    if user_exist:
        count = user_dict[face_id]
    num2save = 800
    target = count + num2save
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        if option == VIDEO:
            frame = img_resize(frame, 640, 480)

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
        elif count >= target:
            break
    # 关闭摄像头，释放资源
    capture.release()
    cv.destroyAllWindows()

    user_dict[face_id] = count
    with open('./user.txt', 'w') as f:
        for item in user_dict.items():
            f.write(f'{item[0]}:{item[1]}\n')


def show_user_list():
    print('已录入信息名单:')
    print('*-' * 30)
    with open('./user.txt') as f:
        for line in f:
            user_id = line.split(':')[0]
            num = line.split(':')[1][:-1]
            print(f'用户{user_id}--->已录入 {num} 张')


def clear_data():
    with open('./user.txt', 'w') as f:
        pass
    shutil.rmtree('./data')
    os.mkdir('./data')
    print('已清空录入数据!')
