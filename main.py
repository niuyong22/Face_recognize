import data_record
import recognizer

if __name__ == '__main__':
    print('*********************人脸识别**********************')
    while True:
        option = int(input('输入选择(1.录入信息 2.查看已有信息名单 3.身份识别 4.清空录入数据 5.退出): '))
        if option == 1:
            sub_option = int(input('输入信息录入方式(0.从摄像头录取 1.从视频中录取): '))
            data_record.record_train_data(sub_option)
        elif option == 2:
            data_record.show_user_list()
        elif option == 3:
            sub_option = int(input('选择人脸识别算法(0.pca特征脸 1.LBPHFaceRecognizer 2.EigenFaceRecognizer): '))
            recognizer.predict(sub_option)
        elif option == 4:
            data_record.clear_data()
        elif option == 5:
            break
