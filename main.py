from faces import *
from model import *
from download_image import *
import os
import keyboard
import numpy as np
import cv2


CAMERA_PATH = os.path.dirname(__file__) + '\\images\\camera'

# training từ thư mục
def item_training_from_folder():
    print("""
            ***********************************************
            *        TRAINING TỪ THƯ MỤC HÌNH ẢNH         *
            ***********************************************
            ------------by nguyenchicuong2402--------------
    """)

    path = str(input('>> Nhập đường dẫn thư mục hình ảnh: '))
    
    if os.path.exists(path) and os.path.isdir(path):
        known_training(path)
    else:
        print('[ERROR] Đường dẫn không tồn tại')

    print('[INFO] Nhấn ESC để quay lại')
    while True:
        if keyboard.is_pressed('esc'):
            break


# training từ camera camera
def item_training_from_camera():
    print("""
            ***********************************************
            *             TRAINING TỪ CAMERA              *
            ***********************************************
            ------------by nguyenchicuong2402--------------
    """)

    # nhập thông tin người
    id = str(input('>> Nhập ID: '))

    cap = cv2.VideoCapture(0)

    if cap.isOpened() == False:
        print('[ERROR] Không thể mở camera')
        return
    else:
        print('[INFO] Camera đang hoạt động')
        print('[INFO] Nhấn ESC để dừng camera')

        i = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if ret == True:
                length, image = face_detect(frame, show_rec=True)
                cv2.imshow('Training', image)

                if length == 1:
                    face_encoding(id, image)
                    print('[INFO] Đã training {}/50'.format(i+1))
                    i += 1
                    if i == 50: break
            else:
                break;

            if cv2.waitKey(25) & 0xFF == 27:
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()
        print('[INFO] Đã đóng camera')

        # bắt đầu training
        if i != 0:
            known_training()


# học củng cố
def item_training_from_google():
    ids = select_person().split('\n')

    for id in ids:
        download(id, 10, r'download')

    known_training('download')



# nhận diện khuôn mặt từ camera
def item_recognizer_from_camera():
    print("""
            ***********************************************
            *        NHẬN DIỆN KHUÔN MẶT TỪ CAMERA        *
            ***********************************************
            ------------by nguyenchicuong2402--------------
    """)
    cap = cv2.VideoCapture(0)

    if cap.isOpened() == False:
        print('[ERROR] Không thể mở camera')
    else:
        print('[INFO] Camera đang hoạt động')
        print('[INFO] Nhấn ESC để dừng camera')

        i = 1
        while cap.isOpened():
            ret, frame = cap.read()

            if ret == True:
                length, img = face_detect(frame)
                image, id, percent = face_recognizer(img)
                cv2.imshow('Nhan dien guong mat', image)
            else:
                break;

            if cv2.waitKey(25) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


# nhận diện khuôn mặt từ hình ảnh
def item_recognizer_from_image():
    print("""
            ***********************************************
            *       NHẬN DIỆN KHUÔN MẶT TỪ HÌNH ẢNH       *
            ***********************************************
            ------------by nguyenchicuong2402--------------
    [INFO] Nhập đường dẫn rỗng hoặc sai để thoát
    """)

    while True:
        imgPath = input('>> Nhập đường dẫn hình ảnh: ')
        if os.path.isfile(imgPath):
            img = cv2.imread(imgPath)
            if img is not None:
                print('[INFO] Đã mở hình ảnh')

                image, id, percent = face_recognizer(img)
                cv2.imshow('Nhan dien khuon mat', image)

                cv2.waitKey(0)
            else:
                print('[ERROR] Không thể mở hình ảnh')
        else:
            print('[ERROR] Không tìm thấy hình ảnh')
            break
        
    cv2.destroyAllWindows()


# menu training
def menu_training():
    print("""
            ***********************************************
            *               TRAINING DỮ LIỆU              *
            ***********************************************
            ------------by nguyenchicuong2402--------------

                1. Training qua thư mục hình ảnh
                2. Training qua camera
                3. Training qua tìm kiếm Google (học thêm)
                0. Thoát
    """)

    choose = 0
    while (True):
        try:
            choose = int(input('>> Chọn: '))
        except ValueError:
            print('[ERROR] Vui lòng nhập số')
        else:
            if choose in range(0, 4):
                break
            else:
                print('[ERROR] Vui lòng nhập số từ 0 - 3')

    if choose == 1:
        item_training_from_folder()
    elif choose == 2:
        item_training_from_camera()
    elif choose == 3:
        item_training_from_google()


# menu nhận diện
def menu_recognizer():
    print("""
            ***********************************************
            *             NHẬN DIỆN KHUÔN MẶT             *
            ***********************************************
            ------------by nguyenchicuong2402--------------

                1. Nhận diện từ camera
                2. Nhận diện từ hình ảnh
                0. Thoát
    """)

    choose = 0
    while (True):
        try:
            choose = int(input('>> Chọn: '))
        except ValueError:
            print('[ERROR] Vui lòng nhập số')
        else:
            if choose in range(0, 3):
                break
            else:
                print('[ERROR] Vui lòng nhập số từ 0 - 2')

    if choose == 1:
        item_recognizer_from_camera()
    elif choose == 2:
        item_recognizer_from_image()


# menu xoá tất cả dữ liệu đã training
def menu_reset_data():
    print("""
            ***********************************************
            *             XOÁ TOÀN BỘ DỮ LIỆU             *
            ***********************************************
            ------------by nguyenchicuong2402--------------

    [WARNING] Thao tác này sẽ xoá tất cả dữ liệu trước đó
    """)

    choose = 0
    while (True):
        choose = input('>> Bạn có muốn tiếp tục không? (Y/N): ')
        if choose.lower().strip() == 'y':
            print('[INFO] Đang bắt đầu quá trình xoá dữ liệu')
            delete_training_data()
            print('[INFO] Đã xoá dữ liệu')
            break
        elif choose.lower().strip() == 'n':
            print('[INFO] Huỷ giao tác xoá dữ liệu')
            break
    

# menu chính
def menu():
    print("""
            ***********************************************
            *             CHƯƠNG TRÌNH CHÍNH              *
            ***********************************************
            ------------by nguyenchicuong2402--------------
            
                1. Nhận dạng khuôn mặt
                2. Training dữ liệu
                3. Reset data
                0. Thoát
    """)

    while (True):
        try:
            choose = int(input('>> Chọn: '))
        except ValueError:
            print('[ERROR] Vui lòng nhập số')
        else:
            if choose in range(0, 4):
                return choose
            else:
                print('[ERROR] Vui lòng nhập số từ 0 - 3')


if __name__ == '__main__':
    while True:
        choose = menu()
        if choose == 1:
            menu_recognizer()
        elif choose == 2:
            menu_training()
        elif choose == 3:
            menu_reset_data()
        elif choose == 0:
            break

    print('Tạm biệt !!!')
