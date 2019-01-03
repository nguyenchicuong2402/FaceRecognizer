import numpy as np
import imutils
import os
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from model import *

MIN_PERCENT_FACE_SAME = 70

# load file
print('[INFO] Đang tải dữ liệu detector')
detector = load_detector()
print('[INFO] Đang tải dữ liệu embedder')
embedder = load_embedder()
print('[INFO] Đang tải dữ liệu face cascade')
face_cascade = load_face_cascade()


# tạo vector face
def face_ROI(image):
    # lấy kích thước ảnh
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize 
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # ensure that the detection with the largest probability also
        # means our minimum probability test (thus helping filter out
        # weak detections)
        if confidence > CONFIDENCE:
            # tạo khung chứa khuôn mặt nhận diện được
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                return

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = face_blod = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            
            return vec, box
        else:
            return None, None
    else:
        return None, None

# mã hoá khuôn mặt từ hình ảnh
def face_encoding(id, image):
    # thêm tên và embedding face
    vec, box = face_ROI(image)
    if vec is not None:
        embedding = vec.flatten()

        # lưu lại dữ liệu đã xử lý
        data = {"embedding": embedding, "id": id}
        save_embedding(data)


# xử lý dữ liệu trước khi training
def extract_embeddings(dataTraining):
    # kiểm tra đường dẫn
    if os.path.exists(dataTraining) == False:
        print('[ERROR] Không tìm thấy thư mục')
        return

    # tổng số khuôn mặt nhận biết được
    total = 0

    # lấy tất cả đường dẫn hình ảnh trong thư mục
    imagePaths = list(paths.list_images(dataTraining))
    for (i, imagePath) in enumerate(imagePaths):
        print("(+) Đang xử lý hình ảnh {}/{}".format(i + 1, len(imagePaths)))
        id = imagePath.split(os.path.sep)[-2]

        # load ảnh 
        try:
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
        except:
            print('[ERROR] Lỗi tải hình ảnh')
            continue
        else:
            face_encoding(id, image)
            total += 1

    print('[INFO] Đã encode {} khuôn mặt trên tổng số {}'.format(total, len(imagePaths)))


# training dữ liệu đã chọn lọc
def known_training(dataTraining=None):
    # tiền xử lý dữ liệu trước khi training
    if dataTraining is not None:
        extract_embeddings(dataTraining)

    # load the face embeddings
    data = load_embedding()

    # lưu id khuôn mặt
    label_encoder = LabelEncoder()
    ids = label_encoder.fit_transform(data["ids"])
    print(ids)

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("[INFO] Bắt đầu training ...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], ids)

    # write the actual face recognition model to disk
    save_recognizer(recognizer)

    # write the label encoder to disk
    save_label_encoder(label_encoder)

    print("[INFO] Đã hoàn thành training")


# training dữ liệu không chọn lọc
def unknown_training(dataTraining):
    # kiểm tra thư mục có tồn tại không
    if os.path.exists(dataTraining) == False:
        print('[ERROR] Thư mục không tồn tại')
        return False

    # lấy tất cả đường dẫn hình ảnh trong thư mục
    imagePaths = list(paths.list_images(dataTraining))
    for (i, imagePath) in enumerate(imagePaths):
        try:
            img = cv2.imread(imagePath)
            image, id, percent = face_recognizer(img)

            # nếu nhỏ hơn thì xoá hình
            if percent < MIN_PERCENT_FACE_SAME:
                if os.path.exists(imagePath):
                    os.remove(imagePath)
        except:
            print('[ERROR] Lỗi xử lý hình trước khi training')

    # bắt đầu training
    training(dataTraining)


# nhận diện gương mặt
def face_recognizer(image):
    # load the actual face recognition model along with the label encoder
    recognizer = load_recognizer()
    label_encoder = load_label_encoder()

    #resize image
    try:
        image = imutils.resize(image, width=600)
    except:
        print('[ERROR] Lỗi file hình ảnh')
    else:
        vec, box = face_ROI(image)
        (startX, startY, endX, endY) = box.astype("int")

        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        info = np.argmax(preds)
        percent = preds[info] * 100
        id = label_encoder.classes_[info]

        # draw the bounding box of the face along with the associated probability
        text = "{} [{:.2f}%]".format(id, percent)
        cv2.rectangle(image, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(image, text, (startX - 25, endY + 25),
            cv2.FONT_HERSHEY_COMPLEX , 0.75, (0, 0, 255), 1)

    return image, id, percent


# detect khuôn mặt
def face_detect(image, show_rec=None):
    # kiểm tra ảnh có null không
    if image is not None:
        # chuyển ảnh màu thành ảnh xám
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # quét các khuôn mặt trong ảnh
        faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

        if show_rec is not None:
            # với mỗi khuôn mặt vẽ 1 hình vuông để nhận biết
            for (x, y, w, h) in faces:
                # image, toạ độ x-y, chiều rộng + chiều cao khung, màu, độ dày
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # trả về hình ảnh
    return len(faces), image

