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


# xử lý dữ liệu trước khi training
def extract_embeddings(dataPath):
    # kiểm tra đường dẫn
    if os.path.exists(dataPath) == False:
        print('[ERROR] Không tìm thấy thư mục')
        return

    # chứa dữ liệu những khuôn mặt và tên của người
    knownEmbeddings = []
    knownNames = []
    # tổng số khuôn mặt nhận biết được
    total = 0

    # lấy tất cả đường dẫn hình ảnh trong thư mục
    imagePaths = list(paths.list_images(dataPath))
    for (i, imagePath) in enumerate(imagePaths):
        print("(+) Đang xử lý hình ảnh {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load ảnh 
        try:
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
        except:
            print('[ERROR] Lỗi tải hình ảnh')
            continue

        # construct a blob from the image
        imageBlob = blod_image(image)

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
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI and grab the ROI dimensions
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = blod_face_image(face)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # thêm tên và embedding face
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # lưu lại dữ liệu đã xử lý
    print("(*) serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "ids": knownNames}
    save_embedding(data)


# training dữ liệu đã chọn lọc
def known_training(dataPath):
    # tiền xử lý dữ liệu trước khi training
    extract_embeddings(dataPath)

    # load the face embeddings
    data = load_embedding()

    # lưu id khuôn mặt
    label_encoder = LabelEncoder()
    ids = label_encoder.fit_transform(data["ids"])

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
def unknown_training(dataPath):
    # kiểm tra thư mục có tồn tại không
    if os.path.exists(dataPath) == False:
        print('[ERROR] Thư mục không tồn tại')
        return False

    # lấy tất cả đường dẫn hình ảnh trong thư mục
    imagePaths = list(paths.list_images(dataPath))
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
    training(dataPath)


# nhận diện gương mặt
def face_recognizer(image):
    # load the actual face recognition model along with the label encoder
    recognizer = load_recognizer()
    label_encoder = load_label_encoder()

    #resize image
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    imageBlob = blod_image(image)

    # apply OpenCV's deep learning-based face detector to localize faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > CONFIDENCE:
            # compute the (x, y)-coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = blod_face_image(face)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            info = np.argmax(preds)
            percent = preds[info] * 100
            id = label_encoder.classes_[info]

            # draw the bounding box of the face along with the associated probability
            text = "{} [{:.2f}%]".format(id, percent)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX - 25, endY + 25),
                cv2.FONT_HERSHEY_COMPLEX , 0.75, (0, 0, 255), 1)

    return image, id, percent


# detect khuôn mặt
def face_detect(image):
    result = False
    # kiểm tra ảnh có null không
    if image is not None:
        # chuyển ảnh màu thành ảnh xám
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # quét các khuôn mặt trong ảnh
        faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

        # với mỗi khuôn mặt vẽ 1 hình vuông để nhận biết
        for (x, y, w, h) in faces:
            # image, toạ độ x-y, chiều rộng + chiều cao khung, màu, độ dày
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if len(faces) > 0:
            print('[INFO] Tìm thấy {0} khuôn mặt'.format(len(faces)))
            result = True

    # trả về hình ảnh
    return result, image

