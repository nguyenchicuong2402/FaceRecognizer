import pickle
import cv2
import os


CONFIDENCE = 0.5
EMBEDDING_MODEL_PATH = os.path.dirname(__file__) + '\\data_model\\openface\\openface_nn4.small2.v1.t7'
EMBEDDING_PATH = os.path.dirname(__file__) + '\\data_model\\EMBEDDINGS.pickle'
RECOGNIZER_PATH = os.path.dirname(__file__) + '\\data_model\\RECOGNIZER.pickle'
LABEL_ENCODER_PATH = os.path.dirname(__file__) + '\\data_model\\LABEL_ENCODER.pickle'
PROTO_PATH = os.path.dirname(__file__) + '\\data_model\\face_detection_model\\deploy.prototxt'
MODEL_PATH = os.path.dirname(__file__) + '\\data_model\\face_detection_model\\res10_300x300_ssd_iter_140000.caffemodel'
FACE_CASCADE_PATH = os.path.dirname(__file__) + '\\data_model\\face_detection_model_opencv\\haarcascades\\haarcascade_frontalface_default.xml'


# xoá dữ liệu training
def delete_training_data():
    if os.path.exists(EMBEDDING_PATH):
        os.remove(EMBEDDING_PATH)
        print('[INFO] Đã xoá file EMBEDDING')

    if os.path.exists(RECOGNIZER_PATH):
        os.remove(RECOGNIZER_PATH)
        print('[INFO] Đã xoá file RECOGNIZER')

    if os.path.exists(LABEL_ENCODER_PATH):
        os.remove(LABEL_ENCODER_PATH)
        print('[INFO] Đã xoá file LABEL ENCODER')


# load detector
def load_detector():
    try:
        net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
        return net
    except:
        print('[ERROR] Lỗi load file Net Caffe')


# load embedder
def load_embedder():
    try:
        net = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL_PATH)
        return net
    except:
        print('[ERROR] Lỗi load file Net Torch')


# save embedding
def save_embedding(data):
    # nếu đã có file thì tiến hành chèn vào cuối file
    if os.path.isfile(EMBEDDING_PATH):
        embeddings = []
        ids = []

        dt = load_embedding()
        for item in dt['embeddings']:
            embeddings.append(item)
        for id in dt['ids']:
            ids.append(id)


        for item in data['embeddings']:
            embeddings.append(item)
        for id in data['ids']:
            ids.append(id)

        data = {"embeddings": embeddings, "ids": ids}

        for d in data['embeddings']:
            print(d)
        
    try:
        f = open(EMBEDDING_PATH, "wb")
        f.write(pickle.dumps(data))
    except:
        print('[ERROR] Lổi ghi file Embedding')
    finally:
        f.close()


# save label encoder
def save_label_encoder(label_encoder):
    try:
        f = open(LABEL_ENCODER_PATH, "wb")
        f.write(pickle.dumps(label_encoder))
    except:
        print('[ERROR] Lỗi ghi file Label Encoder')
    finally:
        f.close()


def save_recognizer(recognizer):
    try:
        f = open(RECOGNIZER_PATH, "wb")
        f.write(pickle.dumps(recognizer))
    except:
        print('[ERROR] Lỗi ghi file Recognizer')
    finally:
        f.close()

# load embedding
def load_embedding():
    try:
        if os.path.isfile(EMBEDDING_PATH):
            embedddings = pickle.loads(open(EMBEDDING_PATH, "rb").read())
            return embedddings
        else:
            print('[ERROR] Vui lòng Training trước')
    except:
        print('[ERROR] Lỗi load file Embedding')
        

# load recognizer
def load_recognizer():
    try:
        if os.path.isfile(RECOGNIZER_PATH):
            recognizer = pickle.loads(open(RECOGNIZER_PATH, "rb").read())
            return recognizer
        else:
            print('[ERROR] Vui lòng training trước')
    except:
        print('[ERROR] Lỗi load file recognizer')


# load label encoder
def load_label_encoder():
    try:
        if os.path.isfile(LABEL_ENCODER_PATH):
            label_encoder = pickle.loads(open(LABEL_ENCODER_PATH, "rb").read())
            return label_encoder
        else:
            print('[ERROR] Vui lòng training dữ liệu trước')
    except:
        print('[ERROR] Lỗi load file label encoder')


# blod image
def blod_image(image):
    try:
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        return imageBlob
    except:
        print('[ERROR] Lỗi Blod Image')


# blod face image
def blod_face_image(face):
    try:
        face_blod = cv2.dnn.blobFromImage(face, 1.0 / 255,
            (96, 96), (0, 0, 0), swapRB=True, crop=False)
        return face_blod
    except:
        print('[ERROR] Lỗi Blod face image')


# load file face cascade
def load_face_cascade():
    if os.path.exists(FACE_CASCADE_PATH) == True:
        try:
            return cv2.CascadeClassifier(FACE_CASCADE_PATH)
        except:
            print('[ERROR] Lỗi load file face cascade')
    else:
        print('[ERROR] Không tìm thấy file face cascade ')