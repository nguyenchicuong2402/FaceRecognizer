import pickle
import cv2
import os


CONFIDENCE = 0.5
EMBEDDING_MODEL_PATH = os.path.dirname(__file__) + r'\data_model\openface\openface_nn4.small2.v1.t7'
EMBEDDING_PATH = os.path.dirname(__file__) + r'\data_model\EMBEDDINGS.pickle'
EMBEDDINGS_DEFAULT_PATH = os.path.dirname(__file__) + r'\data_model\DEFAULT_EMBEDDINGS.pickle'
RECOGNIZER_PATH = os.path.dirname(__file__) + r'\data_model\RECOGNIZER.pickle'
LABEL_ENCODER_PATH = os.path.dirname(__file__) + r'\data_model\LABEL_ENCODER.pickle'
PROTO_PATH = os.path.dirname(__file__) + r'\data_model\face_detection_model\deploy.prototxt'
MODEL_PATH = os.path.dirname(__file__) + r'\data_model\face_detection_model\res10_300x300_ssd_iter_140000.caffemodel'
FACE_CASCADE_PATH = os.path.dirname(__file__) + r'\data_model\face_detection_model_opencv\haarcascades\haarcascade_frontalface_default.xml'
ID_PERSON_PATH = os.path.dirname(__file__) + r'\data_model\ID_PERSON.txt'

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
def save_embedding(encode):
    embeddings = []
    ids = []

    # đọc face encode đã có 
    data = load_embedding()
    for item in data['embeddings']:
        embeddings.append(item)
    for id in data['ids']:
        ids.append(id)

    # thêm encode mới
    embeddings.append(encode['embedding'])
    ids.append(encode['id'])

    data = {"embeddings": embeddings, "ids": ids}

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
        # load file embedding default
        else:
            embedddings = pickle.loads(open(EMBEDDINGS_DEFAULT_PATH, "rb").read())
            return embedddings
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


# load file face cascade
def load_face_cascade():
    if os.path.exists(FACE_CASCADE_PATH) == True:
        try:
            return cv2.CascadeClassifier(FACE_CASCADE_PATH)
        except:
            print('[ERROR] Lỗi load file face cascade')
    else:
        print('[ERROR] Không tìm thấy file face cascade ')


# load id những người đã thêm
def select_person():
    try:
      ,l  file = open(ID_PERSON_PATH, "r")
    except:
        print('[ERROR] Không thể mở file ID Person')
    else:
        return file.read()
    finally:
        file.close()


