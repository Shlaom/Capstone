from django.http import HttpResponse
from django.shortcuts import render

from django.core.mail import EmailMessage
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading


import numpy as np
import cv2
import copy
from sklearn.preprocessing import LabelEncoder
from skimage.transform import resize
from keras.models import load_model

model_path = './model/keras/facenet_keras.h5'
model = load_model(model_path)

face_detector = "./face_detector/"
prototxt = face_detector + "deploy.prototxt"  # prototxt 파일 : 모델의 레이어 구성 및 속성 정의
weights = face_detector + "res10_300x300_ssd_iter_140000.caffemodel"  # caffemodel 파일 : 얼굴 인식을 위해 ResNet 기본 네트워크를 사용하는 SSD(Single Shot Detector) 프레임워크를 통해 사전 훈련된 모델 가중치 사용
net = cv2.dnn.readNet(weights, prototxt)  # cv2.dnn.readNet() : 네트워크를 메모리에 로드

minimum_confidence = 0.5
minimum_pixel_size = 10

"""def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def calc_embs(imgs, margin, batch_size):
    aligned_images = prewhiten(imgs)
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs"""

class FaceDemo(object):
    def __init__(self):
        #self.vc = None
        self.margin = 10
        self.batch_size = 1
        self.n_img_per_person = 30
        self.is_interrupted = False
        self.data = {}
        self.le = None
        #self.clf = None
        self.mean_embs = [[ 4.89895039e-02, -1.01075421e-01,  2.05164452e-02, -4.99203576e-02,
        5.16341986e-02, -3.60137719e-02,  4.90995832e-03, -7.56084861e-02,
        2.34752144e-02, -5.98150707e-02, -6.70285527e-02, -1.23107243e-01,
        8.30463817e-02, -1.06744865e-01,  4.58737050e-02, -8.11372410e-02,
        1.41562238e-01, -1.32006500e-02,  1.10179872e-01, -5.22426756e-03,
        2.13514914e-05,  2.46688524e-02, -1.21527021e-01,  7.60444404e-02,
        6.95898862e-02,  2.77209641e-02,  3.52082941e-02,  2.44756752e-02,
       -3.23506074e-02,  1.99356892e-02,  8.47499638e-02,  4.01525296e-02,
       -2.49115705e-03,  2.04849738e-02,  8.86129459e-02, -1.25052858e-01,
        4.47382565e-02,  2.95593814e-02,  8.81125537e-02, -1.00242588e-02,
        8.33695071e-02,  3.18321566e-02,  9.91354677e-03, -5.09051414e-02,
       -9.49022743e-02, -2.99594314e-02,  2.88249618e-02,  1.17430658e-02,
        3.47948434e-02, -1.91045142e-02,  2.47923569e-03,  6.84951073e-02,
        2.32476879e-02, -1.06318023e-01,  6.10156593e-02, -1.64315971e-03,
        9.96302097e-02, -3.75809053e-02, -4.05098489e-03, -1.26287901e-01,
        6.98642068e-02, -4.87399016e-02,  8.36635733e-03,  2.27696890e-01,
       -3.98493963e-02,  9.79243418e-02,  7.34886790e-02, -8.64522127e-02,
        4.99262427e-02,  1.21599598e-02,  5.80044780e-02, -2.26042522e-01,
        9.20709190e-02, -1.44840984e-01, -1.07062715e-01, -2.52242246e-02,
       -2.24676639e-02,  1.27627811e-02, -3.71675125e-02, -1.01654110e-01,
        9.18653353e-03, -1.79133223e-02,  5.46393355e-02,  1.47592905e-01,
        9.09115993e-02, -3.75343281e-02,  1.66302807e-01, -1.24852591e-01,
       -2.10050761e-01,  9.52958886e-02,  2.57851383e-02,  1.28961399e-02,
       -1.52750740e-01, -2.01814691e-02,  6.60260954e-02, -4.60247958e-02,
        1.22310736e-01, -4.61433949e-02, -1.30950800e-01,  1.81281805e-02,
       -5.88858425e-02,  1.64887221e-01,  6.19369931e-02,  5.24384887e-02,
       -1.99465956e-02,  1.29903279e-01, -4.88755843e-02, -1.74328422e-02,
       -6.84097267e-02, -1.47166737e-01, -1.15578962e-01, -2.38846980e-02,
        6.69558256e-02,  1.43282795e-02, -4.57717128e-02, -4.54649701e-02,
       -8.04420250e-02,  3.60894958e-02,  8.25166078e-02, -4.47277407e-02,
        2.16121024e-02,  1.19052460e-02,  1.94384290e-02,  1.98354689e-02,
        7.79475539e-02, -1.32114381e-01,  2.68692436e-02, -8.29275343e-03]]
        #self.mean_embs = []
        self.mosaic_margin = 30
        self.W = None
        self.H = None
        #self.num_registered_faces = 0
        self.threshold = 0.8

        self.video = cv2.VideoCapture('rtmp://192.168.207.119/live/demo')
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def _signal_handler(self, signal, frame):
        self.is_interrupted = True

    def prewhiten(self, x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size
        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size
        else:
            raise ValueError('Dimension should be 3 or 4')

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        y = (x - mean) / std_adj
        return y

    def l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    def calc_embs(self, imgs, margin, batch_size):
        aligned_images = self.prewhiten(imgs)
        pd = []
        for start in range(0, len(aligned_images), batch_size):
            pd.append(model.predict_on_batch(aligned_images[start:start + batch_size]))
        embs = self.l2_normalize(np.concatenate(pd))

        return embs

    def capture_images(self, name='Unknown'):
        vc = cv2.VideoCapture(0)
        #vc = cv2.VideoCapture('./videos/sample5.mp4')
        #self.vc = vc
        '''if vc.isOpened():
            is_capturing, _ = vc.read()
        else:
            is_capturing = False'''

        imgs = []   #학습에 사용될 얼굴 이미지 배열
        #signal.signal(signal.SIGINT, self._signal_handler)
        self.is_interrupted = False
        print("Capturing...")
        count = 0
        while True:
            is_capturing, frame = vc.read()
            #frame = imutils.resize(frame, width= 640, height= 480)
            #rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #facenet에 들어갈 때 RGB로 변환 필요. frame 전체 말고 faces나 imgs를 RGB로 바꾸면 연산 감소할 듯. 다시 생각해보니 그냥 BGR로 넣어도 상관없지 않나?
            rgb_frame = frame   #cvtColor()안하고 그냥 넣어도 잘 됨

            (H, W) = rgb_frame.shape[:2]    #프레임 크기 측정
            blob = cv2.dnn.blobFromImage(rgb_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)  # setInput() : blob 이미지를 네트워크의 입력으로 설정

            detections = net.forward()  # forward() : 네트워크 실행(얼굴 검출)
            #print(detections[0, 0, 0, :])

            for i in range(0, detections.shape[2]):
                # 얼굴 인식 확률 추출
                confidence = detections[0, 0, i, 2]

                # 얼굴 인식 확률이 최소 확률보다 큰 경우
                if confidence > minimum_confidence:
                    if count == 0:
                        input("정면(Press Enter key when you ready)")
                    elif count == self.n_img_per_person/5:
                        input("좌(Press Enter key when you ready)")
                    elif count == (self.n_img_per_person/5) * 2:
                        input("우(Press Enter key when you ready)")
                    elif count == (self.n_img_per_person/5) * 3:
                        input("상(Press Enter key when you ready)")
                    elif count == (self.n_img_per_person/5) * 4:
                        input("하(Press Enter key when you ready)")
                    # bounding box 위치 계산
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H]) #w,h안곱하면 소수로 나옴
                    (left, top, right, bottom) = box.astype("int")

                    (left, top) = (max(0, left), max(0, top)) #왼쪽 위가 맞음
                    (right, bottom) = (min(W - 1, right), min(H - 1, bottom)) #오른쪽 아래가 맞음

                    img = rgb_frame[top:bottom, left:right, :]  # cannot warp image with dimensions (0,0,3)식으로 어쩌고 하는 에러 해결
                    if img.shape == (0, 0, 3):
                        continue
                    #if (right-left) <= minimum_pixel_size | (bottom-top) <= minimum_pixel_size:
                    #    continue

                    img = resize(rgb_frame[top:bottom, left:right, :],
                                 (160, 160), mode='reflect')  # 학습용 이미지로 전처리
                    imgs.append(img)
                    cv2.rectangle(frame,
                                  (left, top),
                                  (right, bottom),
                                  (255, 0, 0), thickness=2)
                    count += 1
            cv2.imshow('capture', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:  # Esc 키를 누르면 종료
                break

            if len(imgs) == self.n_img_per_person:
                self.data[name] = np.array(imgs)
                break
        print("Completed!\n")
        vc.release()
        cv2.destroyAllWindows()

    def train(self):
        labels = []
        embs = []
        names = self.data.keys()
        print("Preparing training datas...")
        for name, imgs in self.data.items():
            embs_ = self.calc_embs(imgs, self.margin, self.batch_size)
            labels.extend([name] * len(embs_))
            embs.append(embs_)

        #embs = np.concatenate(embs)
        print("Completed!\n")
        print("Training...")
        le = LabelEncoder().fit(labels)
        y = le.transform(labels)
        print(y)
        ##################################################
        mean_embs = []
        for i in range(0, np.shape(embs)[0]):
            sum = [0] * 128
            for j in embs[i]:
                sum += j
            mean_embs.append(sum/np.shape(embs)[1])
        self.mean_embs = mean_embs
        ##################################################
        #clf = SVC(kernel='linear', probability=True).fit(embs, y)

        self.le = le
        #self.clf = clf
        print("Completed!\n")

    def _findEuclideanDistance(self, source_representation, test_representation):
        if type(source_representation) == list:
            source_representation = np.array(source_representation)

        if type(test_representation) == list:
            test_representation = np.array(test_representation)

        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def _do_mosaic(self, frame, unknown_imgs_locs):
        for i in unknown_imgs_locs:
            (left, top, right, bottom) = i

            mos_margin_left = max(left - self.mosaic_margin, 0)
            mos_margin_top = max(top - self.mosaic_margin, 0)
            mos_margin_right = min(right + self.mosaic_margin, self.W)
            mos_margin_bottom = min(bottom + self.mosaic_margin, self.H)

            face_img = frame[mos_margin_top:mos_margin_bottom, mos_margin_left:mos_margin_right]  # 탐지된 얼굴 이미지 crop
            face_img = cv2.resize(face_img, dsize=(0, 0), fx=0.04, fy=0.04)  # 축소
            face_img = cv2.resize(face_img, (mos_margin_right - mos_margin_left, mos_margin_bottom - mos_margin_top), interpolation=cv2.INTER_AREA)  # 확대
            frame[mos_margin_top:mos_margin_bottom, mos_margin_left:mos_margin_right] = face_img  # 탐지된 얼굴 영역 모자이크 처리
        return frame

    def _detect_faces(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)  # setInput() : blob 이미지를 네트워크의 입력으로 설정
        detections = net.forward()  # forward() : 네트워크 실행(얼굴 인식)

        detected_faces_locs = []

        for i in range(0, detections.shape[2]):
            # 얼굴 인식 확률 추출
            confidence = detections[0, 0, i, 2]

            # 얼굴 인식 확률이 최소 확률보다 큰 경우
            if confidence > minimum_confidence:
                # bounding box 위치 계산
                box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])  # w,h안곱하면 소수로 나옴
                (left, top, right, bottom) = box.astype("int")

                (left, top) = (max(0, left), max(0, top))  # 왼쪽 위가 맞음
                (right, bottom) = (min(self.W - 1, right), min(self.H - 1, bottom))  # 오른쪽 아래가 맞음

                '''if ((right == left) | (bottom == top)):
                    continue'''
                if (right - left < minimum_pixel_size) | (bottom - top < minimum_pixel_size):
                    continue

                detected_faces_locs.append((left, top, right, bottom))
        return detected_faces_locs

    def _recognize_faces(self, frame, detected_faces_locs):
        unknown_faces_locs = []

        for locs in detected_faces_locs:
            (left, top, right, bottom) = locs

            img = resize(frame[top:bottom, left:right, :], (160, 160), mode='reflect')  # 학습용 이미지로 전처리
            embs = self.calc_embs(img[np.newaxis], self.margin, 1)

            threshold = 0.8
            # pred = "Unknown"
            for i in range(0, np.shape(self.mean_embs)[0]):
                # dst = DeepFace.dst.findEuclideanDistance(self.mean_embs[i], embs)
                dst = self._findEuclideanDistance(self.mean_embs[i], embs)
                if dst <= threshold:
                    break
                if (i == (np.shape(self.mean_embs)[0] - 1)):
                    unknown_faces_locs.append((left, top, right, bottom))
        return unknown_faces_locs

    def _optimized_recognize_faces(self, frame, detected_faces_locs):
        cpy_mean_embs = copy.deepcopy(self.mean_embs)

        for locs in detected_faces_locs:
            (left, top, right, bottom) = locs

            img = resize(frame[top:bottom, left:right, :], (160, 160), mode='reflect')  # 학습용 이미지로 전처리
            embs = self.calc_embs(img[np.newaxis], self.margin, 1)

            for mean_embs in cpy_mean_embs: #등록된 얼굴 수만큼 반복
                print(mean_embs)
                dst = self._findEuclideanDistance(mean_embs, embs)
                if dst <= self.threshold:    #동일인물이면 해당 얼굴을 검출된 얼굴 배열에서 제거, 등록된 얼굴도 배열에서 제거
                    detected_faces_locs.remove(locs)
                    cpy_mean_embs.remove(mean_embs)
            if len(cpy_mean_embs) == 0:
                break
        return detected_faces_locs

    def get_frame(self):
        frame = self.frame      #self.frame을 새 변수에 저장하지 않고 뒤에서 self.frame을 이용해서 코딩하면 성능 엄청 떨어짐. why?
        (self.H, self.W) = frame.shape[:2]

        detected_faces_locs = self._detect_faces(frame)
        unknown_faces_locs = self._optimized_recognize_faces(frame, detected_faces_locs)
        #unknown_faces_locs = self._recognize_faces(frame, detected_faces_locs)

        frame = self._do_mosaic(frame, unknown_faces_locs)

        frame_flip = cv2.flip(frame, 1)  # 좌우반전 flip
        _, jpeg = cv2.imencode('.jpg', frame_flip)  # jpeg:인코딩 된 이미지

        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

# 메인 페이지
def main(request):
    return render(request, 'main.html')

# 영상 송출 페이지
def home(request):
    return render(request, 'home.html')

@gzip.gzip_page
def video(request):
    try:
        # 응답 본문이 데이터를 계속 추가할 것이라고 브라우저에 알리고 브라우저에 원래 데이터를 데이터의 새 부분으로 교체하도록 요청
        # 즉, 서버에서 얻은 비디오가 jpeg 사진으로 변환되어 브라우저에 전달, 브라우저는 비디오 효과를 위해 이전 이미지를 새 이미지로 지속적 교체
        face_demo = FaceDemo()
        return StreamingHttpResponse(gen(face_demo), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def mypage(request):
    return render(request, 'navBar/myPage/myPage.html')