import io

from rest_framework import viewsets, permissions, generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view

from django.http.response import HttpResponse
from .serializer import ClassifySerializer, SimilaritySerializer

# ------------------------------------------------------------------------------------------------------------------------------
import keras.applications.nasnet
import numpy as np
import os
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Lambda, Input
from keras.applications.nasnet import NASNetLarge
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow as tf
from tqdm import tqdm

from numpy.linalg import norm
from numpy import dot

import cx_Oracle
import os

import requests
from PIL import Image
from io import BytesIO

# ------------------------------------------------------------------------------------------------------------------------------
root = os.getcwd()

img_size = (331, 331, 3)
dog_model = keras.models.load_model(
    os.path.join(root, 'breed_classification', 'models', 'dog_model.h5'))
cat_model = keras.models.load_model(
    os.path.join(root, 'breed_classification', 'models', 'cat_model.h5'))

dog_breeds = ['australian kelpie', 'cardigan welsh corgi', 'chesapeake bay retriever', 'chihuahua', 'chow chow',
              'clumber', 'cocker spaniel', 'curly-coated retriever', 'dalmatian', 'dandie dinmont', 'doberman pinscher',
              'english foxhound', 'english setter', 'english springer spaniel', 'entlebucher', 'eskimo dog',
              'flat-coated retriever', 'french bulldog', 'german shepherd', 'german short-haired pointer',
              'giant schnauzer', 'gordon setter dog', 'great dane', 'great pyrenees', 'greater swiss mountain dog',
              'groenendael', 'ibizan_hound', 'irish setter', 'irish terrier', 'irish water spaniel', 'irish wolfhound',
              'italian greyhound', 'jack russell terrier', 'japanese chin', 'keeshond', 'kerry blue terrier',
              'koeran jindo', 'komondor', 'kuvasz', 'labrador retriever', 'lakeland terrier', 'leonberg dog',
              'lhasa apso', 'malamute', 'malinois', 'maltese dog', 'mexican hairless', 'miniature pinscher',
              'miniature poodle', 'miniature schnauzer', 'newfoundland dog', 'norfolk terrier', 'norwegian elkhound',
              'norwich terrier', 'old english sheepdog', 'otterhound', 'papillon dog', 'pekinese dog',
              'pembroke welsh corgi', 'pomeranian', 'pug', 'redbone coonhound', 'rhodesian ridgeback', 'rottweiler',
              'saint bernard', 'saluki', 'samoyed', 'schipperke', 'scotch terrier', 'scottish deerhound',
              'sealyham terrier', 'shetland sheepdog', 'shiba', 'shih tzu', 'siberian husky', 'silky terrier',
              'standard poodle', 'standard schnauzer', 'sussex spaniel', 'tibetan mastiff', 'tibetan terrier',
              'toy poodle', 'vizsla', 'weimaraner', 'welsh springer spaniel', 'west highland white terrier', 'whippet',
              'wire-haired fox terrier', 'yorkshire terrier']

cat_breeds = ['abyssinian cat', 'american shorthair cat', 'balinese cat', 'bengal cat', 'birman cat', 'bombay cat',
              'british longhair cat', 'british shorthair cat', 'chartreux cat', 'devon rex cat', 'havana brown cat',
              'highland fold cat', 'japanese bobtail cat', 'maine coon cat', 'manx cat', 'munchkin cat', 'nebelung cat',
              'norway forest cat', 'persian cat', 'persian chinchilla cat', 'ragdoll cat', 'russian blue cat',
              'savannah cat', 'scottish fold cat', 'selkirk rex cat', 'siamese cat', 'siberian cat', 'singapura cat',
              'snowshoe cat', 'somali cat', 'sphynx cat', 'tonkinese cat', 'turkish angora cat']

# print(os.path.join(root, 'breed_classification', 'pics'))

# ------------------------------------------------------------------------------------------------------------------------------
LOCATION = r"C:\Users\joy14\PycharmProjects\AIServer\AI\instantclient_21_10"  # 나중에 상대경로로 지정
os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]  # 환경변수 등록, ec2에 올리려면 따로 지정해줘야할듯

# 연결
con = cx_Oracle.connect("scott", "tiger", "127.0.0.1:1521/xepdb1", encoding="UTF-8")
cursor = con.cursor()


def get_sql_result(sql):
    return cursor.execute(sql)


'''
a = get_sql_result("select * from user_tables")
for line in a:
    print(line[0])
'''

'''
# 사용 예시
sql = "select * from user_tables"
a = cursor.execute(sql)
for line in a:
    print(line)
con.close()
'''


# ------------------------------------------------------------------------------------------------------------------------------

def images_to_array(data_dir):
    os.chdir(data_dir)
    images_names = os.listdir(data_dir)
    test_size = len(images_names)
    # print(images_names)

    X = np.zeros([test_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    # for i in tqdm(range(test_size)):
    for i in tqdm(range(test_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name)
        img_pixels = tf.keras.preprocessing.image.load_img(img_dir, target_size=img_size)
        X[i] = img_pixels

    # print('Ouptut Data Size: ', X.shape)
    return X


'''

def str_to_imgarray(list):
    X = [img.resize((331, 331)) for img in list]
    X = [np.array(img) for img in X]
    # print(np.shape(X))
    return X

'''


def get_features(model_name, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)
    feature_maps = feature_extractor.predict(data, batch_size=64, verbose=1)

    # print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


def gen_test_features(pics_array):
    # 전처리기 똑같은거 통합할 것
    nasnet_preprocessor = keras.applications.nasnet.preprocess_input
    xception_preprocessor = keras.applications.inception_resnet_v2.preprocess_input
    inception_preprocessor = keras.applications.inception_resnet_v2.preprocess_input
    inc_resnet_preprocessor = keras.applications.inception_resnet_v2.preprocess_input

    # 특징
    nasnet_features_test = get_features(NASNetLarge, nasnet_preprocessor, img_size, pics_array)
    xception_features_test = get_features(Xception, xception_preprocessor, img_size, pics_array)
    inception_features_test = get_features(InceptionV3, inception_preprocessor, img_size, pics_array)
    inc_resnet_features_test = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, pics_array)

    return np.concatenate([nasnet_features_test,
                           xception_features_test,
                           inception_features_test,
                           inc_resnet_features_test], axis=-1)


def predict(features, model):
    return model.predict(features, batch_size=128)


# for similarity
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


# ------------------------------------------------------------------------------------------------------------------------------
@api_view(['GET'])
def HelloAPI(request):
    return Response("hello world")


@api_view(['GET', 'POST'])
def Breed_classify(request):
    if request.method == 'GET':

        # GET request의 http header에 postid을 보낸다면 HTTP_POSTID와 같이 HTTP와 언더바 뒤의 대문자로 긁어와야한다
        # 헤더에 언더바를 써서는 장고에서 인식할 수 없다.
        post_id = request.META.get('HTTP_POSTID')
        # print(request.META.get('HTTP_POSTID'))

        # 여기 코드에 DB에서 POSTPICTURES table의 postid로 사진'들'을 찾은 다음,
        # 위에 따옴표 3개짜리 주석에 있는 코드들로 확인하고
        # 개수를 따져서 가장 높은거 1개를 정하고 Response에 담아서 보내줘야한다

        urls = []

        # sql과 실행
        sql = 'select picture from postpictures where post_id = ' + str(post_id)
        res = cursor.execute(sql)

        # 앞 뒤 자르고 url만 남김
        for line in res:
            urls.append(line.__str__()[2:-3])

        '''
        urls안에 있는 url들을 하나하나 keras의 load.img로 가져온다음
        그걸 numpy배열로 만들어 list comprehension으로 하나의 list로 만들고
        그 리스트로 numpy 배열을 만들어 (사진수, 331, 331, 3) 크기의 numpy 배열을 만든다
        '''
        images = []
        for url in urls:
            response = requests.get(url)
            images.append(Image.open(BytesIO(response.content)))

        images = [image.resize((331, 331)) for image in images]
        images = np.array([np.array(image) for image in images])

        test_images_features = gen_test_features(images)

        y_pred = predict(test_images_features, dog_model)
        ans = []
        for i in range(len(y_pred)):
            ans.append(f'{urls[i]} : {dog_breeds[np.argmax(y_pred[i])]}')

        return Response(ans)



    # 여기서는 multipart_data를 받는 방법을 파악하여
    # 사진의 분류들중 가장 많은수를 확인해야한다.
    elif request.method == 'POST':

        # 이미지 개수 가져옴
        image_len = len(request.data)

        # 이미지 개수만큼 'image' + '번호'를 가져옴
        images = []
        image_names = []
        for i in range(image_len):
            images.append(request.data.get('image' + str(i + 1)))

        # 가져온 이미지 바이트 스트림을 pillow로 변환
        images = [Image.open(image) for image in images]

        # 변환된 이미지를 331x331 크기의 이미지로 변환후 np 배열로 바꿈
        images = [image.resize((331, 331)) for image in images]
        images = np.array([np.array(image) for image in images])

        # 배열로 특징 벡터를 뽑고 모델에 넣어 나온 값으로 추정

        test_images_features = gen_test_features(images)

        y_pred = predict(test_images_features, dog_model)
        ans = []
        for i in range(len(y_pred)):
            ans.append(f'{image_names[i]} : {dog_breeds[np.argmax(y_pred[i])]}')

        return Response(ans)


@api_view(['GET', 'POST'])
def Image_Similarity(request):
    if request.method == 'GET':

        inception_preprocessor = keras.applications.inception_resnet_v2.preprocess_input
        dir_path = os.path.join(root, 'breed_classification', 'two_pics')
        img = images_to_array(dir_path)
        # print(np.shape(img))
        feature_vector = get_features(InceptionV3, inception_preprocessor, img_size, img)

        return Response(cos_sim(feature_vector[0], feature_vector[1]))

    elif request.method == 'POST':
        # 이미지 개수 가져옴
        image_len = len(request.data)

        # 이미지 2개 아니면 거름
        if image_len != 2:
            return Response('BAD REQUEST', status=status.HTTP_400_BAD_REQUEST)

        # 이미지 개수만큼 'image' + '번호'를 가져옴
        images = []

        for i in range(image_len):
            images.append(request.data.get('image' + str(i + 1)))

        # 가져온 이미지 바이트 스트림을 pillow로 변환
        images = [Image.open(image) for image in images]

        # 변환된 이미지를 331x331 크기의 이미지로 변환후 np 배열로 바꿈
        images = [image.resize((331, 331)) for image in images]
        images = np.array([np.array(image) for image in images])

        # 배열로 특징 벡터를 뽑고 모델에 넣어 나온 값으로 추정

        feature_vector = gen_test_features(images)

        return Response(cos_sim(feature_vector[0], feature_vector[1]))


# ------------------------------------------------------------------------------------------------------------------------------
'''
        # 인식 코드
        print(np.shape(images_to_array(os.path.join(root, 'breed_classification', "pics"))))
        test_images_features = gen_test_features(images_to_array(os.path.join(root, 'breed_classification', "pics")))
        y_pred = predict(test_images_features, dog_model)
        list = os.listdir(os.path.join(root, 'breed_classification', 'pics'))
        ans = []
        for i in range(len(y_pred)):
            # print(y_pred[i])
            ans.append(f'{list[i]} : {dog_breeds[np.argmax(y_pred[i])]}')

        return Response(ans)
        '''

'''
        serializer = ClassifySerializer(data=request.data, many=True)
        if serializer.is_valid():
            return Response(serializer.data, status=200)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        '''

'''
        post_id = request.META.get('HTTP_POSTID')

        return Response(post_id)
        '''

'''
        serializer = SimilaritySerializer(data=request.data, many=True)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # print(serializer.data[0]['image'])

        # base64 decode & padding
        str1 = serializer.data[0]['image']
        str2 = serializer.data[1]['image']
        str1 = str1 + '=' * (4 - len(str1) % 4)
        str2 = str2 + '=' * (4 - len(str2) % 4)
        img1 = Image.open(BytesIO(base64.b64decode(str1)))
        img2 = Image.open(BytesIO(base64.b64decode(str2)))

        inception_preprocessor = keras.applications.inception_resnet_v2.preprocess_input
        imgs = images_to_array("muyhoa")
        print("hoyamu")
        print(np.shape(imgs))
        feature_vector = get_features(InceptionV3, inception_preprocessor, img_size, imgs)

        return Response(cos_sim(feature_vector[0], feature_vector[1]))
        '''
