import io
from rest_framework.response import Response
from rest_framework.decorators import api_view

from django.http.response import HttpResponse
from .serializer import ClassifySerializer, SimilaritySerializer

# ------------------------------------------------------------------------------------------------------------------------------
import keras.applications.nasnet
import numpy as np
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
import platform

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
if platform.system() == 'Windows':
    LOCATION = r"C:\Users\joy14\PycharmProjects\AIServer\AI\instantclient_21_10_win"  # 나중에 상대경로로 지정
    os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]  # 환경변수 등록, ec2에 올리려면 따로 지정해줘야할듯

# 연결
con = cx_Oracle.connect("scott", "tiger", "host.docker.internal:1521/xepdb1", encoding="UTF-8")
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

    return X


def get_features(model_name, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)
    feature_maps = feature_extractor.predict(data, batch_size=64, verbose=1)

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


def setDBSimilarity(post_id):
    sql = f"select picture, post_id from postpictures where not post_id = {post_id}"
    res = cursor.execute(sql)
    urls = []
    picture_post_id = []
    best_score = 0.0
    best_post_id = -1

    # 앞 뒤 자르고 url만 남김
    for line in res:
        urls.append(line[0].__str__())
        picture_post_id.append(line[1])

    sql = f"select picture from postpictures where post_id = {post_id}"
    res = cursor.execute(sql)
    post_urls = []

    # 앞 뒤 자르고 url만 남김
    for line in res:
        post_urls.append(line.__str__()[2:-3])

    images = []
    for url in urls:
        response = requests.get(url)
        images.append(Image.open(BytesIO(response.content)))

    images = [image.resize((331, 331)) for image in images]
    images = np.array([np.array(image) for image in images])

    post_images = []
    for url in post_urls:
        response = requests.get(url)
        post_images.append(Image.open(BytesIO(response.content)))

    post_images = [image.resize((331, 331)) for image in post_images]
    post_images = np.array([np.array(image) for image in post_images])

    post_feature_vector = gen_test_features(post_images)
    feature_vector = gen_test_features(images)

    for i in range(len(urls)):
        for j in range(len(post_feature_vector)):
            now = cos_sim(post_feature_vector[j], feature_vector[i])
            if now > best_score:
                best_score = now
                best_post_id = picture_post_id[i]
    print(post_id)
    print(best_post_id)

    r = requests.post('http://127.0.0.1:8080/post/analyze', headers={'Content-type': 'application/json'},
                      json={"missingPostId": post_id, "sightPostId": best_post_id})
    print("muyaho")
    print(r.status_code)

    return best_post_id


def get_breed_with_post_id(post_id, species=1):
    urls = []

    sql = 'select picture from postpictures where post_id = ' + str(post_id)
    res = cursor.execute(sql)

    # 앞 뒤 자르고 url만 남김
    for line in res:
        urls.append(line.__str__()[2:-3])

    images = []
    for url in urls:
        response = requests.get(url)
        images.append(Image.open(BytesIO(response.content)))

    images = [image.resize((331, 331)) for image in images]
    images = np.array([np.array(image) for image in images])

    test_images_features = gen_test_features(images)
    y_pred = []
    if species == 1:
        y_pred = predict(test_images_features, dog_model)
    elif species == 0:
        y_pred = predict(test_images_features, cat_model)

    results = {}
    for i in range(len(y_pred)):
        # ans.append(f'{urls[i]} : {dog_breeds[np.argmax(y_pred[i])]}')
        breed = np.argmax(y_pred[i])
        # print(breed)
        if breed in results:
            results[breed] += 1
        else:
            results[breed] = 1

    best_breed = ''
    best_count = -1
    for breed in results.keys():
        if results[breed] > best_count:
            best_count = results[breed]
            best_breed = breed

    return best_breed


# ------------------------------------------------------------------------------------------------------------------------------
@api_view(['GET'])
def HelloAPI(request):
    return Response("hello world")


@api_view(['GET'])
def POSTID(request):
    try:
        post_id = int(request.GET.get('postId'))

        # 게시글 타입, 개/고양이 정보 확인
        post_type = -1
        species = -1  # 0은 고양이, 1은 개
        cursor.execute(f"SELECT type, species FROM posts WHERE post_id = {int(post_id)}")
        result = cursor.fetchone()
        if result:
            post_type = result[0]
            species = result[1]

        # 0은 실종, 실종의 경우에만 유사도 분석/나머지는 품종 분류
        if type == 0:
            setDBSimilarity(post_type)
        else:
            breed = get_breed_with_post_id(post_id, species)
            # UPDATE [테이블] SET [열] = '변경할값' WHERE [조건]
            sql = f"update posts set breed_ai = {breed} where post_id = {post_id}"
            cursor.execute(sql)
            sql = 'commit'
            cursor.execute(sql)

        return HttpResponse(post_id, content_type='text/plain')
    except:
        return HttpResponse('error', status=404)


@api_view(['GET', 'POST'])
def Breed_classify(request):
    if request.method == 'GET':
        post_id = request.META.get('HTTP_POSTID')

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

    # 사진의 분류들중 가장 많은수를 확인해야한다.
    elif request.method == 'POST':
        try:
            # 이미지 개수 가져옴
            image_len = len(request.data)

            # 이미지 개수만큼 'image' + '번호'를 가져옴
            images = []
            image_names = []
            species = int(request.data.get('species'))
            for i in range(image_len - 1):
                images.append(request.data.get('image' + str(i + 1)))
                image_names.append(request.data.get('image' + str(i + 1)).__str__())

            # 가져온 이미지 바이트 스트림을 pillow로 변환
            images = [Image.open(image) for image in images]

            # 변환된 이미지를 331x331 크기의 이미지로 변환후 np 배열로 바꿈
            images = [image.resize((331, 331)) for image in images]
            images = np.array([np.array(image) for image in images])

            # 배열로 특징 벡터를 뽑고 모델에 넣어 나온 값으로 추정

            test_images_features = gen_test_features(images)
            y_pred = []
            if species == 1:
                y_pred = predict(test_images_features, dog_model)
            elif species == 0:
                y_pred = predict(test_images_features, cat_model)

            results = {}
            for i in range(len(y_pred)):
                breed = np.argmax(y_pred[i])
                if breed in results:
                    results[breed] += 1
                else:
                    results[breed] = 1

            best_breed = ''
            best_count = -1
            for breed in results.keys():
                if results[breed] > best_count:
                    best_count = results[breed]
                    best_breed = breed

            return Response(best_breed)
        except:
            return Response('error', status=404)


@api_view(['POST'])
def Image_Similarity(request):
    if request.method == 'POST':
        post_id = request.data.get('post_id')
        best = setDBSimilarity(post_id)

        return Response(best)
