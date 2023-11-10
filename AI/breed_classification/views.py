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

# --------------------------------------------------------------------------------
root = os.getcwd()

img_size = (448, 448, 3)
dog_model = keras.models.load_model(
    os.path.join(root, 'breed_classification', 'models', 'dog_model.h5'))
cat_model = keras.models.load_model(
    os.path.join(root, 'breed_classification', 'models', 'cat_model.h5'))

dog_breeds = ['bishon_frise', 'chihuahua', 'chow_chow', 'dalmatian', 'doberman_pinscher', 'golden_retriever',
              'pomeranian', 'poodle', 'pug', 'siberian_husky', 'welsh_corgi', 'yorkshire_terrier']

cat_breeds = ['bengal_cat', 'bombay_cat', 'british_shorthair_cat', 'ragdoll_cat', 'russian_blue_cat', 'siamese_cat',
              'sphynx_cat']

# print(os.path.join(root, 'breed_classification', 'pics'))

# ------stack_VIT(feature model)-------------------------------------------------

from keras.applications import InceptionResNetV2, EfficientNetB3
from vit_keras import vit
import keras

from keras import Model, layers
from keras.layers import Concatenate, RandomZoom

# from env_var import *
specie_name = 'cat'  # 'dog'
dog_breeds = ['bishon_frise', 'chihuahua', 'chow_chow', 'dalmatian', 'doberman_pinscher', 'golden_retriever',
              'pomeranian', 'poodle', 'pug', 'siberian_husky', 'welsh_corgi', 'yorkshire_terrier']

cat_breeds = ['bengal_cat', 'bombay_cat', 'british_shorthair_cat', 'ragdoll_cat', 'russian_blue_cat', 'siamese_cat',
              'sphynx_cat']
IMAGE_SIZE = (448, 448, 3)

models_search = {
    'InceptionResNetV2': [InceptionResNetV2, keras.applications.inception_resnet_v2.preprocess_input],
    'EfficientNetB3': [EfficientNetB3, keras.applications.efficientnet.preprocess_input],
    'vit_l32': [vit.vit_l32, vit.preprocess_inputs],
}
models_for_stacking = ['vit_l32', 'EfficientNetB3', 'InceptionResNetV2']

# 이미지에 노이즈를 주어 증강, augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    RandomZoom(0.1),
], name='data_augmentation')

breeds_count = 12


def get_vit_model_feat(app_class, shape, inputs, prep_input_fx):
    base_model = app_class(
        image_size=shape[:2],  # 이미지 가로세로
        activation='softmax',
        pretrained=True,
        include_top=False,
        pretrained_top=False,
        classes=breeds_count
    )
    x = prep_input_fx(inputs)  # 전처리
    outputs = base_model(x)  # 한걸 모델에 넣어 output
    return outputs


def get_keras_model_feat(app_class, shape, inputs, prep_input_fx):
    base_model = app_class(
        include_top=False,
        weights='imagenet',  # ImageNet으로 학습한 가중치를 이용하는 모델들과 관련? https://keras.io/ko/applications/
        input_shape=shape,
    )
    x = prep_input_fx(inputs)
    x = base_model(x)
    outputs = keras.layers.GlobalAveragePooling2D()(x)
    return outputs


def build_feat_model(models_names, shape, aug_layer=None):
    all_outputs = []
    inputs = keras.Input(shape=shape)
    if aug_layer != None:
        aug_inputs = aug_layer(inputs)
    else:
        aug_inputs = inputs
    for model_type in models_names:
        model_class = models_search[model_type][0]  # 모델?
        model_prep_input = models_search[model_type][1]  # 모델별 이미지 전처리기?
        if model_type.startswith('vit'):
            model_outputs = get_vit_model_feat(model_class, shape, aug_inputs, model_prep_input)
        else:
            model_outputs = get_keras_model_feat(model_class, shape, aug_inputs, model_prep_input)
        all_outputs.append(model_outputs)
    concat_outputs = Concatenate()(all_outputs)
    model = Model(inputs, concat_outputs)
    return model


def get_feature_model():
    return build_feat_model(models_for_stacking, IMAGE_SIZE, data_augmentation)


dog_feature_model = get_feature_model()
breeds_count = 7
cat_feature_model = get_feature_model()

# -------DB client&connect---------------------------------------------------------------------------------------------------------------

if platform.system() == 'Windows':
    LOCATION = r"C:\Users\joy14\PycharmProjects\AIServer\AI\instantclient_21_10_win"  # 나중에 상대경로로 지정
    os.environ["PATH"] = LOCATION + ";" + os.environ["PATH"]  # 환경변수 등록, ec2에 올리려면 따로 지정해줘야할듯

# 연결
# con = cx_Oracle.connect("scott", "tiger", "host.docker.internal:1521/xepdb1", encoding="UTF-8")
# con = cx_Oracle.connect("scott", "tiger", "127.0.0.1:1521/xepdb1", encoding="UTF-8")
con = cx_Oracle.connect("SCOTT", "tiger", "110.8.166.180:1521/XE", encoding="UTF-8")

cursor = con.cursor()

# -------------------Process--------------------------------------------------------------------------------------------------

def images_to_array(data_dir):
    os.chdir(data_dir)
    images_names = os.listdir(data_dir)
    test_size = len(images_names)

    X = np.zeros([test_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    for i in tqdm(range(test_size)):
        image_name = images_names[i]
        img_dir = os.path.join(data_dir, image_name)
        img_pixels = tf.keras.preprocessing.image.load_img(img_dir, target_size=img_size)
        X[i] = img_pixels

    return X


def get_features_from_model(model, ds, count=1):
    features = []

    for i in range(count):
        predictions = model.predict(ds)
        features.append(predictions)

    return np.concatenate(features)


def gen_vit_keras_test_feature(pics_array, specie_name='dog'):
    # model_features = get_feature_model()
    if specie_name == 'dog':
        return get_features_from_model(dog_feature_model, pics_array, count=1)
    else:
        return get_features_from_model(cat_feature_model, pics_array, count=1)


def predict(features, model):
    return model.predict(features, batch_size=128)


# for similarity
def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))


def setDBSimilarity(post_id, post_type = 0, species='1'):
    print(post_id)

    sql = f"select picture, post_id from postpictures where not post_id = {str(post_id)}"# and (select type from posts where post_id = postpictures.post_id) = {post_type}"

    res = cursor.execute(sql)
    urls = []
    picture_post_id = []
    best_score = 0.0
    best_post_id = -1

    # 앞 뒤 자르고 url만 남김
    for line in res:
        urls.append(line[0].__str__())
        picture_post_id.append(line[1])

    sql = f"select picture from postpictures where post_id = {str(post_id)}"
    res = cursor.execute(sql)
    post_urls = []

    # 앞 뒤 자르고 url만 남김
    for line in res:
        post_urls.append(line.__str__()[2:-3])

    images_origin = []
    images = np.empty((len(urls), 448, 448, 3))

    #print("urls:", urls)
    #print("post_urls: ", post_urls)

    for url in urls:
        response = requests.get(url)
        images_origin.append(Image.open(BytesIO(response.content)).convert('RGB'))

    for i in range(len(images_origin)):
        images[i] = np.array(images_origin[i].resize((448, 448)))

    #print(len(images))
    #print("무야호5")
    #print(images.shape)
    #print("무야호6")

    post_images_origin = []
    post_images = np.empty((len(urls), 448, 448, 3))

    for url in post_urls:
        response = requests.get(url)
        post_images_origin.append(Image.open(BytesIO(response.content)).convert('RGB'))

    for i in range(len(post_images_origin)):
        post_images[i] = np.array(post_images_origin[i].resize((448, 448)))

    '''
    post_images = [image.resize((448, 448)) for image in post_images]  # (331, 331)
    post_images = np.array(post_images)# np.array([np.array(image) for image in post_images])
    '''

    # print(post_images)

    post_feature_vector = []  # gen_vit_keras_test_feature(post_images, 'dog')  # gen_test_features(post_images)
    feature_vector = []  # gen_vit_keras_test_feature(images, 'dog')  # gen_test_features(images)
    #print("무야호7")
    if species == '1':
        post_feature_vector = gen_vit_keras_test_feature(post_images, 'dog')
        feature_vector = gen_vit_keras_test_feature(images, 'dog')
    elif species == '0':
        post_feature_vector = gen_vit_keras_test_feature(post_images, 'cat')
        feature_vector = gen_vit_keras_test_feature(images, 'cat')
    #print("무야호8")
    #print(post_feature_vector.shape)

    for i in range(len(urls)):
        for j in range(len(post_feature_vector)):
            now = cos_sim(post_feature_vector[j], feature_vector[i])
            if now > best_score:
                best_score = now
                best_post_id = picture_post_id[i]
    # print(post_id)
    print("best_post_id: ", best_post_id)

    #print(best_post_id)
    #print("무야호9")
    try:
        r = requests.post('http://110.8.166.180:4000/post/analyze', headers={'Content-type': 'application/json'},
                          json={"missingPostId": post_id, "sightPostId": best_post_id})
        # https://savethepets.kro.kr/spring/analyze
        #print(r.status_code)
    except:
        print('Error in Backend Spring Server')
    #print("무야호10")
    return best_post_id


def get_breed_with_post_id(post_id, species=1):
    urls = []

    sql = 'select picture from postpictures where post_id = ' + str(post_id)
    res = cursor.execute(sql)

    # 앞 뒤 자르고 url만 남김
    for line in res:
        urls.append(line.__str__()[2:-3])

    images = np.empty((len(urls), 448, 448, 3))
    images_origin = []
    for url in urls:
        response = requests.get(url)
        images_origin.append(Image.open(BytesIO(response.content)).convert('RGB'))

    for i in range(len(images_origin)):
        images[i] = np.array(images_origin[i].resize((448, 448)))

    test_images_features = []
    if species == 1:
        test_images_features = gen_vit_keras_test_feature(images, 'dog')  # gen_test_features(images)
    elif species == 0:
        test_images_features = gen_vit_keras_test_feature(images, 'cat')
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

# postId가 들어오는 순간 종을 분류하고 DB 갱신
# 갱신하고 가장 유사한 post를 골라 DB의 ALARMS도 생성
@api_view(['GET'])
def POSTID(request):
    try:
        post_id = int(request.GET.get('postId'))
        print(post_id)
        breed = -1

        # 게시글 타입, 개/고양이 정보 확인
        post_type = -1
        species = -1  # 0은 고양이, 1은 개
        cursor.execute(f"SELECT type, species FROM posts WHERE post_id = {str(post_id)}")
        result = cursor.fetchone()
        if result:
            post_type = int(result[0])
            species = int(result[1])

        print('post_type = ', post_type)


        # 0은 실종, 1은 목격 둘 다 유사도 분석
        # 목격의 경우에만 품종분류
        if post_type == 0:
            print('before similarity')
            setDBSimilarity(post_id)
            print('similarity success')
        else:
            print('before similarity')
            setDBSimilarity(post_id)
            print('similarity success')

            breed = get_breed_with_post_id(post_id, species)
            print("breed: ", breed)
            # UPDATE [테이블] SET [열] = '변경할값' WHERE [조건]
            sql = f"update posts set breed_ai = {str(breed)} where post_id = {str(post_id)}"
            cursor.execute(sql)
            sql = 'commit'
            cursor.execute(sql)

        return HttpResponse(post_id, content_type='text/plain')
    except:
        return HttpResponse(post_id, content_type='text/plain')
        # return HttpResponse('error', status=404)

# GET은 무시(레거시), POST의 경우 사진을 image1, image2... 순서로 받고
# species도 받아서 품종 분석해서 바로 보내줌
# front에서 즉시 품종 분석할 때 쓴다
@api_view(['GET', 'POST'])
def Breed_classify(request):
    if request.method == 'GET':
        post_id = request.META.get('HTTP_POSTID')
        print(post_id)
        urls = []

        # sql과 실행
        sql = 'select picture from postpictures where post_id = ' + str(post_id)
        res = cursor.execute(sql)

        # 앞 뒤 자르고 url만 남김
        for line in res:
            urls.append(line.__str__()[2:-3])

        images = []
        for url in urls:
            response = requests.get(url)
            images.append(Image.open(BytesIO(response.content)))

        images = [image.resize((448, 488)) for image in images]  # (331, 331)
        images = np.array([np.array(image) for image in images])

        print(res, type(res))

        sql = f'select species from posts where post_id = {str(post_id)}'
        res = cursor.execute(sql)
        species = -1
        for line in res:
            species = int(line[0])

        test_images_features = []
        if species == 1:
            test_images_features = gen_vit_keras_test_feature(images, 'dog')  # gen_test_features(images)
        elif species == 0:
            test_images_features = gen_vit_keras_test_feature(images, 'cat')

        if species == 1:
            y_pred = predict(test_images_features, dog_model)
            ans = []
            for i in range(len(y_pred)):
                ans.append(f'{urls[i]} : {dog_breeds[np.argmax(y_pred[i])]}')

            return Response(ans)
        elif species == 0:
            y_pred = predict(test_images_features, cat_model)
            ans = []
            for i in range(len(y_pred)):
                ans.append(f'{urls[i]} : {cat_breeds[np.argmax(y_pred[i])]}')

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
            images = [Image.open(image).convert('RGB') for image in images]

            # 변환된 이미지를 331x331 크기의 이미지로 변환후 np 배열로 바꿈
            images = [image.resize((448, 448)) for image in images]
            images = np.array([np.array(image) for image in images])

            # 배열로 특징 벡터를 뽑고 모델에 넣어 나온 값으로 추정

            test_images_features = []
            if species == 1:
                test_images_features = gen_vit_keras_test_feature(images, 'dog')  # gen_test_features(images)
            elif species == 0:
                test_images_features = gen_vit_keras_test_feature(images, 'cat')

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
        except Exception as ex:
            return Response('error', status=500)

# post_id 값을 json으로 보내주면 가장 닮은 사진을 가진 post의 post_id를 보내준다
# 이것도 레거시 인가?
@api_view(['POST'])
def Image_Similarity(request):
    if request.method == 'POST':
        post_id = request.data.get('post_id')

        sql = f'select species from posts where post_id = {str(post_id)}'
        res = cursor.execute(sql)
        species = -1
        for line in res:
            species = line[0]

        best = setDBSimilarity(post_id) #, species)

        return Response(best)


# --------------------------------------------------------------------------------------------------------

'''
def get_sql_result(sql):
    return cursor.execute(sql)
'''

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

'''
urls안에 있는 url들을 하나하나 keras의 load.img로 가져온다음
그걸 numpy배열로 만들어 list comprehension으로 하나의 list로 만들고
그 리스트로 numpy 배열을 만들어 (사진수, 331, 331, 3) 크기의 numpy 배열을 만든다
'''

'''
def gen_test_features(pics_array):
    # 전처리기 원래 위치

    # 특징
    nasnet_features_test = get_features(NASNetLarge, nasnet_preprocessor, img_size, pics_array)
    xception_features_test = get_features(Xception, xception_preprocessor, img_size, pics_array)
    inception_features_test = get_features(InceptionV3, inception_preprocessor, img_size, pics_array)
    inc_resnet_features_test = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, pics_array)

    return np.concatenate([nasnet_features_test,
                           xception_features_test,
                           inception_features_test,
                           inc_resnet_features_test], axis=-1)
'''

'''
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
'''

'''
# 전처리기
nasnet_preprocessor = keras.applications.nasnet.preprocess_input
xception_preprocessor = keras.applications.inception_resnet_v2.preprocess_input
inception_preprocessor = keras.applications.inception_resnet_v2.preprocess_input
inc_resnet_preprocessor = keras.applications.inception_resnet_v2.preprocess_input
'''

'''
def get_features(model_name, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model_name(weights='imagenet', include_top=False,
                            input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)
    feature_maps = feature_extractor.predict(data, batch_size=64, verbose=1)

    return feature_maps
'''
