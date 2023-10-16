import requests
from PIL import Image
from io import BytesIO
import numpy as np

urls = ['https://savethepets.s3.ap-northeast-2.amazonaws.com/posts/6/0.jpg', 'https://savethepets.s3.ap-northeast-2.amazonaws.com/posts/6/1.png']
images = np.empty((len(urls), 448, 448, 3))
images_origin = []
print("urls:", urls)

for url in urls:
    response = requests.get(url)
    images_origin.append(Image.open(BytesIO(response.content)).convert('RGB'))

for i in range(len(images_origin)):
    images[i] = np.array(images_origin[i].resize((448, 448)))

print(images.shape)

'''
How to convert PIL to NumPy array?
pip install pillow numpy.
from PIL import Image import numpy as np.
img = Image. open('image.jpg')
img_rgb = img. convert('RGB')
img_array = np. array(img_rgb)
print(img_array. shape)
2023. 7. 23.
'''

'''
print("urls:", urls)
images = np.empty((len(urls), 448, 448, 3))
images_origin = []
for url in urls:
    response = requests.get(url)
    images_origin.append(Image.open(BytesIO(response.content)))

for i in range(len(images)):
    images_origin[i] = images_origin[i].resize((448, 448))

for i in range(len(images)):
    images[i] = np.array(images_origin[i])

print(images)
print(images.shape)
'''

'''
print("urls:", urls)
images = []
for url in urls:
    response = requests.get(url)
    images.append(Image.open(BytesIO(response.content)))

images = [image.resize((448, 448)) for image in images]  # (331, 331)
images = np.array(images)
'''