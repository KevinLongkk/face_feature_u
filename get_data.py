import os
import numpy as np
from PIL import Image

_DATA_SIZE = 5000
_IMAGE_SZIE = 224
def get_data(predict=False):
    if predict:
        global _DATA_SIZE
        _DATA_SIZE = 150
    _X = None
    _pic_label = []
    _coordinate = None
    txt_path = r'./dataset/coordinate.txt'
    image_path = r'./dataset/data_iamge'
    with open(txt_path, 'r') as f:
        data = f.readlines()
        index = 0
        for line in data:
            if index == _DATA_SIZE:
                break
            index += 1
            k = line.split(' ')
            pic_label = k[0]
            coordinate = k[1:-1]
            _pic_label.append(pic_label)
            if _coordinate is not None:
                _coordinate = np.concatenate((_coordinate, coordinate), axis=0)
            else:
                _coordinate = coordinate
        print(_pic_label)
        print(_coordinate)

    pic_path_list = os.listdir(image_path)
    index = 0
    for pic_path in _pic_label:
        if index == _DATA_SIZE:
            break
        index += 1
        raw_data = Image.open(os.path.join(image_path, pic_path))
        img_array = np.asarray(raw_data)
        print(img_array.shape)
        try:
            img_array = img_array.reshape(1, _IMAGE_SZIE*_IMAGE_SZIE*3)
        except:
            img_array = np.concatenate((img_array, img_array, img_array), axis=1)
            img_array = img_array.reshape(1, _IMAGE_SZIE*_IMAGE_SZIE*3)
        if _X is None:
            _X = img_array
        else:
            _X = np.concatenate((_X, img_array), axis=0)
    print(_X.shape)
    return _X, _coordinate
