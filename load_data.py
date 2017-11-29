import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plot
import math

_IMAGE_SZIE = 224
_pic_label = []
_coordinate = None

with open(r'./dataset/WebFaces_GroundThruth.txt', 'r') as f:
    data = f.readlines()
    for line in data:
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

pic_path_list = os.listdir(r'./dataset/picture')
step = 0
f = open(r'./dataset/coordinate.txt', 'w')
for path in _pic_label:
    pic_path = os.path.join(r'./dataset/picture', path)
    pic = Image.open(pic_path)
    w, h = pic.size
    pic = pic.resize((_IMAGE_SZIE, _IMAGE_SZIE), Image.ANTIALIAS)
    pic.save(os.path.join(r'./dataset/data_iamge', path))
    prop_x = _IMAGE_SZIE / w
    prop_y = _IMAGE_SZIE / h
    print(step)
    f.write(path+' ')
    for i in range(8*step, 8*(step+1)):
        if i % 2 == 0:
            f.write(str(float(_coordinate[i])*prop_x)+' ')
        else:
            f.write(str(float(_coordinate[i])*prop_y) + ' ')
    f.write('\n')
    step += 1
f.close()

# pic_path = os.path.join(r'./dataset/picture', _pic_label[0])
# pic = Image.open(pic_path)
# x, y = pic.size
# pic = pic.resize((_IMAGE_SZIE, _IMAGE_SZIE), Image.ANTIALIAS)
# pic_array = np.array(pic)
# prop_x = _IMAGE_SZIE/x
# prop_y = _IMAGE_SZIE/y
# # pic = pic_array.reshape([3, x, y])
# # x, y = pic_array.size
# print(x, y)
# print(pic_array.shape)
# # pic_array = np.resize(pic_array, [245, 225, 3])
# plot.imshow(pic_array)
# plot.plot(float(_coordinate[0])*prop_x, float(_coordinate[1])*prop_y, 'r*')
# plot.plot(float(_coordinate[2])*prop_x, float(_coordinate[3])*prop_y, 'r*')
# # plot.plot(_coordinate[4]*prop_x, _coordinate[5]*prop_y, 'r*')
# # plot.plot(_coordinate[6]*prop_x, _coordinate[7]*prop_y, 'r*')
# plot.show()

