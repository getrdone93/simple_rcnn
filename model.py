import tensorflow as tf
import os
import argparse
import cv2
import numpy as np
from pycocotools.coco import COCO
import cv2
import os.path as path
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

SMALL_OBJ = 32 ** 2
IMAGE_IDS_FILE = 'image_ids.txt'

def see_dets(resized, ita):
    k = list(resized.keys())[60]
    og_img, og_annos = ita[k]
    ox, oy, ow, oh = map(lambda c: round(c), og_annos[0])
    print(ox, oy, ow, oh)
    draw_detection(image=og_img, sx=ox, sy=oy, ex=ox + ow + oh, ey=oy + ow + oh)
    cv2.imwrite('original.jpg', og_img)

    new_img, new_annos = resized[k]
    nx, ny, nw, nh = new_annos[0]
    draw_detection(image=new_img, sx=nx, sy=ny, ex=nx + nw + nh, ey=ny + nw + nh)
    cv2.imwrite('new.jpg', new_img)

def draw_detection(image, sx, sy, ex, ey, lt=2):
    width = ex - sx
    height = ey - sy
    color = (0, 0, 255)
    cv2.line(image, (sx, sy), (sx + width, sy), color, lt)
    cv2.line(image, (sx, sy), (sx, sy + height), color, lt)
    cv2.line(image, (sx + width, sy), (sx + width, sy + height), color, lt)
    cv2.line(image, (sx, sy + height), (sx + width, sy + height), color, lt)

def rtx_fix():
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def hardware_setup(use_cpu):
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        rtx_fix()

def parse_args():
    parser = argparse.ArgumentParser(description='Run detection on images')
    parser.add_argument('--train-images', required=True, help='Path to training images')
    parser.add_argument('--test-images', required=True, help='Path to testing images')
    parser.add_argument('--annotations', required=True, help='Path to annotations file')

    return parser.parse_args()

def read_file(path):
    with open(path, 'r') as f:
        data = f.read()
    return list(map(lambda i: int(i), data.split('\n')[:-1])) if data else None

def read_images(image_path, img_id_file, coco_images):
    ids = read_file(path=path.join(image_path, img_id_file))
    result = {}
    for i in ids:
        fn = coco_images[i]['file_name']
        result[i] = cv2.imread(path.join(image_path, fn))
    return result

def image_to_bboxes(images, coco_obj, target_area):
    result, bad_ids = {}, []
    for img_id, img_data in images.items():
        bboxes = list(map(lambda a: a['bbox'],
                         filter(lambda an: an['area'] < target_area,
                            map(lambda a: coco_obj.anns[a],
                                coco_obj.getAnnIds(imgIds=[img_id])))))
        if len(bboxes) == 5:
            result[img_id] = (img_data, bboxes)
        else:
            bad_ids.append(img_id)
        
    return result, bad_ids

def load_image(image_path, image_shape):
    image = img_to_array(load_img(image_path, target_size=image_shape))
    return image.reshape((image.shape[0], image.shape[1], image.shape[2]))

def scale_bbox(bbox, orig_shape, to_shape):
    oh, ow, _ = orig_shape
    th, tw = to_shape
    x_ratio = tw / ow
    y_ratio = th / oh
    return [round(bbox[0] * x_ratio),
            round(bbox[1] * y_ratio),
            round(bbox[2] * x_ratio),
            round(bbox[3] * y_ratio)]

def rescale_bboxes(image_to_bboxes, img_shape):
    resized = {}
    p = 2
    for img_id, img_bboxes in image_to_bboxes.items():
        data, bboxes = img_bboxes
        rscl_bboxes = list(map(lambda a: scale_bbox(bbox=a, orig_shape=data.shape, 
                                             to_shape=img_shape), bboxes))        
        resized[img_id] = (cv2.resize(data, img_shape), rscl_bboxes)
    return resized

def example_tensors(image_bboxes):
    xs, ys = zip(*[(data[0], data[1]) for img_id, data in image_bboxes.items()])
    return np.array(xs), tf.keras.utils.normalize(np.array(ys).reshape((-1, 20)))

def conv_net(in_shape):
    w, h = in_shape
    net = Sequential()
    net.add(Conv2D(32, (3, 3), activation='relu', input_shape=(w, h, 3)))
    net.add(MaxPooling2D((2, 2)))
    net.add(Conv2D(64, (3, 3), activation='relu'))
    net.add(MaxPooling2D((2, 2)))
    net.add(Flatten())
    net.add(Dense(20, activation='softmax'))
    return net

def tensors_from_images(images, coco_obj):
    itb, bis = image_to_bboxes(images=images, coco_obj=coco_obj, target_area=SMALL_OBJ)
    rescaled = rescale_bboxes(image_to_bboxes=itb, img_shape=(224, 224))
    return example_tensors(image_bboxes=rescaled)

def train_net(train, test, img_shape, batch_size):
    tr_xs, tr_ys = train
    tst_xs, tst_ys = test

    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_itr = datagen.flow(tr_xs, tr_ys, batch_size=batch_size)
    test_itr = datagen.flow(tst_xs, tst_ys, batch_size=batch_size)
    model = conv_net(in_shape=img_shape)

    # bx, by = train_itr.next()
    # print("test x: {}, test y: {}, min x: {}, max x: {}, min y: {}, max y: {}"\
    #       .format(bx.shape, by.shape, bx.min(), bx.max(), by.min(), by.max()))
    # exit()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit_generator(train_itr, steps_per_epoch=len(train_itr), epochs=20)
    _, acc = model.evaluate_generator(test_itr, steps=len(test_itr), verbose=0)

    print('Test Accuracy: %.3f' % (acc * 100))

if __name__ == '__main__':
    hardware_setup(use_cpu=False)
    img_shape = (224, 224)
    args = parse_args()
    val = COCO(args.annotations)
    train, test = list(map(lambda p: read_images(image_path=p, img_id_file=IMAGE_IDS_FILE, 
                                   coco_images=val.imgs), (args.train_images, args.test_images)))
    tr_tensors, tst_tensors = map(lambda ds: tensors_from_images(images=ds, coco_obj=val),
                                  (train, test))

    print("train_xs: {}, train_ys: {}".format(tr_tensors[0].shape, tr_tensors[1].shape))
    print("test_xs: {}, test_ys: {}".format(tst_tensors[0].shape, tst_tensors[1].shape))
    train_net(train=tr_tensors, test=tst_tensors, batch_size=8, img_shape=img_shape)

    
