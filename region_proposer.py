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
from keras.models import load_model
from keras.layers import Dropout
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as kb
from datetime import datetime

SMALL_OBJ = 32 ** 2
IMAGE_IDS_FILE = 'image_ids.txt'
TRAINED_PATH = './trained'

def parse_args():
    parser = argparse.ArgumentParser(description='Run detection on images')
    parser.add_argument('--train-images-path', required=False, help='Path to training images')
    parser.add_argument('--test-images-path', required=True, help='Path to testing images')
    parser.add_argument('--annotations', required=True, help='Path to annotations file')
    parser.add_argument('--epochs', required=False, default=20, type=int)
    parser.add_argument('--use-gpu', required=False, default=False, action='store_true')
    parser.add_argument('--train', required=False, default=False, action='store_true')
    parser.add_argument('--predict', required=False, default=False, action='store_true')
    parser.add_argument('--pretrained-model', required=False, type=str)

    return parser.parse_args()

def rtx_fix():
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def hardware_setup(use_gpu):
    if use_gpu:
        rtx_fix()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
    return list(result.items())

def image_to_bboxes(images, coco_obj, target_area):
    result, bad_ids = {}, []
    for img_id, img_data in images:
        bboxes = list(map(lambda a: a['bbox'],
                         filter(lambda an: an['area'] < target_area,
                            map(lambda a: coco_obj.anns[a],
                                coco_obj.getAnnIds(imgIds=[img_id])))))
        if len(bboxes) == 5:
            result[img_id] = (img_data, bboxes)
        else:
            bad_ids.append(img_id)
        
    return list(result.items()), bad_ids

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
    for img_id, img_bboxes in image_to_bboxes:
        data, bboxes = img_bboxes
        rscl_bboxes = list(map(lambda a: scale_bbox(bbox=a, orig_shape=data.shape, 
                                             to_shape=img_shape), bboxes))        
        resized[img_id] = (cv2.resize(data, img_shape), rscl_bboxes)
    return list(resized.items())

def example_tensors(image_bboxes):
    xs, ys = zip(*[(data[0], data[1]) for img_id, data in image_bboxes])
    return np.array(xs), np.array(ys).reshape((-1, 20))

#top left small dets model
def custom_model(in_shape):
    w, h = in_shape
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(w, h, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(w, h, 3)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', 
                     kernel_regularizer=regularizers.l1(0.01)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(20, activation='softmax', 
                    kernel_regularizer=regularizers.l1(0.01)))
    return model

def normalize_bbox(im_w, im_h, bbox):
    x, y, w, h = bbox
    return [x/im_w, y/im_h, w/im_w, h/im_h]

def normalize_bboxes(image_to_bboxes, im_w, im_h):
    normalized = {}
    for img_id, img_bboxes in image_to_bboxes:
        img_data, bboxes = img_bboxes
        norm_bboxes = list(map(lambda b: normalize_bbox(im_w=im_w, im_h=im_h, bbox=b),
                               bboxes))
        normalized[img_id] = (img_data, norm_bboxes)
    return list(normalized.items())

def tensors_from_images(images, coco_obj):
    itb, bis = image_to_bboxes(images=images, coco_obj=coco_obj, target_area=SMALL_OBJ)
    rescaled = rescale_bboxes(image_to_bboxes=itb, img_shape=(224, 224))
    normalized = normalize_bboxes(image_to_bboxes=rescaled, im_w=224, im_h=224)
    return example_tensors(image_bboxes=normalized)

def avg_coordinate_distance(y_true, y_pred):
    return kb.mean(kb.sum(kb.abs(y_true[0:2] - y_pred[0:2])))

def avg_w_h_distance(y_true, y_pred):
    return kb.mean(kb.sum(kb.abs(y_true[2:4] - y_pred[2:4])))

def max_coordinate(y_true, y_pred):
    return kb.max(y_pred[0:2])

def max_w_h(y_true, y_pred):
    return kb.max(y_pred[2:4])

def train_model(train, test, img_shape, batch_size, epochs):
    tr_xs, tr_ys = train
    tst_xs, tst_ys = test

    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_itr = datagen.flow(tr_xs, tr_ys, batch_size=batch_size)
    test_itr = datagen.flow(tst_xs, tst_ys, batch_size=batch_size)

    tr_bx, tr_by = train_itr.next()
    tst_bx, tst_by = test_itr.next()

    model = custom_model(in_shape=img_shape)
    print(model.summary())
    model.compile(optimizer=SGD(lr=0.01, momentum=0.8), loss='mean_absolute_error', 
                  metrics=[avg_coordinate_distance, 
                           avg_w_h_distance, 
                           max_coordinate, 
                           max_w_h])
    model.fit_generator(train_itr, steps_per_epoch=len(train_itr), epochs=epochs)
    model.evaluate_generator(test_itr, steps=len(test_itr), verbose=0)

    return model

def save_model(model, fp, epochs):
    sp = path.join(fp, str(epochs) + '_' + datetime.now().strftime("%d-%m-%y_%Hh-%Mm-%Ss") + '.h5')
    print("Saving model to {}".format(sp))
    model.save_weights(sp)

def train_runtime(annotations, train_images_path, test_images_path, img_shape,
                  epochs, img_id_file):
    val = COCO(annotations)
    train, test = list(map(lambda p: read_images(image_path=p, img_id_file=img_id_file, 
                                   coco_images=val.imgs), (train_images_path, test_images_path)))
    tr_tensors, tst_tensors = map(lambda ds: tensors_from_images(images=ds, coco_obj=val),
                                  (train, test))

    print("train_xs: {}, train_ys: {}, xs_min: {}, xs_max: {}, ys_min: {}, ys_max: {}"\
          .format(tr_tensors[0].shape, tr_tensors[1].shape,
                  tr_tensors[0].min(), tr_tensors[0].max(),
                  tr_tensors[1].min(), tr_tensors[1].max()))
    print("test_xs: {}, test_ys: {}, xs_min: {}, xs_max: {}, ys_min: {}, ys_max: {}"\
          .format(tst_tensors[0].shape, tst_tensors[1].shape, 
                  tst_tensors[0].min(), tst_tensors[0].max(),
                  tst_tensors[1].min(), tst_tensors[1].max()))
    model = train_model(train=tr_tensors, test=tst_tensors, batch_size=8,
                             img_shape=img_shape, epochs=epochs)

    save_model(model=model, fp=TRAINED_PATH, epochs=epochs)

    return model

def original_and_tensors(annotations, images_path, img_id_file, img_shape):
    val = COCO(annotations)
    images = read_images(image_path=images_path, img_id_file=img_id_file, 
                                   coco_images=val.imgs)
    itb, _ = image_to_bboxes(images=images, coco_obj=val, target_area=SMALL_OBJ)
    xs, ys = tensors_from_images(images=images, coco_obj=val)

    return itb, (xs, ys), val

def model_from_disk(model_path, img_shape):
    model = custom_model(img_shape)
    model.load_weights(model_path)

    return model

def draw_detection(image, sx, sy, w, h, lt=2):
    ex = sx + w + h
    ey = sy + w + h
    width = ex - sx
    height = ey - sy
    color = (0, 0, 255)
    image = cv2.line(image, (sx, sy), (sx + width, sy), color, lt)
    image = cv2.line(image, (sx, sy), (sx, sy + height), color, lt)
    image = cv2.line(image, (sx + width, sy), (sx + width, sy + height), color, lt)
    image = cv2.line(image, (sx, sy + height), (sx + width, sy + height), color, lt)
    return image

def compare_dets(model, images, tensors): 
    xs, ys = tensors
    preds = model.predict(xs)

    preds = preds.reshape((-1, 5, 4))

    num_dets, w, h, _ = xs.shape
    preds[:, :, 0:4:2] *= w
    preds[:, :, 1:4:2] *= h

    for i in range(len(preds)):
        print(preds[i, :, :])
        inp = input()
        if inp == 'q':
            break

    for og_img, i in zip(images, range(len(images))):
        img_id, img_bboxes = og_img
        oi, bboxes = img_bboxes
        det_img = np.copy(oi)
        for bbox in bboxes:
            x, y, w, h = map(lambda i: round(i), bbox)
            draw_detection(image=det_img, sx=x, sy=y, w=w, h=h)
        
        pred_img = xs[i]
        pred_bboxes = preds[i].tolist()
        pred_det_img = np.copy(pred_img)
        print(pred_det_img.shape)
        for pred_bbox in pred_bboxes:
            px, py, pw, ph = map(lambda i: round(i), pred_bbox)
            print((px, py, pw, ph))
            draw_detection(image=pred_det_img, sx=px, sy=py, w=pw, h=ph)

        cv2.imwrite("og_image_{}_{}.jpg".format(i, img_id), det_img)
        cv2.imwrite("pred_image_{}.jpg".format(i), pred_det_img)
        input()

def prediction_runtime(annotations, images_path, img_id_file, img_shape, model_path):
    images, tensors, coco_obj = original_and_tensors(annotations=annotations, images_path=images_path, 
                                             img_id_file=img_id_file, img_shape=img_shape)
    model = model_from_disk(model_path=model_path, img_shape=img_shape)
    compare_dets(model=model, images=images, tensors=tensors)

def denormalize(w, h, preds):
    result = np.copy(preds)
    result[:, :, 0:4:2] *= w
    result[:, :, 1:4:2] *= h
    return result

def predict(model, tensors):
    xs, ys = tensors
    preds = model.predict(xs)
    preds = preds.reshape((-1, 5, 4))
    ys = ys.reshape((-1, 5, 4))
    num_dets, w, h, _ = xs.shape
    preds = denormalize(w=w, h=h, preds=preds)
    ys = denormalize(w=w, h=h, preds=ys)
    
    return xs, preds, ys

if __name__ == '__main__':
    args = parse_args()
    hardware_setup(use_gpu=args.use_gpu)
    if args.train:
        train_runtime(annotations=args.annotations, train_images_path=args.train_images_path,
                      test_images_path=args.test_images_path, img_shape=(224, 224), epochs=args.epochs, 
                      img_id_file=IMAGE_IDS_FILE)
    
    if args.predict:
        prediction_runtime(annotations=args.annotations, images_path=args.test_images_path, 
                           img_id_file=IMAGE_IDS_FILE, img_shape=(224, 224), 
                           model_path=args.pretrained_model)
    
