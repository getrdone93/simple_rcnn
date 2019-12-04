from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow as tf
import os
import argparse
import cv2
import numpy as np
from pycocotools.coco import COCO
import cv2
import os.path as path

def old_detection():
    use_cpu()
    args = parse_args()
    image = load_image(image_path=args.image_path, image_shape=(224, 224))
    regions = four_crop_region(image=image)
    crops = resize_crops(crops=regions, shape=(224, 224))
    preds = list(map(lambda t: (simple_prediction(image=t[0]), t[1]), crops))
    det = all_detections(image=image, crops=preds)

    cv2.imwrite('detection.jpg', det)
    cv2.imwrite('image.jpg', image)
    
    crops = batch_images(images=crops)

def rtx_fix():
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def print_preds(preds):
    for p in preds:
        _, cl, sc = p
        print("class: {}, score: {}".format(cl, sc))

def load_image(image_path, image_shape):
    image = img_to_array(load_img(image_path, target_size=image_shape))
    return image.reshape((image.shape[0], image.shape[1], image.shape[2]))

def four_crop_region(image):
    dim = image.shape[1]
    
    return ((image[0:dim//2, 0:dim//2], ((0, 0), (dim//2, dim//2))),
            (image[0:dim//2, dim//2:dim], ((0, dim//2), (dim//2, dim))),
            (image[dim//2:dim, 0:dim//2], ((dim//2, 0), (dim, dim//2))),
            (image[dim//2:dim, dim//2:dim], ((dim//2, dim//2), (dim, dim))))

def draw_detection(image, sx, sy, ex, ey):
    width = ex - sx
    height = ey - sy
    line_thick = 3
    color = (0, 0, 255)
    cv2.line(image, (sx, sy), (sx + width, sy), color, line_thick)
    cv2.line(image, (sx, sy), (sx, sy + height), color, line_thick)
    cv2.line(image, (sx + width, sy), (sx + width, sy + height), color, line_thick)
    cv2.line(image, (sx, sy + height), (sx + width, sy + height), color, line_thick)

def grab_regions(regions):
    return list(map(lambda r: r[0], regions))

def resize_crops(crops, shape):
    return list(map(lambda t: (cv2.resize(t[0], shape), t[1]), crops))

def batch_images(images):
    return list(map(lambda i: np.expand_dims(i, axis=0), images))

def simple_prediction(image):
    model = VGG16()
    image = np.expand_dims(image, axis=0)
    prep_image = preprocess_input(image)
    preds = decode_predictions(model.predict(prep_image), top=10)
    preds = preds[0] if len(preds) == 1 else preds

    return preds

def output_predictions(crops):
    for e in zip(crops, range(len(crops))):
        c, i = e
        preds = simple_prediction(image=c)
        fn = 'image_' + str(i) + '.jpg'
        print("crop: {}, shape: {}, \npreds:".format(fn, c.shape))
        print_preds(preds=preds)

def use_cpu(use_cpu=True):
    if use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        rtx_fix()

def all_detections(image, crops):
    det_image = np.copy(image)
    for r in preds:
        p, coord = r
        s, e = coord
        sx, sy = s
        ex, ey = e
        _, hi_cls, score = p[0]
        score = str(round(score * 100, 2)) + '%'
        xshift = 5
        yshift = 13
        draw_detection(image=det_image, sx=sx, sy=sy, ex=ex, ey=ey)
        cv2.putText(det_image,
                    hi_cls,
                    (sx + xshift, sy + ((ey - sy) // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    2)
        cv2.putText(det_image,
                    score,
                    (sx + xshift, sy + ((ey - sy) // 2) + yshift),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    2)
    return det_image

def parse_args():
    parser = argparse.ArgumentParser(description='Run detection on images')
    parser.add_argument('--train-images', required=True, help='Path to training images')
    parser.add_argument('--test-images', required=True, help='Path to testing images')
    parser.add_argument('--annotations', required=True, help='Path to annotations file')

    return parser.parse_args()

IMAGE_IDS_FILE = 'image_ids.txt'

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

if __name__ == '__main__':
    args = parse_args()
    val = COCO(args.annotations)
    imgs = read_images(image_path=args.test_images, img_id_file=IMAGE_IDS_FILE, 
                       coco_images=val.imgs)
    train, test = list(map(lambda p: read_images(image_path=p, img_id_file=IMAGE_IDS_FILE, 
                                   coco_images=val.imgs), (args.train_images, args.test_images)))

    print('train: ', len(train))
    print('test: ', len(test))

