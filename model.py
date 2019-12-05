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

def rtx_fix():
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def use_cpu(use_cpu=True):
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

def draw_detection(image, sx, sy, ex, ey, lt=2):
    width = ex - sx
    height = ey - sy
    color = (0, 0, 255)
    cv2.line(image, (sx, sy), (sx + width, sy), color, lt)
    cv2.line(image, (sx, sy), (sx, sy + height), color, lt)
    cv2.line(image, (sx + width, sy), (sx + width, sy + height), color, lt)
    cv2.line(image, (sx, sy + height), (sx + width, sy + height), color, lt)

if __name__ == '__main__':
    args = parse_args()
    val = COCO(args.annotations)
    imgs = read_images(image_path=args.test_images, img_id_file=IMAGE_IDS_FILE, 
                       coco_images=val.imgs)
    train, test = list(map(lambda p: read_images(image_path=p, img_id_file=IMAGE_IDS_FILE, 
                                   coco_images=val.imgs), (args.train_images, args.test_images)))
    itb, bis = image_to_bboxes(images=train, coco_obj=val, target_area=SMALL_OBJ)
    rescaled = rescale_bboxes(image_to_bboxes=itb, img_shape=(224, 224))
    see_dets(rescaled, itb)
