import region_proposer as rp
import argparse
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
import tkinter as tk
from PIL import ImageTk, Image
from matplotlib import cm
from random import randint

def parse_args():
    parser = argparse.ArgumentParser(description='Run detection on images')
    parser.add_argument('--use-gpu', required=False, default=False, action='store_true')

    return parser.parse_args()

def crop_resize_region(x, y, w, h, img, res_shape):
   region = np.copy(img[x:x+h, y:y+w])
   region = cv2.resize(region.astype('float32'), res_shape)
   return region

def all_detections(anno_path, images_path, model_path, image_shape, image_ids_file):
    itb, tensors, coco_obj = rp.original_and_tensors(annotations=anno_path, images_path=images_path,
                                                     img_id_file=image_ids_file, img_shape=image_shape)

    model = rp.model_from_disk(model_path=model_path, img_shape=image_shape)
    images, preds, gts = rp.predict(model=model, tensors=tensors)
    preds = np.rint(preds).astype(int).tolist()
    gts = np.rint(gts).astype(int).tolist()
    images = list(map(lambda i: np.asarray(i), images.tolist()))
    return images, preds, gts

def image_to_crops(image, preds, gts, crop_shape):
    return {'image': image,
            'pred_crops': list(map(lambda p:
                              (crop_resize_region(x=p[0], y=p[1], w=p[2], h=p[3], img=image, 
                                                  res_shape=crop_shape),
                               (p[0], p[1], p[2], p[3])), preds)),
            'gt_crops' : list(map(lambda gt: 
                            (crop_resize_region(x=gt[0], y=gt[1], w=gt[2], h=gt[3], img=image,
                                                res_shape=crop_shape),
                             (gt[0], gt[1], gt[2], gt[3])), gts))}

def classify_crop(crop, model):
    image = np.expand_dims(crop, axis=0)
    prep_image = preprocess_input(image)
    preds = decode_predictions(model.predict(prep_image), top=10)
    return preds[0] if len(preds) == 1 else preds

def classify_all_crops(images, preds, gts, model):
    classified = []
    for img, p, gt in zip(images, preds, gts):
        itc = image_to_crops(image=img, preds=p, gts=gt, crop_shape=(224, 224))
        classified.append({'image': itc['image'],
                           'pred_crops': list(map(lambda t: (classify_crop(crop=t[0][0], model=model), t[0][1], t[1]),
                                             zip(itc['pred_crops'], range(1, len(itc['pred_crops']) + 1)))),
                           'gt_crops': list(map(lambda t: (classify_crop(crop=t[0][0], model=model), t[0][1], t[1]),
                                             zip(itc['gt_crops'], range(1, len(itc['gt_crops']) + 1))))})
    return classified

def draw_number(image, text, sx, sy, w, h):
    cv2.putText(image,
                text,
                (sx, sy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                2)

def shift_pred(x, y, w, h):
    new_x = x + randint(0, 20)
    new_y = y + randint(0, 6)
    return (new_x, new_y, w, h)

def draw_detection(image, dets):
    det_image = np.copy(image)
    for d, n in dets:
        x, y, w, h = d
        sx, sy, w, h = shift_pred(x=x, y=y, w=w, h=h)
        det_image = rp.draw_detection(image=det_image, sx=sx, sy=sy, w=w, h=h, lt=1)
        draw_number(image=det_image, text=str(n), sx=sx, sy=sy, w=w, h=h)
    return det_image

def draw_dets(image_cc):
    image_dets = []
    for i, image in zip(range(1, len(image_cc) + 1), image_cc):
        image_data = image['image']
        pred_coord = list(map(lambda t: (t[1], t[2]), image['pred_crops']))
        gt_coord = list(map(lambda t: (t[1], t[2]), image['gt_crops']))
        pred_det_image = draw_detection(image=image_data, dets=pred_coord)
        gt_det_image = draw_detection(image=image_data, dets=gt_coord)

        image_dets.append({'pred_image': pred_det_image,
                           'gt_image': gt_det_image,
                           'gt_crops': image['gt_crops'],
                           'pred_crops': image['pred_crops']})
    return image_dets

def write_disk(dets, use_gpu):
    out_shape = (448, 448)
    for i, d in enumerate(dets):
        if use_gpu:
            pred_image = d['pred_image']
            gt_image = d['gt_image']
        else:
            pred_image = cv2.UMat(d['pred_image'].get().astype('f'))
            gt_image = cv2.UMat(d['gt_image'].get().astype('f'))

        new_pred_image = cv2.resize(pred_image, out_shape)
        new_gt_image = cv2.resize(gt_image, out_shape)

        cv2.imwrite("pred_image_{}.jpg".format(i), new_pred_image)
        cv2.imwrite("gt_image_{}.jpg".format(i), new_gt_image)

if __name__ == '__main__':
    args = parse_args()
    rp.hardware_setup(args.use_gpu)

    anno_path = './test_dir/instances_val2014.json'
    images_path = './demo_data/'
    model_path = './trained/5_07-12-19_08h-46m-18s.h5'
    image_shape = (224, 224)

    images, preds, gts = all_detections(anno_path=anno_path, images_path=images_path, 
                                        model_path=model_path, image_shape=image_shape, 
                                        image_ids_file=rp.IMAGE_IDS_FILE)
    image_cc = classify_all_crops(images=images, preds=preds, gts=gts, model=VGG16())
    im_dets = draw_dets(image_cc=image_cc)
    write_disk(dets=im_dets, use_gpu=args.use_gpu)

    
