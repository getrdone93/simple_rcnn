import region_proposer as rp
import argparse
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input
from random import randint
import pickle
import os.path as path
from functools import reduce

PRED_EXAMPLE_TO_EXP = {1: "The low image intensities and high smoothing factor\nmust have an effect on the classifier. The region proposer predicts boxes at\nthe upper left and is not close to the ground truth. This may be due to overfitting.",
                  2: "The high intensities of the image might contribute to\nthe classifier thinking nematode. The region proposer predicts boxes at the upper\nleft and is not close to ground truth. The classifier might not do well\nwith upscaled region crops.",
                  3: "The classifier predicts cleaver with low confidence.\nThe region proposer predicts upper left quadrant. Perhaps upscaled regions\nthrow the classifier off."}

GT_EXAMPLE_TO_EXP = {1: "The ground truth boxes are different from the\npredicted boxes. They do not produce a different class because the upscaled\nregions throw off the classifier.",
                     2: "Again, the ground truth boxes are different from the\npredicted boxes. They also do not produce a different class score.",
                     3: "This example is interesting because I figured the\nclassifier would be able to decipher a mouse or keyboard. The\nclassifier, however, seems to be thrown off by the upscaled regions."}

def crop_resize_region(x, y, w, h, img, res_shape):
   region = np.copy(img[x:x+h, y:y+w])
   region = cv2.resize(region.astype('float32'), res_shape)
   return region

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
    preds = decode_predictions(model.predict(prep_image), top=3)
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
                (0, 255, 0),
                2)

def shift_pred(x, y, w, h):
    new_x = x + randint(0, 20)
    new_y = y + randint(0, 6)
    return (new_x, new_y, w, h)

def draw_detection(image, dets, pred=False):
    det_image = np.copy(image)
    for d, n in dets:
        x, y, w, h = d
        sx, sy, w, h = shift_pred(x=x, y=y, w=w, h=h) if pred else [x, y, w, h]
        det_image = rp.draw_detection(image=det_image, sx=sx, sy=sy, w=w, h=h, lt=1)
        draw_number(image=det_image, text=str(n), sx=sx, sy=sy, w=w, h=h)
    return det_image

def draw_dets(image_cc):
    image_dets = []
    for i, image in zip(range(1, len(image_cc) + 1), image_cc):
        image_data = image['image']
        pred_coord = list(map(lambda t: (t[1], t[2]), image['pred_crops']))
        gt_coord = list(map(lambda t: (t[1], t[2]), image['gt_crops']))
        pred_det_image = draw_detection(image=image_data, dets=pred_coord, pred=True)
        gt_det_image = draw_detection(image=image_data, dets=gt_coord)

        image_dets.append({'pred_image': pred_det_image,
                           'gt_image': gt_det_image,
                           'gt_crops': image['gt_crops'],
                           'pred_crops': image['pred_crops']})
    return image_dets

def output_descriptions(window_name, det, ex, crop_key, cats):
    num_stars = 90
    print("\n")
    print("*" * num_stars)
    print("Description for window {}:".format(window_name))
    print("\nInput/Output explanation: {}\n".format(ex))
    for ce, gt_c in zip(det[crop_key], cats):
        crop_class, coord, num = ce
        if crop_key == 'gt_crops':
            print("box_number: %-2d ground truth class: %-20s" % (num, gt_c))
        else:
            print("box_number: {}".format(num))
        for _, cn, conf in crop_class:
            print("\tpredicted class: %-20s confidence pct: %-.2f%%"
                  % (str(cn), conf * 100))
    print("*" * num_stars, '\n')

def show_output(dets, use_gpu, cats):
    out_shape = (448, 448)
    for i, d, cs in zip(range(1, len(dets) + 1), dets, cats):
        if use_gpu:
            pred_image = d['pred_image']
            gt_image = d['gt_image']
        else:
            pred_image = cv2.UMat(d['pred_image'].get().astype('f'))
            gt_image = cv2.UMat(d['gt_image'].get().astype('f'))

        new_pred_image = cv2.resize(pred_image, out_shape)
        new_gt_image = cv2.resize(gt_image, out_shape)
        new_pred_image = cv2.normalize(new_pred_image, None, alpha=0, beta=1, 
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        new_gt_image = cv2.normalize(new_gt_image, None, alpha=0, beta=1, 
                                       norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        pred_window_name = "region_proposer_VGG_16_example_{}".format(i)
        gt_window_name = "groundtruth_boxes_VGG_16_example_{}".format(i)
        cv2.imshow(pred_window_name, new_pred_image)
        cv2.imshow(gt_window_name, new_gt_image)
        output_descriptions(window_name=pred_window_name, crop_key='pred_crops',
                            det=d, ex=PRED_EXAMPLE_TO_EXP[i], cats=cs)
        output_descriptions(window_name=gt_window_name, det=d, crop_key='gt_crops',
                            ex=GT_EXAMPLE_TO_EXP[i], cats=cs)
    print("\n\tClick on one of the windows and hit ENTER to exit the program.\n")
    cv2.waitKey(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Run detection on images')
    parser.add_argument('--use-gpu', required=False, default=False, action='store_true')
    parser.add_argument('--annotations', required=False, default='./test_dir/instances_val2014.json')
    parser.add_argument('--model-path', required=False, default='./demo_data/3_09-12-19_18h-42m-01s.h5')
    parser.add_argument('--data-path', required=False, default='./demo_data')

    return parser.parse_args()

def all_detections(anno_path, images_path, model_path, image_shape, image_ids_file):
    """Use this to get all of the data structures required for the demo. Then
       use write_out_structures to write them out to disk.
    """
    itb, tensors, coco_obj = rp.original_and_tensors_and_cats(annotations=anno_path,
                                                              images_path=images_path,
                                                              img_id_file=image_ids_file,
                                                              img_shape=image_shape)
    cats = list(map(lambda t: list(map(lambda bc: bc[1], t[1][1])), itb))
    model = rp.model_from_disk(model_path=model_path, img_shape=image_shape)
    images, preds, gts = rp.predict(model=model, tensors=tensors)
    preds = np.rint(preds).astype(int).tolist()
    gts = np.rint(gts).astype(int).tolist()
    images = list(map(lambda i: np.asarray(i), images.tolist()))
    return images, preds, gts, cats

def write_out_structures(images, preds, gts, cats):
    files = (('./demo_data/images.dat', images), ('./demo_data/preds.dat', preds),
               ('./demo_data/gts.dat', gts), ('./demo_data/cats.dat', cats))
    for f, dat in files:
        print("writing {} file".format(f))
        with open(f, 'ab') as dat_file:
            pickle.dump(dat, dat_file)

if __name__ == '__main__':
    args = parse_args()
    rp.hardware_setup(args.use_gpu)
    images, preds, gts, cats = list(map(lambda f: pickle.load(open(path.join(args.data_path, f), 'rb')),
                                  ['images.dat', 'preds.dat', 'gts.dat', 'cats.dat']))
    image_cc = classify_all_crops(images=images, preds=preds, gts=gts, model=VGG16())
    im_dets = draw_dets(image_cc=image_cc)
    show_output(dets=im_dets, use_gpu=args.use_gpu, cats=cats)

