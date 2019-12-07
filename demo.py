import region_proposer as rp
import argparse
import numpy as np
import cv2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input

def parse_args():
    parser = argparse.ArgumentParser(description='Run detection on images')
    parser.add_argument('--use-gpu', required=False, default=False, action='store_true')

    return parser.parse_args()

def crop_resize_region(x, y, w, h, img, res_shape):
   region = np.copy(img[x:x+h, y:y+w])
   region.resize(res_shape)

   print("img: {}, res_shape: {}, region: {}".format(img.shape, res_shape, region.shape))
   #input()
   return region

def all_detections():
    anno_path = './test_dir/instances_val2014.json'
    images_path = './data/test'
    model_path = './trained/5_07-12-19_08h-46m-18s.h5'
    image_shape = (224, 224)
    itb, tensors, coco_obj = rp.original_and_tensors(annotations=anno_path, images_path=images_path,
                         img_id_file=rp.IMAGE_IDS_FILE, img_shape=image_shape)
    print(tensors[0].shape)
    for im, img in itb:
        print(img[0].shape)
        break
    #input()

    model = rp.model_from_disk(model_path=model_path, img_shape=image_shape)
    images, preds, gts = rp.predict(model=model, tensors=tensors)
    preds = np.rint(preds).astype(int).tolist()
    gts = np.rint(gts).astype(int).tolist()
    images = list(map(lambda i: np.asarray(i), images.tolist()))

    return images, preds, gts

def image_to_crops(image, preds, gts, crop_shape):
    return {'image': image,
            'pred_crops': map(lambda p:
                              (crop_resize_region(x=p[0], y=p[1], w=p[2], h=p[3], img=image, 
                                                  res_shape=crop_shape),
                               (p[0], p[1], p[2], p[3])), preds),
            'gt_crops' : map(lambda gt: 
                            (crop_resize_region(x=gt[0], y=gt[1], w=gt[2], h=gt[3], img=image, 
                                                res_shape=crop_shape),
                             (gt[0], gt[1], gt[2], gt[3])), gts)}

def classify_crop(crop, model):
    image = np.expand_dims(crop, axis=0)
    prep_image = preprocess_input(image)
    preds = decode_predictions(model.predict(prep_image), top=10)
    return preds[0] if len(preds) == 1 else preds

def classify_all_crops(images, preds, gts, model):
    classified = []
    for img, p, gt in zip(images, preds, gts):
        itc = image_to_crops(image=img, preds=p, gts=gt, crop_shape=(224, 224, 3))
        classified.append({'image': itc['image'],
                           'pred_crops': map(lambda t: (classify_crop(crop=t[0], model=model), t[1]),
                                             itc['pred_crops']), 
                           'gt_crops': map(lambda t: (classify_crop(crop=t[0], model=model), t[1]),
                                             itc['gt_crops'])})
    return classified

def show_output(image_cc):
    for image in image_cc:
        cv2.imwrite('image.jpg', image['image'])
        print('pred_crops: ', list(image['pred_crops']))
        print('gt_crops: ', list(image['gt_crops']))
        input()

if __name__ == '__main__':
    args = parse_args()
    rp.hardware_setup(args.use_gpu)
    images, preds, gts = all_detections()

    #take 3 for demo
    images = images[0:3]
    preds = preds[0:3]
    gts = gts[0:3]

    image_cc = classify_all_crops(images=images, preds=preds, gts=gts, model=VGG16())
    show_output(image_cc=image_cc)

    
