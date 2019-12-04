import operator as op
import argparse
from pycocotools.coco import COCO
import shutil
import os.path as path
from functools import reduce
import os

SMALL_OBJ = 32 ** 2
IMAGE_ID_FN = 'image_ids.txt'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Find small objects')
    parser.add_argument('--from-image-path', required=True)
    parser.add_argument('--to-train-path', required=True)
    parser.add_argument('--to-test-path', required=True)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--classes', nargs='+', required=False)
    parser.add_argument('--split-percent', required=True, type=float)

    return parser.parse_args()

def area_histo(coco_val, area):
    histo = {}
    for img_id in coco_val.imgs:
        img_anns = coco_val.getAnnIds(imgIds=[img_id])
        small_ia = list(filter(lambda a: coco_val.anns[a]['area'] < area, img_anns))
        num_ia = len(small_ia)
        if num_ia not in histo:
            histo[num_ia] = []
        histo[num_ia].append((img_id, coco_val.imgs[img_id]))

    return histo      

def split_images(histo, obj_num, sp):
    imgs = histo[obj_num]
    ni = len(imgs)
    num_train = round(ni * sp)
    num_test = ni - num_train
    train, test = [], []

    for i in range(0, num_train):
        train.append(imgs[i])
    for j in range(num_train, num_train+num_test):
        test.append(imgs[j])

    return train, test

def copy_images(img_coll, from_path, to_path):
    copied = []
    for e in img_coll:
        _, img = e
        copied.append(shutil.copyfile(path.join(from_path, img['file_name']),
                        path.join(to_path, img['file_name'])))
    return copied

def write_ids(ids, file_path):
    str_ids = list(map(lambda i: str(i), ids))
    with open(file_path, 'w') as fp:
        written = fp.write('\n'.join(str_ids) + '\n')

    return written

if __name__ == '__main__':
    args = parse_arguments()
    val = COCO(args.annotations)
    histo = area_histo(coco_val=val, area=SMALL_OBJ)
    train, test = split_images(histo=histo, obj_num=5, sp=args.split_percent)
    print('Copying images...')
    copied = list(reduce(lambda l1, l2: l1 + l2,
                         map(lambda ic: copy_images(img_coll=ic[0],
                            from_path=args.from_image_path, to_path=ic[1]),
                             ((train, args.to_train_path), (test, args.to_test_path)))))

    train_ids = list(map(lambda e: e[0], train))
    test_ids = list(map(lambda e: e[0], test))

    list(map(lambda e: write_ids(ids=e[0], file_path=e[1]),
             ((train_ids, path.join(args.to_train_path, IMAGE_ID_FN)),
              (test_ids, path.join(args.to_test_path, IMAGE_ID_FN)))))

    print("Successfully copied {} images and wrote out id files".format(len(copied)))
    


    
