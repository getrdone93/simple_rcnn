import cv2
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import cm
from functools import reduce
import numpy as np
import time
import tkinter as tk
import argparse
from PIL import ImageTk, Image
from random import randint
import os.path as path
import os
import operator as op

TEMP_DIR = 'tmp'
ORIG_HISTO_FN = 'orig_histo.png'
EQ_HISTO_FN = 'eq_histo.png'

ORIG_HISTO_PATH = path.join(TEMP_DIR, ORIG_HISTO_FN)
EQ_HISTO_PATH = path.join(TEMP_DIR, EQ_HISTO_FN)

def show_images(ims):
    for im, i in zip(ims, range(0, len(ims))):
        cv2.imshow('image_' + str(i), im)

    while True:
        cv2.waitKey(1000)
        if input() == 'q':
            break
    cv2.destroyAllWindows()

def histogram(data):
    histo = {}
    for i in data:
        if i not in histo:
            histo[i] = 0
        histo[i] = histo[i] + 1
    return histo

def img_histo_added(img):
    h0, h1, h2 = map(lambda d: histogram(list(img[:, :, d].reshape(-1))), (0, 1, 2))
    all_keys = set(list(h0.keys()) + list(h1.keys()) + list(h2.keys()))
    return {k: reduce(lambda v1, v2: v1 + v2 , 
                map(lambda histo: histo.get(k, 0), 
                    (h0, h1, h2))) for k in all_keys}

def img_histo(img):
    return map(lambda d: histogram(list(img[:, :, d].reshape(-1))), (0, 1, 2))

def histogram_plots(img_histos):
    xs = range(256)
    num_bins = 256
    plt.hist(list(xs), num_bins, facecolor='blue', alpha=0.5)
    plt.show()

def equalization(mat, histo, ab_range, cd_range, c):
    new_mat = np.zeros(len(mat.reshape(-1))).reshape(mat.shape)
    ds = mat.shape
    start = time.time()
    for r in range(0, mat.shape[0]):
        for c in range(0, mat.shape[1]):
            if mat[r][c] in ab_range:
                ab = sum(map(lambda i: histo.get(i, mat[r][c]), ab_range))
                ap = sum(map(lambda i: histo.get(i, mat[r][c]), range(min(ab_range), mat[r][c] + 1)))\
                     if min(ab_range) <= mat[r][c] else 0
                dc = max(cd_range) - min(cd_range)
                new_mat[r][c] = int((dc / ab) * ap) + c
            else:
                new_mat[r][c] = mat[r][c]
    return new_mat

def equalize_image(image, ab_range, cd_range):
    new_img = image.copy()
    for h, d in zip(img_histo(image), (0, 1, 2)):
        print('EQUALIZER, is at image dimension ' + str(d))
        eq_mat = equalization(image[:, :, d], h, ab_range, cd_range, 0)
        new_img[:, :, d] = eq_mat

    return new_img

def equalize_image_name(image_path, image_name):
    return image_path.replace(image_name, 'histo-equalized-' + image_name)

def invoke_equalize(adv, bdv, cdv, ddv, image_path, image_ext, image_name):
    a, b, c, d = int(adv.get()), int(bdv.get()), int(cdv.get()), int(ddv.get())

    if abs(c - d) > 0 and abs(a - b) > 0:
        eq_img = equalize_image(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), 
                                   range(a, b), range(c, d))
        path = equalize_image_name(image_path, image_name)

        make_histo_from_image(eq_img, EQ_HISTO_PATH)
        cv2.imwrite(path, eq_img)
        print('invoke_equalize, wrote path ' + str(path))
    else:
        print("abs(a - b) and abs(c - d) must both be greater than 0")

def build_gui(image_path, image_ext, image_name, old_histo_path, eq_histo_path):
    start_y = 50
    image_gap = 40
    w_dim = 1000
    label_gap = 20
    slider_gap = 10
    button_gap = 60
    image_label_gap = 20

    window = tk.Tk() 
    window.title('Histogram equalization') 
    window.geometry('1500x1000')
    window.configure(background='grey')
    
    eq_path = equalize_image_name(image_path, image_name)

    #histos
    image = ImageTk.PhotoImage(Image.open(image_path))
    orig_histo = ImageTk.PhotoImage(Image.open(old_histo_path))
    eq_histo = ImageTk.PhotoImage(Image.open(eq_histo_path))

    #equalization
    if path.exists(eq_path):
        print('build_gui, reading in equalized image')
        eq_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.imread(eq_path), 'RGB'))
    else:
        print('build_gui, reading in original image')
        eq_image = image = ImageTk.PhotoImage(Image.open(image_path))

    #left side
    og  = 'Original'
    left_image_label = tk.Text(window, height=1, width=len(og))
    left_image_label.place(x=0, y=start_y - image_label_gap)
    left_image_label.insert(tk.END, og)
    panel1 = tk.Label(image=image)
    panel1.place(x=0, y=start_y)
    panel2 = tk.Label(image=orig_histo)
    panel2.place(x=0, y=start_y + image.height() + image_gap)

    #right side
    eq = 'Equalized'
    right_image_label = tk.Text(window, height=1, width=len(eq))
    right_image_label.place(x=w_dim - image.width(), y=start_y - image_label_gap)
    right_image_label.insert(tk.END, eq)
    panel3 = tk.Label(image=eq_image)
    panel3.place(x=w_dim - eq_image.width(), y=start_y)
    panel4 = tk.Label(image=eq_histo)
    panel4.place(x=w_dim - image.width(), y=start_y + image.height() + image_gap)

    #sliders right side
    adv, bdv, cdv, ddv = (tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar(), tk.DoubleVar())
    for sl, ig, sv, iv in zip(['a', 'b', 'c', 'd'], (1, 2, 3, 4), (adv, bdv, cdv, ddv), (0, 255, 0, 255)):    
        slider_x = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, variable=sv)
        slider_x.set(iv)
        sl_y = start_y + image.height() + image_gap + eq_image.height() + (image_gap * ig)
        sl_x = w_dim - image.width()
        slider_x.place(x=sl_x + label_gap, y=sl_y)

        text_a = tk.Text(window, height=1, width=2)
        text_a.place(x=sl_x, y=sl_y)
        text_a.insert(tk.END, sl)

    #equalize button
    eq_button = tk.Button(window, text='equalize', width=25, 
                          command=lambda: invoke_equalize(adv, bdv, cdv, ddv, 
                                                          image_path, image_ext, image_name))
    eq_button.place(x=w_dim - image.width(), y=start_y + image.height() + image_gap + eq_image.height()\
                    + (image_gap * 4) + button_gap)
    
    button = tk.Button(window, text='Stop', width=15, command=window.destroy) 
    button.pack()

    window.mainloop() 

def make_histo_from_image(image, fn):
    added_histo = img_histo_added(image)
    k, v = max(added_histo.items(), key=op.itemgetter(1))
    added_histo.pop(k)

    ys = added_histo.values()
    plt.figure(figsize=((image.shape[0] / 100) + 3, image.shape[1] / 100))
    plt.bar(added_histo.keys(), ys)
    plt.savefig(fn)
    plt.close()

def parse_args():
     parser = argparse.ArgumentParser(description='Histogram equalization')
     
     parser.add_argument('--image', required=False, help='Path to input image')
     parser.add_argument('--case', required=False, help='test case run')
     return parser.parse_args()

def make_dirs(dirs):
    created = []
    for d in dirs:
        if not path.exists(d):
            os.makedirs(d)
            created.append(d)

    return created

def run_gui(args):
    make_dirs([TEMP_DIR])
    image = cv2.imread(args.image)
    make_histo_from_image(image, ORIG_HISTO_PATH)

    if not path.exists(EQ_HISTO_PATH):
        make_histo_from_image(image, EQ_HISTO_PATH)

    base = path.basename(args.image)
    ext = base[base.index('.'):]
    name = base[:base.index('.')]
    build_gui(args.image, ext, name, ORIG_HISTO_PATH, EQ_HISTO_PATH)

def test_case(ab_range, cd_range, explanation):
    in_file = 'images/apple.jpeg'

    image = cv2.imread(in_file)
    make_histo_from_image(image, '/tmp/ih.png')
    ih = cv2.imread('/tmp/ih.png')

    eq_image = equalize_image(image, ab_range, cd_range)
    make_histo_from_image(eq_image, '/tmp/eh.png')
    eh = cv2.imread('/tmp/eh.png')

    print("Inputs: ab_range: [%d, %d], cd_range: [%d, %d], \n\nExplanation: %s" 
          % (min(ab_range), max(ab_range), min(cd_range), max(cd_range), explanation))
    cv2.imshow('original', image)
    cv2.imshow('original histogram', ih)
    cv2.imshow('equalized', eq_image)
    cv2.imshow('equalized histogram', eh)
    while True:
        cv2.waitKey(1000)
        if input() == 'q':
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parse_args()
    if args.case == '1':
        test_case(range(0, 2), range(0, 2), "This is not a good input because the ranges are not wide enough to produce\nA visible effect on the image. As a result, only the black border pixels are changed.")
    elif args.case == '2':
        test_case(range(70, 100),range(100, 130), "This case is interesting and a good input. The middle values are equalized and the shiny spot\non the apple is turned to pink and yellow. The shiny portion at the bottom turns yellow as well.")
    elif args.case == '3':
        test_case(range(200, 255), range(200, 255), "In this case the ranges are equalized and the apple experiences distortion, which is a bad input. The left\nside appears to turn green while the outer edges have some pixel values other than white.")
    elif args.image:
        run_gui(args)
    else:
        print("bad inputs")

    
