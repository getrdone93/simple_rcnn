import argparse
import tkinter as tk
from PIL import ImageTk, Image
import os.path as path
import cv2

def binary_image(image, th):
    new_image = image.copy()
    for d in (0, 1, 2):
        mat = new_image[:, :, d]
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                mat[r][c] = 255 if mat[r][c] > th else 0

    return new_image

def binary_image_name(image_path, image_name):
    return image_path.replace(image_name, 'binary-' + image_name)

def binarize_image_hook(image_path, image_name, val):
    bv = val.get()
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    bi = binary_image(image, bv)
    b_path = binary_image_name(image_path, image_name)
    cv2.imwrite(b_path, bi)

    print('wrote binary image to path ' + str(b_path))

def build_gui(image_path, image_name):
    w_dim = 800
    start_y = 50
    slider_gap = 50

    window = tk.Tk() 
    window.title('Image binarization') 
    window.geometry('800x800')
    window.configure(background='grey')

    #read images
    image = ImageTk.PhotoImage(Image.open(image_path))
    b_path = binary_image_name(image_path, image_name)
    if path.exists(b_path):
        print('build_gui, reading in binary image')
        binary_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.imread(b_path), 'RGB'))
    else:
        print('build_gui, reading original image')
        binary_image = image

    #left image
    og  = 'Original'
    left_image_label = tk.Text(window, height=1, width=len(og))
    left_image_label.place(x=0, y=0)
    left_image_label.insert(tk.END, og)
    panel1 = tk.Label(image=image)
    panel1.place(x=0, y=start_y)

    #right image
    bin_text = 'Binary'
    right_image_label = tk.Text(window, height=1, width=len(bin_text))
    right_image_label.place(x=w_dim - image.width(), y=0)
    right_image_label.insert(tk.END, bin_text)
    panel2 = tk.Label(image=binary_image)
    panel2.place(x=w_dim - binary_image.width(), y=start_y)

    #slider
    sv = tk.IntVar()
    slider = tk.Scale(window, from_=0.0, to=255.0, orient=tk.HORIZONTAL, variable=sv)
    slider.set(0)
    slider.place(x=slider_gap, y=start_y + image.height() + slider_gap)

    #binarize button
    exec_button = tk.Button(window, text='Binarize', width=15, 
                            command=lambda: binarize_image_hook(image_path, image_name, sv))
    exec_button.place(x=slider_gap, y=start_y + image.height() + slider_gap * 2)

    button = tk.Button(window, text='Stop', width=15, command=window.destroy) 
    button.pack()

    window.mainloop() 

def parse_args():
     parser = argparse.ArgumentParser(description='Image binarization')
     
     parser.add_argument('--image', required=False, help='Path to input image')
     parser.add_argument('--case', required=False, help='test case run')
     return parser.parse_args()

def run_gui(args):
    base = path.basename(args.image)
    name = base[:base.index('.')]
    build_gui(args.image, name)

def test_case_1():
    bt = 0
    in_file = 'images/apple.jpeg'
    ri = cv2.imread(in_file)
    bi = binary_image(ri, 0)


    print("Input file: %s, binary threshold: %d, \n\nExplanation: %s" 
          % (in_file, bt, "This is a bad input because the threshold was set too low. Therefore,\nmost of the pixels were set to white"))
    cv2.imshow('original', ri)
    cv2.imshow('binarized', bi)
    while True:
        cv2.waitKey(1000)
        if input() == 'q':
            break
    cv2.destroyAllWindows()

def test_case_2():
    bt = 255
    in_file = 'images/apple.jpeg'
    ri = cv2.imread(in_file)
    bi = binary_image(ri, bt)


    print("Input file: %s, binary threshold: %d, \n\nExplanation: %s" 
          % (in_file, bt, "This is a bad input because the threshold was set too high. Therefore,\nmost of the pixels were set to black"))
    cv2.imshow('original', ri)
    cv2.imshow('binarized', bi)
    while True:
        cv2.waitKey(1000)
        if input() == 'q':
            break
    cv2.destroyAllWindows()

def test_case_3():
    bt = 80
    in_file = 'images/apple.jpeg'
    ri = cv2.imread(in_file)
    bi = binary_image(ri, bt)


    print("Input file: %s, binary threshold: %d, \n\nExplanation: %s" 
          % (in_file, bt, "This is a good input because the threshold was set at a level that allowed\nfor pixels to be both black and white, thus producing the desired effect"))
    cv2.imshow('original', ri)
    cv2.imshow('binarized', bi)
    while True:
        cv2.waitKey(1000)
        if input() == 'q':
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    if args.case == '1':
        test_case_1()
    elif args.case == '2':
        test_case_2()
    elif args.case == '3':
        test_case_3()
    elif args.image:
        run_gui(args)
    else:
        print("bad inputs")
    
