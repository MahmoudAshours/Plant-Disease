import shutil
import tkinter as tk
from tkinter.filedialog import askopenfilename
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
from PIL import Image, ImageTk
import tensorflow as tf

# Preparing the User interface window by :
# 1) Giving it a title Plant disease detection
# 2) Setting it do 500 x 510 px
# 3) Configuring background color of the window to blue
# 4) Adding label with text Click below to choose picture for testing disease....
# 5) Position the title widget in the parent widget in a grid.

window = tk.Tk()
window.title("Plant disease detection")
window.geometry("500x510")
window.configure(background="blue")
title = tk.Label(text="Click below to choose picture for testing disease....", background="lightgreen", fg="Brown",
                 font=("", 15))
title.grid()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)


# Preparing the user interface window when showing that the result is bacteria :
# 1) Destroying the window ( the first window of upload image)
# 2) Creating a new window user tKinter
# 3) Giving it a title Plant disease detection
# 4) Setting it do 500 x 510 px
# 5) Configuring background color of the window to blue
# 6) Adding the exit button above ( to exit the window)
# 7) Adding the remedies label which help the user to know what to do if found bacteria


def bact():
    window.destroy()
    window1 = tk.Tk()
    window1.title("Plant disease detection")
    window1.geometry("500x510")
    window1.configure(background="blue")

    def exit():
        window1.destroy()

    rem = "The remedies for Bacterial Spot are:\n\n "
    remedies = tk.Label(text=rem, background="lightgreen",
                        fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Discard or destroy any affected plants. \n  Do not compost them. \n " \
           " Rotate yoour tomato plants yearly to prevent re-infection next year. \n Use copper fungicites"
    remedies1 = tk.Label(text=rem1, background="lightgreen", fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


# Preparing the user interface window when showing that the result is virus :
# 1) Destroying the window ( the first window of upload image)
# 2) Creating a new window user tKinter
# 3) Giving it a title Plant disease detection
# 4) Setting it do 650x510 px
# 5) Configuring background color of the window to lightgreen
# 6) Adding the exit button above ( to exit the window)
# 7) Adding the remedies label which help the user to know what to do if found virus


def vir():
    window.destroy()
    window1 = tk.Tk()

    window1.title("Plant disease detection")

    window1.geometry("650x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()

    rem = "The remedies for Yellow leaf curl virus are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                        fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)
    rem1 = " Monitor the field, handpick diseased plants and bury them. \n  Use sticky yellow plastic traps. \n  Spray insecticides such as organophosphates, carbametes during the seedliing stage. \n Use copper fungicites"
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


# Preparing the user interface window when showing that the result is latebl :
# 1) Destroying the window ( the first window of upload image)
# 2) Creating a new window user tKinter
# 3) Giving it a title Plant disease detection
# 4) Setting it do 650x510 px
# 5) Configuring background color of the window to lightgreen
# 6) Adding the exit button above ( to exit the window)
# 7) Adding the remedies label which help the user to know what to do if found latebl

def latebl():
    window.destroy()
    window1 = tk.Tk()
    window1.title("Plant disease detection")
    window1.geometry("520x510")
    window1.configure(background="lightgreen")

    def exit():
        window1.destroy()

    rem = "The remedies for Late Blight are: "
    remedies = tk.Label(text=rem, background="lightgreen",
                        fg="Brown", font=("", 15))
    remedies.grid(column=0, row=7, padx=10, pady=10)

    rem1 = " Monitor the field, remove and destroy infected leaves. \n  Treat organically with copper spray. \n  Use chemical fungicides,the best of which for tomatoes is chlorothalonil."
    remedies1 = tk.Label(text=rem1, background="lightgreen",
                         fg="Black", font=("", 12))
    remedies1.grid(column=0, row=8, padx=10, pady=10)

    button = tk.Button(text="Exit", command=exit)
    button.grid(column=0, row=9, padx=20, pady=20)

    window1.mainloop()


# Analysis function which loads the pretrained model in CNN.py , formatting it and verifying data
# Then analysing the incoming external picture & labeling it
# Whether the incoming picture is healthy , bacterial , viral or late blight
# Each of the previous conditions has its own window where it selects the remedy and show it to the user using tKinter
def analysis():
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'healthyvsunhealthy-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    # verify_data = np.load('verify_data.npy')

    ops.reset_default_graph()

    conv_net = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    conv_net = conv_2d(conv_net, 32, 3, activation='relu')
    conv_net = max_pool_2d(conv_net, 3)

    conv_net = conv_2d(conv_net, 64, 3, activation='relu')
    conv_net = max_pool_2d(conv_net, 3)

    conv_net = conv_2d(conv_net, 128, 3, activation='relu')
    conv_net = max_pool_2d(conv_net, 3)

    conv_net = conv_2d(conv_net, 32, 3, activation='relu')
    conv_net = max_pool_2d(conv_net, 3)

    conv_net = conv_2d(conv_net, 64, 3, activation='relu')
    conv_net = max_pool_2d(conv_net, 3)

    conv_net = fully_connected(conv_net, 1024, activation='relu')
    conv_net = dropout(conv_net, 0.8)

    conv_net = fully_connected(conv_net, 4, activation='softmax')
    conv_net = regression(conv_net, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(conv_net, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):

        model.load(MODEL_NAME)
        print('model loaded!')

    for num, data in enumerate(verify_data):

        img_data = data[0]

        data = img_data.reshape((IMG_SIZE, IMG_SIZE, 3))
        model_out = model.predict([data])[0]

        print(np.argmax(model_out))
        str_label = ""
        if np.argmax(model_out) == 0:
            str_label = 'healthy'
        elif np.argmax(model_out) == 1:
            str_label = 'bacterial'
        elif np.argmax(model_out) == 2:
            str_label = 'viral'
        elif np.argmax(model_out) == 3:
            str_label = 'lateblight'

        if str_label == 'healthy':
            status = "HEALTHY"
        else:
            status = "UNHEALTHY"

        message = tk.Label(text='Status: ' + status, background="lightgreen", fg="Brown", font=("", 15))
        message.grid(column=0, row=3, padx=10, pady=10)
        if str_label == 'bacterial':
            disease_name = "Bacterial Spot "
            disease = tk.Label(text='Disease Name: ' + disease_name, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=bact)
            button3.grid(column=0, row=6, padx=10, pady=10)
        elif str_label == 'viral':
            disease_name = "Yellow leaf curl virus "
            disease = tk.Label(text='Disease Name: ' + disease_name, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=vir)
            button3.grid(column=0, row=6, padx=10, pady=10)
        elif str_label == 'lateblight':
            disease_name = "Late Blight"
            disease = tk.Label(text='Disease Name: ' + disease_name, background="lightgreen",
                               fg="Black", font=("", 15))
            disease.grid(column=0, row=4, padx=10, pady=10)
            r = tk.Label(text='Click below for remedies...', background="lightgreen", fg="Brown", font=("", 15))
            r.grid(column=0, row=5, padx=10, pady=10)
            button3 = tk.Button(text="Remedies", command=latebl)
            button3.grid(column=0, row=6, padx=10, pady=10)
        else:
            r = tk.Label(text='Plant is healthy', background="lightgreen", fg="Black",
                         font=("", 15))
            r.grid(column=0, row=4, padx=10, pady=10)
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=20, pady=20)


# Opening the external photo , reading the image & rendering it then analysing it
def open_photo():
    dir_path = "testpicture"
    file_list = os.listdir(dir_path)
    for file_name in file_list:
        os.remove(dir_path + "/" + file_name)
    # Path of destination you want to test
    # you can change it according to the image location you have
    file_name = askopenfilename(initialdir='/home/ashour/PycharmProjects/StemApp/train/train/',
                                title='Select image for analysis ',
                                filetypes=[('image files', '.jpg')])
    # Path of destination you want to make the testing take place.
    dst = "/home/ashour/PycharmProjects/StemApp/testpicture"

    shutil.copy(file_name, dst)
    load = Image.open(file_name)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady=10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady=10)


button1 = tk.Button(text="Get Photo", command=open_photo)
button1.grid(column=0, row=1, padx=10, pady=10)

window.mainloop()
