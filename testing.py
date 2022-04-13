#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of testing code of this paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)


import os
import re
import numpy as np
import skimage
from skimage import io
import tensorflow.compat.v1 as tf
import training as Network
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # select GPU device

tf.reset_default_graph()

model_path = './model/'
pre_trained_model_path = './model/trained/model'

img_path = './TestData/input/'  # the path of testing images
results_path = './TestData/results/'  # the path of de-rained images


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    rainy = tf.cast(image_decoded, tf.float32) / 255.0
    return rainy


def derainAll():
    imgName = os.listdir(img_path)
    num_img = len(imgName)

    whole_path = []
    for i in range(num_img):
        whole_path.append(img_path + imgName[i])

    filename_tensor = tf.convert_to_tensor(whole_path, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
    dataset = dataset.map(_parse_function)
    dataset = dataset.prefetch(buffer_size=10)
    dataset = dataset.batch(batch_size=1).repeat()
    iterator = dataset.make_one_shot_iterator()

    rain = iterator.get_next()

    output = Network.inference(rain, is_training=False)
    output = tf.clip_by_value(output, 0., 1.)
    output = output[0, :, :, :]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'):
            """
            if tf.train.get_checkpoint_state(model_path):
                ckpt = tf.train.latest_checkpoint(model_path)  # try your own model
                saver.restore(sess, ckpt)
                print("Loading model")
            """
            saver.restore(sess, pre_trained_model_path)  # try a pre-trained model
            print("Loading pre-trained model")

            for i in range(num_img):
                derained, ori = sess.run([output, rain])
                derained = np.uint8(derained * 255.)
                index = imgName[i].rfind('.')
                name = imgName[i][:index]
                skimage.io.imsave(results_path + name + '_derained.png', derained)
                print('%d / %d images processed' % (i + 1, num_img))

        print('Images Derained Successfully')
    sess.close()


def derain(filepath):
    if filepath == "":
        print("Please Choose a Rainy Image.")
        return
    tf.reset_default_graph()
    filepath = [filepath]
    filename_tensor = tf.convert_to_tensor(filepath, dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor))
    dataset = dataset.map(_parse_function)
    dataset = dataset.prefetch(buffer_size=10)
    dataset = dataset.batch(batch_size=1).repeat()
    iterator = dataset.make_one_shot_iterator()

    rain = iterator.get_next()

    output = Network.inference(rain, is_training=False)
    output = tf.clip_by_value(output, 0., 1.)
    output = output[0, :, :, :]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        with tf.device('/gpu:0'):
            saver.restore(sess, pre_trained_model_path)  # try a pre-trained model
            print("Loading pre-trained model")

            derained, ori = sess.run([output, rain])
            derained = np.uint8(derained * 255.)
            name = re.search("^.*\/(.*)\.(png|jpg|jpeg)$", filepath[0]).group(1)
            skimage.io.imsave(results_path + name + '_derained.png', derained)

        print('Image Derained Successfully')
    sess.close()
    return results_path + name + '_derained.png'

if __name__ == '__main__':
    derain("./TestData/input/1.jpg")
    derain("./TestData/input/2.jpg")
    #derainAll()

