import os
from os.path import dirname, abspath

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from skimage.measure import compare_ssim as ssim
import numpy as np
import math
import cv2

from train_utils import parse_params
from Utils import prep_save, clean_from_hazy
from Models import Models
from our_dehaze import DarkChannel, AtmLight, TransmissionEstimate, TransmissionRefine, Recover

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parent_dir = dirname(dirname(abspath(__file__)))
results_folder = parent_dir + '/Results/' + 'dehaze'
checkpoints_path = results_folder + '/checkpoints/'

image_folder = parent_dir + '/Datasets/hazy'
orig_image_folder = parent_dir + '/Datasets/orig_images'

config_file_path = results_folder + '/params.ini'
args = parse_params(config_file_path)
args.image_dim = [128, 128, 3]
checkpoint = 30

# create TF placeholders and build network
tf.compat.v1.reset_default_graph()
in_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, args.image_dim[2]], name="in_placeholder")
out_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1], name='out_placeholder')
phase = tf.compat.v1.placeholder(tf.bool, name='phase')
net_class = Models(args)
net_class.build_model(in_placeholder, phase)

# open TF session and load saved model
sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.Saver()
sess.run(tf.compat.v1.global_variables_initializer())
ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoints_path)
if ckpt and ckpt.model_checkpoint_path:
    ckpt_path = checkpoints_path + 'my_model-' + str(checkpoint)
    saver.restore(sess, ckpt_path)

# create output folder
output_folder = results_folder + '/Qual_output/paper'
our_output_folder = results_folder + '/Qual_output/our'

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
if not os.path.exists(our_output_folder):
    os.makedirs(our_output_folder, exist_ok=True)

# iterate over images in '/Datasets/Dehaze/Qual'
image_list = sorted(os.listdir(image_folder))


def get_PSNR_SSIM(orig_im, J):
    orig_im = orig_im.astype(np.float64)
    MSE = np.sqrt(np.mean(np.power(orig_im - J, 2)))
    PSNR = 20 * math.log10(1.0 / MSE)
    SSIM = ssim(orig_im, J, multichannel=True, gaussian_weights=True, sigma=1.5,
                use_sample_covariance=False)
    return PSNR, SSIM


def our_dehaze(I, src, file_name):
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    te = TransmissionEstimate(I, A, 15)
    t = TransmissionRefine(src, te)
    J = Recover(I, t, A, 0.1)
    cv2.imwrite(our_output_folder + "/" + file_name, J * 255)
    cv2.waitKey()


def main():
    for i in range(len(image_list)):
        file_name = image_list[i]
        image = np.array(Image.open(image_folder + '/' + file_name))
        print('image (%d/%d)' % (i + 1, len(image_list)))
        im_for_tf = np.expand_dims(image / 255.0, axis=0)
        t_network = np.squeeze(sess.run([net_class.network_out], feed_dict={in_placeholder: im_for_tf, phase: False}))
        J_network = clean_from_hazy(image, t_network, args.dehaze_win, args.dehaze_omega, args.dehaze_thresh)
        J_network = (J_network - np.min(J_network)) / (np.max(J_network) - np.min(J_network))
        Image.fromarray(prep_save(J_network)).save(output_folder + "/" + file_name)

        src = cv2.imread(output_folder + "/" + file_name)
        I = src.astype('float') / 255
        our_dehaze(I, src, file_name)
        our_dehazed_image = np.array(Image.open(f'{our_output_folder + "/" + file_name}')) / 255.0

        orig_image = np.array(Image.open(orig_image_folder + '/' + file_name)) / 255
        J_PSNR = 20 * math.log10(1.0 / np.sqrt(np.mean(np.power(orig_image - J_network, 2))))
        our_J_PSNR = 20 * math.log10(1.0 / np.sqrt(np.mean(np.power(orig_image - our_dehazed_image, 2))))

        # print(f' J_PSNR = {J_PSNR}', end='\t')
        # print(f' our_J_PSNR = {our_J_PSNR}')

        title_file = file_name.split(".")[0]
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 4, 1)
        plt.axis("off")
        pil_img = Image.open(f'{orig_image_folder + "/" + file_name}', 'r')
        plt.imshow(pil_img)
        plt.title(f"Original {title_file}")

        plt.subplot(1, 4, 2)
        plt.axis("off")
        pil_img = Image.open(f'{image_folder + "/" + file_name}', 'r')
        plt.imshow(pil_img)
        plt.title(f"Hazy {title_file}")

        plt.subplot(1, 4, 3)
        plt.axis("off")
        pil_img = Image.open(f'{output_folder + "/" + file_name}', 'r')
        plt.imshow(pil_img)
        plt.title(f"Paper Dehazed {title_file}" + "\nPSNR = " + str(J_PSNR))

        plt.subplot(1, 4, 4)
        plt.axis("off")
        pil_img = Image.open(f'{our_output_folder + "/" + file_name}', 'r')
        plt.imshow(pil_img)
        plt.title(f"Our Dehazed {title_file}" + "\nPSNR = " + str(our_J_PSNR))

        plt.show()
    sess.close()


if __name__ == '__main__':
    main()
