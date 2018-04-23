import os
import random
from time import time
from PIL import Image
import numpy as np
import tensorflow as tf
from adv_imagenet_models import inception_resnet_v2
from tensorflow.contrib.slim.nets import inception

random.seed(0)
slim = tf.contrib.slim
work_dir = os.getcwd()

tf.flags.DEFINE_string(
    'input_dir', os.path.join(work_dir, 'development_set/images/'), 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', os.path.join(work_dir, 'output_dir/'), 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 32, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS
FLAGS.batch_size = 32
FLAGS.image_width = 299
FLAGS.image_height = 299
eps = 2.0 * FLAGS.max_epsilon / 255.0


def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    if img.size[0] == 299 and img.size[1] == 299:
        img_cropped = img
    elif img.size[0] < img.size[1]:
        h = int(float(342 * img.size[1]) / img.size[0])
        img = img.resize((342, h), Image.ANTIALIAS)
        x = random.randint(0, img.size[0] - 299)
        y = random.randint(0, img.size[1] - 299)
        img_cropped = img.crop((x, y, x + 299, y + 299))
    else:
        w = int(float(342 * img.size[0]) / img.size[1])
        img = img.resize((w, 342), Image.ANTIALIAS)
        x = random.randint(0, img.size[0] - 299)
        y = random.randint(0, img.size[1] - 299)
        img_cropped = img.crop((x, y, x + 299, y + 299))
    cropped_im_array = np.array(img_cropped, dtype=np.float32)
    cropped_im_array = cropped_im_array / 255.0
    cropped_im_array = cropped_im_array * 2.0 - 1.0
    return cropped_im_array

def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        image = preprocess_image(filepath)
        images[idx, :, :, :] = image
        filenames.append(os.path.basename(filepath))
        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
            Image.fromarray(img).save(f, format='PNG')

def ens3_old():
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        dir = os.path.join(work_dir, 'ens_ckpts/ens3_adv_inception_v3')
        saver = tf.train.import_meta_graph(os.path.join(dir, 'ens3_adv_inception_v3.ckpt.meta'))
        saver.restore(sess, os.path.join(dir, 'ens3_adv_inception_v3.ckpt'))
        graph = sess.graph
        logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
        x = graph._nodes_by_id[1].outputs[0]
        return logits, sess, x

def ens3():
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        dir = os.path.join(work_dir, 'ens_ckpts/ens3_adv_inception_v3')
        x = tf.placeholder(dtype=tf.float32, shape=[32, 299, 299, 3])
        logits = inception.inception_v3(x, num_classes=1001, is_training=False)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(dir, 'ens3_adv_inception_v3.ckpt'))
        return logits, sess, x

def ens4():
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        dir = os.path.join(work_dir, 'ens_ckpts/ens4_adv_inception_v3')
        saver = tf.train.import_meta_graph(os.path.join(dir, 'ens4_adv_inception_v3.ckpt.meta'))
        saver.restore(sess, os.path.join(dir, 'ens4_adv_inception_v3.ckpt'))
        # graph = tf.get_default_graph()
        graph = sess.graph
        logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
        x = graph._nodes_by_id[1].outputs[0]
        return logits, sess, x

def res():
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        dir = os.path.join(work_dir, 'ens_ckpts/ens_adv_inception_resnet_v2')
        saver = tf.train.import_meta_graph(os.path.join(dir, 'ens_adv_inception_resnet_v2.ckpt.meta'))
        saver.restore(sess, os.path.join(dir, 'ens_adv_inception_resnet_v2.ckpt'))
        graph = sess.graph
        logits = graph.get_tensor_by_name('InceptionResnetV2/Logits/Predictions:0').op.inputs[0]
        x = graph._nodes_by_id[1].outputs[0]
        return logits, sess, x

def model_loss(y,logits):
    out = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
    return out

def input_jump(imgs):
    assert isinstance(imgs, np.ndarray)
    assert len(imgs.shape) == 4
    shape = imgs.shape
    noise = (np.random.random_sample(shape) + np.ones(shape)) * eps * 0.05
    perturbed_img = noise + imgs
    return perturbed_img

def find_adv():
    logits_ens3, sess_ens3, x_ens3 = ens3()
    logits_ens4, sess_ens4, x_ens4 = ens4()
    logits_res, sess_res, x_res = res()
    y_ens3 = tf.one_hot(tf.argmax(logits_ens3, 1), 1001, axis=-1)
    y_ens4 = tf.one_hot(tf.argmax(logits_ens4, 1), 1001, axis=-1)
    y_res = tf.one_hot(tf.argmax(logits_res, 1), 1001, axis=-1)
    # from IPython import embed
    # embed()
    loss_ens3 = model_loss(y_ens3, logits_ens3)
    loss_ens4 = model_loss(y_ens4, logits_ens4)
    loss_res = model_loss(y_res, logits_res)
    gradient_ens3 = tf.gradients(loss_ens3, x_ens3)[0]
    gradient_ens4 = tf.gradients(loss_ens4, x_ens4)[0]
    gradient_res = tf.gradients(loss_res, x_res)[0]

    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    images_generator = load_images(FLAGS.input_dir, batch_shape)
    total_start = time()
    times = []
    for filenames, images in images_generator:
        start = time()
        pert_img = input_jump(images)
        g1 = sess_ens3.run(gradient_ens3, feed_dict={x_ens3:pert_img})
        g2 = sess_ens4.run(gradient_ens4, feed_dict={x_ens4:pert_img})
        g3 = sess_res.run(gradient_res, feed_dict={x_res: pert_img})
        g = g1 + g2 + g3
        g_sign = np.sign(g) * eps
        print("g1[0] {}".format(g1[0]))
        # print(g2[0])
        # print(g3[0])
        adv_images = images + g_sign
        # save_images(adv_images, filenames, FLAGS.output_dir)
        end = time()
        print("Time: {} seconds".format(end-start))
        times.append(end-start)
    total_end = time()
    print(times)
    print("Times: {}".format(total_end-total_start))

find_adv()