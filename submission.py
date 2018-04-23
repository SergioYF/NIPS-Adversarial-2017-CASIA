import os
import random
from time import time
from PIL import Image
import numpy as np
import tensorflow as tf

random.seed(0)
slim = tf.contrib.slim
work_dir = os.getcwd()

tf.flags.DEFINE_string(
    'input_dir', os.path.join(work_dir, 'development_set/images/'), 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', os.path.join(work_dir, 'result.csv'), 'Output directory with images.')

tf.flags.DEFINE_integer('image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer('image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer('batch_size', 32, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS
FLAGS.batch_size = 32
FLAGS.image_width = 299
FLAGS.image_height = 299


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

def ens3():
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

def predict():
    logits_ens3, sess_ens3, x_ens3 = ens3()
    logits_ens4, sess_ens4, x_ens4 = ens4()
    logits_res, sess_res, x_res = res()
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    images_generator = load_images(FLAGS.input_dir, batch_shape)
    total_start = time()
    times = []
    with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        for filenames, images in images_generator:
            start = time()
            l1 = sess_ens3.run(logits_ens3, feed_dict={x_ens3:images})
            l2 = sess_ens4.run(logits_ens4, feed_dict={x_ens4:images})
            l3 = sess_res.run(logits_res, feed_dict={x_res:images})
            l = l1 + l2 + l3
            preds = np.argmax(l, axis=1)
            for filename, label in zip(filenames, preds):
                out_file.write('{0},{1}\n'.format(filename, label))
            end = time()
            times.append(end-start)
    total_end = time()
    print(times)
    print("Times: {}".format(total_end-total_start))

predict()
