import threading
from time import time

import tensorflow as tf
from models.load_data import *
from tensorflow.contrib.slim.nets import inception
from adv_imagenet_models import inception_resnet_v2
from attacks import fgsm

summary_path = '/home/yangfan/ensemble_training'

ALPHA = 0.5
EPSILON = 0.3
learning_rate = 0.001
display_step = 100
test_step = 100
epochs = 90
save_path = '/home/yangfan/ensemble_training/ens4-inception.ckpt'

train_size = imagenet_size()
num_batches = int(float(train_size) / BATCH_SIZE)

x = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1000])
lr = tf.placeholder(tf.float32)
# keep_prob = tf.placeholder(tf.float32)

with tf.device('/cpu:0'):
    q = tf.FIFOQueue(BATCH_SIZE, [tf.float32, tf.float32], shapes=[[299, 299, 3], [1000]])
    enqueue_op = q.enqueue_many([x, y])
    x_b, y_b = q.dequeue_many(BATCH_SIZE / 4)

logits, end_points = inception.inception_v3(x_b, is_training=True)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits)

with tf.name_scope('l2_loss'):
    l2_loss = tf.reduce_sum(5e-4 * tf.stack([tf.nn.l2_loss(v) for v in tf.get_collection('weights')]))
    tf.summary.scalar('l2_loss', l2_loss)

with tf.name_scope('loss'):
    loss = cross_entropy + l2_loss
    tf.summary.scalar('loss', loss)

with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

global_step = tf.Variable(0, trainable=False)
epoch = tf.div(global_step, num_batches)

with tf.name_scope('optimizer'):
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)

merged = tf.summary.merge_all()
saver = tf.train.Saver()
coord = tf.train.Coordinator()
init = tf.global_variables_initializer()

def get_adversarial(sess_trained, x, probs, image):
    x_adv = fgsm(x=x, predictions=probs, eps=0.3, clip_max=1.0, clip_min=-1.0)
    img_adv = sess_trained.run(x_adv, feed_dict={x:image})
    return img_adv

with tf.Session(config=tf.ConfigProto) as sess_train:
    sess_train.run(init)
    def enqueue_batches():
        while not coord.should_stop():
            ori_imgs, labels = read_batch(batch_size=BATCH_SIZE/2)
            adv_imgs = get_adversarial(sess_trained=sess_train, x=x_b, probs=end_points['Predictions'], image=ori_imgs)
            imgs = np.vstack([ori_imgs, adv_imgs])
            lbs = np.vstack([ori_imgs, adv_imgs])
            sess_train.run(enqueue_op, feed_dict={x:imgs, y:lbs})

    num_threads = 4
    for i in range(num_threads):
        t = threading.Thread(target=enqueue_batches)
        t.setDaemon(True)
        t.start()

    train_writer = tf.summary.FileWriter(os.path.join(summary_path, 'train'), sess_train.graph)
    start_time = time()

    for e in range(sess_train.run(epoch), epochs):
        for i in range(num_batches):

            _, step = sess_train.run([optimizer, global_step], feed_dict={lr:learning_rate})

            if step == 170000 or step == 350000:
                learning_rate /= 10

            if step % display_step == 0:
                c, a = sess_train.run([loss, accuracy], feed_dict={lr:learning_rate})
                print('Epoch: {:03d} Step/Batch: {:09d} --- Loss: {:.7f} Training accuracy: {:.4f}'.format(e, step, c, a))

            if step % test_step == 0:
                pass

    end_time = time()
    print('Elapsed time: {}').format(format_time(end_time - start_time))
    saved = saver.save(sess_train, save_path)
    print('Variables saved in file: %s' % saved)
    coord.request_stop()

