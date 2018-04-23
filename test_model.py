import tensorflow as tf
from models.load_data import *
from tensorflow.contrib.slim.nets import inception
from adv_imagenet_models import inception_resnet_v2, inception_resnet_v2_arg_scope

def config_model(x=None, model_name=None, new=True):
    slim = tf.contrib.slim
    if model_name == 'inception_ens3':
        if new:
            assert x is None
            sess = tf.Session()
            dir = '/home/yangfan/ens_ckpts/ens3_adv_inception_v3_2017_08_18'
            saver = tf.train.import_meta_graph(os.path.join(dir, 'ens3_adv_inception_v3.ckpt.meta'))
            saver.restore(sess, os.path.join(dir, 'ens3_adv_inception_v3.ckpt'))
            graph = tf.get_default_graph()
            logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
            x = graph._nodes_by_id[1].outputs[0]
            return logits, sess, x
        else:
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                logits, end_points = inception.inception_v3(x, is_training=False)
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            dir = '/home/yangfan/ens_ckpts/ens3_adv_inception_v3_2017_08_18'
            saver = tf.train.import_meta_graph(os.path.join(dir, 'ens3_adv_inception_v3.ckpt.meta'))
            saver.restore(sess, os.path.join(dir, 'ens3_adv_inception_v3.ckpt'))
            y = tf.placeholder(dtype=tf.float32, shape=[None, 1000])
            return logits, sess, y
    elif model_name == 'inception_ens4':
        if new:
            assert x is None
            sess = tf.Session()
            dir = '/home/yangfan/ens_ckpts/ens4_adv_inception_v3_2017_08_18'
            saver = tf.train.import_meta_graph(os.path.join(dir, 'ens4_adv_inception_v3.ckpt.meta'))
            saver.restore(sess, os.path.join(dir, 'ens4_adv_inception_v3.ckpt'))
            graph = tf.get_default_graph()
            logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
            x = graph._nodes_by_id[1].outputs[0]
            return logits, sess, x
        else:
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                logits, end_points = inception.inception_v3(x, is_training=False)
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            dir = '/home/yangfan/ens_ckpts/ens4_adv_inception_v3_2017_08_18'
            saver = tf.train.import_meta_graph(os.path.join(dir, 'ens4_adv_inception_v3.ckpt.meta'))
            saver.restore(sess, os.path.join(dir, 'ens4_adv_inception_v3.ckpt'))
            y = tf.placeholder(dtype=tf.float32, shape=[None, 1000])
            return logits, sess, y
    elif model_name == 'inception_resnet_v2':
        if new:
            assert x is None
            sess = tf.Session()
            dir = '/home/yangfan/ens_ckpts/ens_adv_inception_resnet_v2_2017_08_18'
            saver = tf.train.import_meta_graph(os.path.join(dir, 'ens_adv_inception_resnet_v2.ckpt.meta'))
            saver.restore(sess, os.path.join(dir, 'ens_adv_inception_resnet_v2.ckpt'))
            graph = tf.get_default_graph()
            logits = graph.get_tensor_by_name('InceptionResnetV2/Logits/Predictions:0').op.inputs[0]
            x = graph._nodes_by_id[1].outputs[0]
            return logits, sess, x
        else:
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(x, is_training=False, num_classes=1000)
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            dir = '/home/yangfan/ens_ckpts/ens_adv_inception_resnet_v2_2017_08_18'
            saver = tf.train.import_meta_graph(os.path.join(dir, 'ens_adv_inception_resnet_v2.ckpt.meta'))
            saver.restore(sess, os.path.join(dir, 'ens_adv_inception_resnet_v2.ckpt'))
            y = tf.placeholder(dtype=tf.float32, shape=[None, 1000])
            return logits, sess, y
    elif model_name == 'original':
        if new:
            assert x is None
            sess = tf.Session()
            dir = '/home/yangfan/ens_ckpts/original'
            saver = tf.train.import_meta_graph(os.path.join(dir, 'inception_v3.ckpt.meta'))
            saver.restore(sess, os.path.join(dir, 'inception_v3.ckpt'))
            graph = tf.get_default_graph()
            logits = graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
            x = graph._nodes_by_id[1].outputs[0]
            return logits, sess, x
        else:
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                logits, end_points = inception.inception_v3(x, is_training=False, num_classes=1001)
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)
            dir = '/home/yangfan/ens_ckpts/original'
            saver = tf.train.import_meta_graph(os.path.join(dir, 'inception_v3.ckpt.meta'))
            saver.restore(sess, os.path.join(dir, 'inception_v3.ckpt'))
            y = tf.placeholder(dtype=tf.float32, shape=[None, 1001])
            return logits, sess, y
    else:
        raise NotImplementedError

def test_model():
    batch_size=32
    model_type = 'inception_resnet_v2'
    logits, sess, x = config_model(model_name=model_type)
    pred = tf.argmax(logits, 1)
    imgs, label_list = read_local_images(batch_size=batch_size)
    pred_list = sess.run(pred, feed_dict={x: imgs})
    ans = []
    for i in range(batch_size):
        ans.append((pred_list[i], label_list[i]))
    print(ans)

test_model()

# sess = tf.Session()
# dir = '/home/yangfan/ens_ckpts/ens_adv_inception_resnet_v2_2017_08_18'
# saver = tf.train.import_meta_graph(os.path.join(dir, 'ens_adv_inception_resnet_v2.ckpt.meta'))
# saver.restore(sess, os.path.join(dir, 'ens_adv_inception_resnet_v2.ckpt'))
# graph = tf.get_default_graph()
# logits = graph.get_tensor_by_name('InceptionResnetV2/Logits/Predictions:0').op.inputs[0]
# x = graph._nodes_by_id[1].outputs[0]
# from IPython import embed
# embed()