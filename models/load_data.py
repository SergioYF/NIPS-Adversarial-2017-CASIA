import os
import random
from time import time
from scipy.io import loadmat
from scipy import misc
from PIL import Image
import numpy as np

random.seed(0)


TRAIN_FOLDER = '/hd1/imagenet-data/train'
VALIDATION_FOLDER = '/hd1/imagenet-data/validation'
META_PATH = '/home/yangfan/meta.mat'
BATCH_SIZE = 128
TIME_START = time()

def load_imagenet_meta(meta_path=META_PATH):
    metadata = loadmat(meta_path, struct_as_record=False)
    synsets = np.squeeze(metadata['synsets'])
    wnids = np.squeeze(np.array([s.WNID for s in synsets]))
    words = np.squeeze(np.array([s.words for s in synsets]))
    return wnids, words


def imagenet_size(im_source=TRAIN_FOLDER):
    n = 0
    for d in os.listdir(im_source):
        for f in os.listdir(os.path.join(im_source, d)):
            n += 1
    return n

def format_time(time):
    m, s = divmod(time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return ('{:02d}d {:02d}h {:02d}m {:02d}s').format(int(d), int(h), int(m), int(s))

def onehot(index):
    onehot = np.zeros(1000)
    onehot[index] = 1.0
    return onehot

def preprocess_image(image_path, new=True):
    if new:
        return preprocess_image_new(image_path)
    else:
        return preprocess_image_old(image_path)

def preprocess_image_old(image_path):
    IMAGENET_MEAN = [123.68, 116.779, 103.939] # rgb format
    img = Image.open(image_path).convert('RGB')
    if img.size[0] < img.size[1]:
        h = int(float(342 * img.size[1]) / img.size[0])
        img = img.resize((342, h), Image.ANTIALIAS)
    else:
        w = int(float(342 * img.size[0]) / img.size[1])
        img = img.resize((w, 342), Image.ANTIALIAS)
    x = random.randint(0, img.size[0] - 299)
    y = random.randint(0, img.size[1] - 299)
    img_cropped = img.crop((x, y, x + 299, y + 299))
    cropped_im_array = np.array(img_cropped, dtype=np.float32)
    for i in range(3):
        cropped_im_array[:,:,i] -= IMAGENET_MEAN[i]
        # cropped_im_array[:, :, i] /= 128.0
    return cropped_im_array

def preprocess_image_new(image_path):
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



def read_image(images_folder):
    image_path = os.path.join(images_folder, random.choice(os.listdir(images_folder)))
    im_array = preprocess_image(image_path)
    return im_array

def read_batch(batch_size=BATCH_SIZE, images_source=TRAIN_FOLDER, wnid_labels=None):
    """ It returns a batch of single images (no data-augmentation)

    	ILSVRC 2012 training set folder should be srtuctured like this:
    	ILSVRC2012_img_train
    		|_n01440764
    		|_n01443537
    		|_n01484850
    		|_n01491361
    		|_ ...

    	Args:
    		batch_size: need explanation? :)
    		images_sources: path to ILSVRC 2012 training set folder
    		wnid_labels: list of ImageNet wnid lexicographically ordered

    	Returns:
    		batch_images: a tensor (numpy array of images) of shape [batch_size, width, height, channels]
    		batch_labels: a tensor (numpy array of onehot vectors) of shape [batch_size, 1000]
    """
    if wnid_labels is None:
        wnid_labels, _ = load_imagenet_meta()
    batch_images = []
    batch_labels = []
    for i in range(batch_size):
        class_index = random.randint(0, 999)
        folder = wnid_labels[class_index]
        # print("class_index = {0}, folder = {1}".format(class_index, folder))
        batch_images.append(read_image(os.path.join(images_source, folder)))
        batch_labels.append(onehot(class_index))
    np.vstack(batch_images)
    np.vstack(batch_labels)
    return batch_images, batch_labels


    # def test():
    #     n = 0
    #     wnid_labels, _ = load_imagenet_meta(meta_path=META_PATH)
    #     train_size = imagenet_size(TRAIN_FOLDER)
    #     print("Train size = {0} Time : {1}".format(train_size, format_time(time()-TIME_START)))
    #     valid_size = imagenet_size(VALIDATION_FOLDER)
    #     print("Valid size = {0} Time : {1}".format(valid_size, format_time(time()-TIME_START)))
    #
    #     while n < 10:
    #         n += 1
    #         img, lbs = read_batch(batch_size=BATCH_SIZE, images_source=TRAIN_FOLDER, wnid_labels=wnid_labels)
    #         print("Batch: {0} Time : {1}".format(n, format_time(time()-TIME_START)))
    #         print(img)



def read_local_images(batch_size=64,
                      image_folder='/home/yangfan/ens_ckpts/development_set/images',
                      label_path='/home/yangfan/ens_ckpts/development_set/images.csv'):
    f = open(label_path)
    f.readline()
    label_dict = {}
    for i in range(1000):
        l = f.readline()
        l = l.split(',')
        img_name = l[0] + ".png"
        # print(img_name)
        img_label = int(l[6])
        label_dict[img_name] = img_label

    imgs = []
    labels = []
    for i in range(batch_size):
        img_name = random.choice(os.listdir(image_folder))
        # print(img_name)
        img_label = label_dict[img_name]
        img_path = os.path.join(image_folder, img_name)
        im_array = preprocess_image(img_path)
        imgs.append(im_array)
        labels.append(img_label)
    return np.array(imgs), np.array(labels)



