#IMAGE READER
import numpy as np
import tensorflow as tf

import os
import random

from image_coder import ImageCoder

def find_image_files(data_dir):
    """
        Build a list of all images files in the data set:
        ----------
        Args:
        data_dir: string, path to the root directory of images. Assumes
        (data_dir/image.png) format
        
        Returns:
        filenames: list of strings; each string is a path to an image file
        following the format data_dir/image%s.png, %s will be replaced
        with color, depth or annoation extension image_color.png
        """
    
    print('[PROGRESS]\tDetermining list of input files from %s' % data_dir)
    
    filenames = []
    
    # Construct the list of image files
    color_file_path = os.path.join(data_dir, '*%s.*') % ('_image')
    label_file_path = os.path.join(data_dir, '*%s.*') % ('_label')
    
    color_files = tf.gfile.Glob(color_file_path)
    label_files = tf.gfile.Glob(label_file_path)
    
    assert len(color_files) == len(label_files)
    
    matching_files = [ x.replace('_image', '%s') for x in color_files ]
    filenames.extend(matching_files)
    
    # Shuffle the ordering of all image files in order to guarantee randomness
    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(shuffled_index)
    
    filenames = [filenames[i] for i in shuffled_index]
    
    print('[INFO    ]\tFound %d images inside %s.' % (len(filenames), data_dir))
    
    #print(filenames)
    return filenames


def read_image_files(filenames):
    
    image_lists = []
    label_lists = []
    coder = ImageCoder()
    for filename in filenames:
        # Concat
        image_file = filename % '_image'
        label_file = filename % '_label'
        
        # Read the image file.
        with tf.gfile.FastGFile(image_file, 'r') as f:
            image_data = f.read()
        # Decode the PNG
        image = coder.decode_png(image_data)
        image = np.array(image, dtype=np.float32)
        # Check that image converted to RGB
        assert len(image.shape) == 3
        #height = image.shape[0]
        #width = image.shape[1]
        assert image.shape[2] == 1
        #print(image.shape[0],image.shape[1],image.shape[2])
        image_lists.append(image)
        
        # Read the label file.
        with tf.gfile.FastGFile(label_file, 'r') as f:
            label_data = f.read()
        # Decode the PNG
        label = coder.decode_png(label_data)
        label = np.array(label, dtype=np.float32)
        # Check that image converted to RGB
        assert len(label.shape) == 3
        #height = image.shape[0]
        #width = image.shape[1]
        assert label.shape[2] == 1
        #print(image.shape[0],image.shape[1],image.shape[2])
        # Convert to one-hot vectors for binary classification
        label = np.dstack((label, 255.-np.squeeze(label)))
        label_lists.append(label)
    
    print('[INFO    ]\tTotal %d images and labels are read. The shape of train_images is %s.' % (len(filenames), np.shape(image_lists)))
    return (image_lists,label_lists)
