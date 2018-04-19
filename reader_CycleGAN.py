import tensorflow as tf
import utils_CycleGAN as utils
from utils import read_feature_names,get_data_paths
import numpy as np
from pprint import pprint

class Reader():
  def __init__(self, tfrecords_file, image_size=np.array([128,160]),
    min_queue_examples=1000, batch_size=8, num_threads=8, name=''):
    """
    Args:
      tfrecords_file: string, tfrecords file path
      min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
      batch_size: integer, number of images per batch
      num_threads: integer, number of preprocess threads
    """
    self.tfrecords_file = tfrecords_file
    self.image_size = image_size
    self.min_queue_examples = min_queue_examples
    self.batch_size = batch_size
    self.num_threads = num_threads
    self.reader = tf.TFRecordReader()
    self.name = name

  def feed(self):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer([self.tfrecords_file])
      reader = tf.TFRecordReader()
      _, serialized_example = self.reader.read(filename_queue)
      features = tf.parse_single_example(
          serialized_example,
          features={
            'image/file_name': tf.FixedLenFeature([], tf.string),
            'image/encoded_image': tf.FixedLenFeature([], tf.string),
          })
      image_buffer = features['image/encoded_image']
      image = tf.image.decode_jpeg(image_buffer, channels=3)
      image = self._preprocess(image)
      images = tf.train.shuffle_batch(
            [image], batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.min_queue_examples + 3*self.batch_size,
            min_after_dequeue=self.min_queue_examples
          )
      tf.summary.image('_input', images)
    return images

  def _preprocess(self, image):
    # image = tf.random_crop(image,[448,560,3])#add random crop
    
    #image = utils.image_augmentation(image)
    image = tf.image.resize_images(image, size=self.image_size)
    image = utils.convert2float(image)
    image = utils.image_augmentation(image)
    #print("self image size: {}".format(self.image_size))
    image.set_shape([self.image_size[0], self.image_size[1], 3])
    return image

class SplitedReader(Reader):
  def feed(self):
    """
    Returns:
      images: 4D tensor [batch_size, image_width, image_height, image_depth]
    """
    with tf.name_scope(self.name):
      filename_queue = tf.train.string_input_producer(self.tfrecords_file)#a list of file path
      reader = tf.TFRecordReader()
      _, serialized_example = self.reader.read(filename_queue)
      #some useful non-image features: "num_grasp_steps", "camera/intrinsics/matrix33"
      features_dict = {
      "grasp/0/image/encoded": tf.FixedLenFeature([],tf.string),
      "grasp/1/image/encoded": tf.FixedLenFeature([],tf.string),
      "grasp/2/image/encoded": tf.FixedLenFeature([],tf.string),
      "grasp/3/image/encoded": tf.FixedLenFeature([],tf.string),
      "grasp/4/image/encoded": tf.FixedLenFeature([],tf.string),
      "grasp/5/image/encoded": tf.FixedLenFeature([],tf.string),
      "grasp/6/image/encoded": tf.FixedLenFeature([],tf.string),
      "grasp/7/image/encoded": tf.FixedLenFeature([],tf.string),
      "grasp/8/image/encoded": tf.FixedLenFeature([],tf.string),
      "grasp/9/image/encoded": tf.FixedLenFeature([],tf.string),
      "gripper/image/encoded": tf.FixedLenFeature([],tf.string),
      "post_drop/image/encoded": tf.FixedLenFeature([],tf.string),
      "post_grasp/image/encoded": tf.FixedLenFeature([],tf.string),
      "present/image/encoded": tf.FixedLenFeature([],tf.string)
      }
      
      # features_dict={          
      #          "grasp/0/image/encoded": tf.FixedLenFeature( [], tf.string)
      #      }
      
      features = tf.parse_single_example(
          serialized_example,features = features_dict
          )
      processed_images = []
      for key in features_dict:
        image_buffer = features[key]
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = self._preprocess(image)
        processed_images.append(image)
      images = tf.train.shuffle_batch(
            processed_images, batch_size=self.batch_size, num_threads=self.num_threads,
            capacity=self.min_queue_examples + 3*self.batch_size,
            min_after_dequeue=self.min_queue_examples
          )
      #images is a list of image batches now, randomly choose one as the data
      images = np.random.choice(images)
      tf.summary.image('_input', images)
    return images

def test_reader():
  #TRAIN_FILE_1 = 'sim_images_a1_low_var.tfrecords'
  #TRAIN_FILE_2 = 'sim_images_a2_low_var.tfrecords'
  TRAIN_FILE_1 = get_data_paths("Data/tfdata1","34")
  # pprint(TRAIN_FILE_1)
  with tf.Graph().as_default():
    reader1 = SplitedReader(TRAIN_FILE_1, batch_size=2)
    #reader1 = Reader(TRAIN_FILE_1, batch_size=2)
    #reader2 = Reader(TRAIN_FILE_2, batch_size=2)
    images_op1 = reader1.feed()
    #images_op2 = reader2.feed()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      step = 0
      while not coord.should_stop():
        #batch_images1, batch_images2 = sess.run([images_op1, images_op2])
        batch_images1= sess.run(images_op1)
        print("image shape: {}".format(batch_images1))
        #print("image shape: {}".format(batch_images2))
        print("="*10)
        step += 1
    except KeyboardInterrupt:
      print('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

if __name__ == '__main__':
  test_reader()
