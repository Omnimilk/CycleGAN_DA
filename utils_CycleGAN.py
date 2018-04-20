import tensorflow as tf
import random
import json

def convert2int(image):
  """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
  """
  return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def convert2float(image):
  """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return (image/127.5) - 1.0

def batch_convert2int(images):
  """
  Args:
    images: 4D float tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D int tensor
  """
  return tf.map_fn(convert2int, images, dtype=tf.uint8)

def batch_convert2float(images):
  """
  Args:
    images: 4D int tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D float tensor
  """
  return tf.map_fn(convert2float, images, dtype=tf.float32)

def readJson(fileName):
      
  with open(fileName) as data_file:
      data = json.load(data_file)
  return data

def image_augmentation(image):
  """Performs data augmentation by randomly permuting the inputs.
  Args:
    image: A float `Tensor` of size [height, width, channels] with values
      in range[0,1].
  Returns:
    The mutated batch of images
  """
  # Apply photometric data augmentation (contrast etc.)
  num_channels = image.get_shape().as_list()[-1]
  if num_channels == 4:
    # Only augment image part
    image, depth = image[:, :, 0:3], image[:, :, 3:4]
  elif num_channels == 1:
    image = tf.image.grayscale_to_rgb(image)
  image = tf.image.random_brightness(image, max_delta=0.001)
  # image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
  # image = tf.image.random_hue(image, max_delta=0.032)
  # image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
  image = tf.clip_by_value(image, 0, 1.0)
  if num_channels == 4:
    image = tf.concat(2, [image, depth])
  elif num_channels == 1:
    image = tf.image.rgb_to_grayscale(image)
  return image

class ImagePool:
  """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image

