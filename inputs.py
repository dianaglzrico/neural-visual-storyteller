# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Input ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np


def parse_sequence_example(serialized, image_feature, caption_feature):
  context, sequence = tf.parse_single_sequence_example(
      serialized,
      context_features={
          "image/data_0": tf.FixedLenFeature([], dtype=tf.string),          
          "image/image_id_0": tf.FixedLenFeature([], dtype=tf.int64),
          "image/order_0": tf.FixedLenFeature([], dtype=tf.int64),                    
          "image/story_id_0": tf.FixedLenFeature([], dtype=tf.int64),
          "image/album_id_0": tf.FixedLenFeature([], dtype=tf.int64),          
          "image/caption_id_0": tf.FixedLenFeature([], dtype=tf.int64),

          "image/data_1": tf.FixedLenFeature([], dtype=tf.string),          
          "image/image_id_1": tf.FixedLenFeature([], dtype=tf.int64),
          "image/order_1": tf.FixedLenFeature([], dtype=tf.int64),                    
          "image/story_id_1": tf.FixedLenFeature([], dtype=tf.int64),
          "image/album_id_1": tf.FixedLenFeature([], dtype=tf.int64),          
          "image/caption_id_1": tf.FixedLenFeature([], dtype=tf.int64),

          "image/data_2": tf.FixedLenFeature([], dtype=tf.string),          
          "image/image_id_2": tf.FixedLenFeature([], dtype=tf.int64),
          "image/order_2": tf.FixedLenFeature([], dtype=tf.int64),                    
          "image/story_id_2": tf.FixedLenFeature([], dtype=tf.int64),
          "image/album_id_2": tf.FixedLenFeature([], dtype=tf.int64),          
          "image/caption_id_2": tf.FixedLenFeature([], dtype=tf.int64),

          "image/data_3": tf.FixedLenFeature([], dtype=tf.string),          
          "image/image_id_3": tf.FixedLenFeature([], dtype=tf.int64),
          "image/order_3": tf.FixedLenFeature([], dtype=tf.int64),                    
          "image/story_id_3": tf.FixedLenFeature([], dtype=tf.int64),
          "image/album_id_3": tf.FixedLenFeature([], dtype=tf.int64),          
          "image/caption_id_3": tf.FixedLenFeature([], dtype=tf.int64),

          "image/data_4": tf.FixedLenFeature([], dtype=tf.string),          
          "image/image_id_4": tf.FixedLenFeature([], dtype=tf.int64),
          "image/order_4": tf.FixedLenFeature([], dtype=tf.int64),                    
          "image/story_id_4": tf.FixedLenFeature([], dtype=tf.int64),
          "image/album_id_4": tf.FixedLenFeature([], dtype=tf.int64),          
          "image/caption_id_4": tf.FixedLenFeature([], dtype=tf.int64),
      },
      sequence_features={
          "image/caption_ids_0": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "image/caption_ids_1": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "image/caption_ids_2": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "image/caption_ids_3": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "image/caption_ids_4": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

  encoded_image_0 = context["image/data_0"]
  caption_0 =  sequence["image/caption_ids_0"] 

  encoded_image_1 = context["image/data_1"]
  caption_1 =  sequence["image/caption_ids_1"]

  encoded_image_2 = context["image/data_2"]
  caption_2 =  sequence["image/caption_ids_2"]

  encoded_image_3 = context["image/data_3"]
  caption_3 =  sequence["image/caption_ids_3"]

  encoded_image_4 = context["image/data_4"]
  caption_4 =  sequence["image/caption_ids_4"]

  
  return encoded_image_0, caption_0, encoded_image_1, caption_1, encoded_image_2, caption_2, encoded_image_3, caption_3, encoded_image_4, caption_4


def prefetch_input_data(reader,
                        file_pattern,
                        is_training,
                        batch_size,
                        values_per_shard,
                        input_queue_capacity_factor=15,
                        num_reader_threads=1,
                        shard_queue_name="filename_queue",
                        value_queue_name="input_queue"):
  """Prefetches string values from disk into an input queue.

  In training the capacity of the queue is important because a larger queue
  means better mixing of training examples between shards. The minimum number of
  values kept in the queue is values_per_shard * input_queue_capacity_factor,
  where input_queue_memory factor should be chosen to trade-off better mixing
  with memory usage.

  Args:
    reader: Instance of tf.ReaderBase.
    file_pattern: Comma-separated list of file patterns (e.g.
        /tmp/train_data-?????-of-00100).
    is_training: Boolean; whether prefetching for training or eval.
    batch_size: Model batch size used to determine queue capacity.
    values_per_shard: Approximate number of values per shard.
    input_queue_capacity_factor: Minimum number of values to keep in the queue
      in multiples of values_per_shard. See comments above.
    num_reader_threads: Number of reader threads to fill the queue.
    shard_queue_name: Name for the shards filename queue.
    value_queue_name: Name for the values input queue.

  Returns:
    A Queue containing prefetched string values.
  """
  data_files = []
  for pattern in file_pattern.split(","):
    data_files.extend(tf.gfile.Glob(pattern))
  if not data_files:
    tf.logging.fatal("Found no input files matching %s", file_pattern)
  else:
    tf.logging.info("Prefetching values from %d files matching %s",
                    len(data_files), file_pattern)

  if is_training:    
    filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=15, name=shard_queue_name)
    capacity = values_per_shard * batch_size
    values_queue = tf.FIFOQueue(capacity=capacity, dtypes=[tf.string], name="fifoTrain_" + value_queue_name)
  else:
    filename_queue = tf.train.string_input_producer(
        data_files, shuffle=False, capacity=1, name=shard_queue_name)
    capacity = values_per_shard * batch_size
    values_queue = tf.FIFOQueue(capacity=capacity, dtypes=[tf.string], name="fifoEval_" + value_queue_name)

  enqueue_ops = []

  for i in range(num_reader_threads):  
    _, value = reader.read(filename_queue)
    enqueue_ops.append(values_queue.enqueue([value]))


  tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
      values_queue, enqueue_ops))

  tf.summary.scalar(
      "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
      tf.cast(values_queue.size(), tf.float32) * (1. / capacity))
  
  return values_queue


def batch_with_dynamic_pad(images_and_captions,
                           batch_size,
                           queue_capacity,
                           add_summaries=True):
  """Batches input images and captions.

  This function splits the caption into an input sequence and a target sequence,
  where the target sequence is the input sequence right-shifted by 1. Input and
  target sequences are batched and padded up to the maximum length of sequences
  in the batch. A mask is created to distinguish real words from padding words.

  Example:
    Actual captions in the batch ('-' denotes padded character):
      [
        [ 1 2 3 4 5 ],
        [ 1 2 3 4 - ],
        [ 1 2 3 - - ],
      ]

    input_seqs:
      [
        [ 1 2 3 4 ],
        [ 1 2 3 - ],
        [ 1 2 - - ],
      ]

    target_seqs:
      [
        [ 2 3 4 5 ],
        [ 2 3 4 - ],
        [ 2 3 - - ],
      ]

    mask:
      [
        [ 1 1 1 1 ],
        [ 1 1 1 0 ],
        [ 1 1 0 0 ],
      ]

  Args:
    images_and_captions: A list of pairs [image, caption], where image is a
      Tensor of shape [height, width, channels] and caption is a 1-D Tensor of
      any length. Each pair will be processed and added to the queue in a
      separate thread.
    batch_size: Batch size.
    queue_capacity: Queue capacity.
    add_summaries: If true, add caption length summaries.

  Returns:
    images: A Tensor of shape [batch_size, height, width, channels].
    input_seqs: An int32 Tensor of shape [batch_size, padded_length * 5].
    target_seqs: An int32 Tensor of shape [batch_size, padded_length * 5].
    mask: An int32 0/1 Tensor of shape [batch_size, padded_length * 5].  
  
    tf.subtract = entre dos tensores x & y: Returns x - y element-wise.
    tf.expand_dims = Inserts a dimension of 1 into a tensor's shape. 
        Given a tensor input, this operation inserts a dimension of 1 at the dimension index axis of input's shape. 
        The dimension index axis starts at zero; if you specify a negative number for axis it is counted backward from the end.
        This operation is useful if you want to add a batch dimension to a single element. 
        For example, if you have a single image of shape [height, width, channels], 
        you can make it a batch of 1 image with expand_dims(image, 0), which will make the shape [1, height, width, channels]. 
    tf.slice = Extracts a slice from a tensor. 
        This operation extracts a slice of size size from a tensor input starting at the location specified by begin. 
        The slice size is represented as a tensor shape, where size[i] is the number of elements of the 'i'th dimension 
        of input that you want to slice. The starting location (begin) for the slice is represented as an offset in 
        each dimension of input. In other words, begin[i] is the offset into the 'i'th dimension of input that you want to slice from.
    tf.ones = Creates a tensor with all elements set to 1.
  """

  enqueue_list = []
  for image_0, caption_0, image_1, caption_1, image_2, caption_2, image_3, caption_3, image_4, caption_4 in images_and_captions:

    caption_length_0 = tf.shape(caption_0)[0]        
    input_length_0 = tf.expand_dims(tf.subtract(caption_length_0, 1), 0)
    input_seq_0 = tf.slice(caption_0, [0], input_length_0)
    target_seq_0 = tf.slice(caption_0, [1], input_length_0)    
    indicator_0 = tf.ones(input_length_0, dtype=tf.int32)

    caption_length_1 = tf.shape(caption_1)[0]    
    input_length_1 = tf.expand_dims(tf.subtract(caption_length_1, 1), 0)
    input_seq_1 = tf.slice(caption_1, [0], input_length_1)
    target_seq_1 = tf.slice(caption_1, [1], input_length_1)    
    indicator_1 = tf.ones(input_length_1, dtype=tf.int32)

    caption_length_2 = tf.shape(caption_2)[0]    
    input_length_2 = tf.expand_dims(tf.subtract(caption_length_2, 1), 0)
    input_seq_2 = tf.slice(caption_2, [0], input_length_2)
    target_seq_2 = tf.slice(caption_2, [1], input_length_2)    
    indicator_2 = tf.ones(input_length_2, dtype=tf.int32)

    caption_length_3 = tf.shape(caption_3)[0]    
    input_length_3 = tf.expand_dims(tf.subtract(caption_length_3, 1), 0)
    input_seq_3 = tf.slice(caption_3, [0], input_length_3)
    target_seq_3 = tf.slice(caption_3, [1], input_length_3)    
    indicator_3 = tf.ones(input_length_3, dtype=tf.int32)

    caption_length_4 = tf.shape(caption_4)[0]    
    input_length_4 = tf.expand_dims(tf.subtract(caption_length_4, 1), 0)
    input_seq_4 = tf.slice(caption_4, [0], input_length_4)
    target_seq_4 = tf.slice(caption_4, [1], input_length_4)    
    indicator_4 = tf.ones(input_length_4, dtype=tf.int32)
    
    enqueue_list.append([image_0, input_seq_0, target_seq_0, indicator_0, image_1, input_seq_1, target_seq_1, indicator_1, image_2, input_seq_2, target_seq_2, indicator_2, image_3, input_seq_3, target_seq_3, indicator_3, image_4, input_seq_4, target_seq_4, indicator_4])

  images_0, input_seqs_0, target_seqs_0, mask_0, images_1, input_seqs_1, target_seqs_1, mask_1, images_2, input_seqs_2, target_seqs_2, mask_2, images_3, input_seqs_3, target_seqs_3, mask_3, images_4, input_seqs_4, target_seqs_4, mask_4 = tf.train.batch_join(
      enqueue_list,
      batch_size=batch_size,
      capacity=queue_capacity,
      dynamic_pad=True,
      name="batch_and_pad")

  if add_summaries:
    lengths_0 = tf.add(tf.reduce_sum(mask_0, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths_0))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths_0))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths_0))

    lengths_1 = tf.add(tf.reduce_sum(mask_1, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths_1))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths_1))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths_1))

    lengths_2 = tf.add(tf.reduce_sum(mask_2, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths_2))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths_2))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths_2))

    lengths_3 = tf.add(tf.reduce_sum(mask_3, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths_3))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths_3))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths_3))

    lengths_4 = tf.add(tf.reduce_sum(mask_4, 1), 1)
    tf.summary.scalar("caption_length/batch_min", tf.reduce_min(lengths_4))
    tf.summary.scalar("caption_length/batch_max", tf.reduce_max(lengths_4))
    tf.summary.scalar("caption_length/batch_mean", tf.reduce_mean(lengths_4))
  
  return images_0, input_seqs_0, target_seqs_0, mask_0, images_1, input_seqs_1, target_seqs_1, mask_1, images_2, input_seqs_2, target_seqs_2, mask_2, images_3, input_seqs_3, target_seqs_3, mask_3, images_4, input_seqs_4, target_seqs_4, mask_4
