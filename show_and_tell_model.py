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

"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from im2txt.ops import image_embedding
from im2txt.ops import image_processing
from im2txt.ops import inputs as input_ops


class ShowAndTellModel(object):
  def __init__(self, config, mode, train_inception=False):
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # Each one is a float32 Tensor with shape [batch_size, height, width, channels].
    self.images_0 = None
    self.images_1 = None
    self.images_2 = None
    self.images_3 = None
    self.images_4 = None

    # Each one is an int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs_0 = None
    self.input_seqs_1 = None
    self.input_seqs_2 = None
    self.input_seqs_3 = None
    self.input_seqs_4 = None

    # Each one is an int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs_0 = None
    self.target_seqs_1 = None
    self.target_seqs_2 = None
    self.target_seqs_3 = None
    self.target_seqs_4 = None

    # Each one is an int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask_0 = None
    self.input_mask_1 = None
    self.input_mask_2 = None
    self.input_mask_3 = None
    self.input_mask_4 = None

    # Each one is a float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings_0 = None
    self.image_embeddings_1 = None
    self.image_embeddings_2 = None
    self.image_embeddings_3 = None
    self.image_embeddings_4 = None

    # Each one is a float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings_0 = None
    self.seq_embeddings_1 = None
    self.seq_embeddings_2 = None
    self.seq_embeddings_3 = None
    self.seq_embeddings_4 = None

    # Each one is a float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # Each one is a float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses_0 = None
    self.target_cross_entropy_losses_1 = None
    self.target_cross_entropy_losses_2 = None
    self.target_cross_entropy_losses_3 = None
    self.target_cross_entropy_losses_4 = None

    # Each one is a float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights_0 = None
    self.target_cross_entropy_loss_weights_1 = None
    self.target_cross_entropy_loss_weights_2 = None
    self.target_cross_entropy_loss_weights_3 = None
    self.target_cross_entropy_loss_weights_4 = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

  def is_training(self):
    return self.mode == "train"

  def process_image(self, encoded_image, thread_id=0):
    return image_processing.process_image(encoded_image,
                                          is_training=self.is_training(),
                                          height=self.config.image_height,
                                          width=self.config.image_width,
                                          thread_id=thread_id,
                                          image_format=self.config.image_format)

  def build_inputs(self):
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed_0 = tf.placeholder(dtype=tf.string, shape=[], name="image_feed_0")
      image_feed_1 = tf.placeholder(dtype=tf.string, shape=[], name="image_feed_1")
      image_feed_2 = tf.placeholder(dtype=tf.string, shape=[], name="image_feed_2")
      image_feed_3 = tf.placeholder(dtype=tf.string, shape=[], name="image_feed_3")
      image_feed_4 = tf.placeholder(dtype=tf.string, shape=[], name="image_feed_4")

      input_feed_0 = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed_0")
      input_feed_1 = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed_1")
      input_feed_2 = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed_2")
      input_feed_3 = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed_3")
      input_feed_4 = tf.placeholder(dtype=tf.int64, shape=[None], name="input_feed_4")

      # Process each image and insert batch dimensions.
      images_0 = tf.expand_dims(self.process_image(image_feed_0), 0)
      images_1 = tf.expand_dims(self.process_image(image_feed_1), 0)
      images_2 = tf.expand_dims(self.process_image(image_feed_2), 0)
      images_3 = tf.expand_dims(self.process_image(image_feed_3), 0)
      images_4 = tf.expand_dims(self.process_image(image_feed_4), 0)

      input_seqs_0 = tf.expand_dims(input_feed_0, 1)
      input_seqs_1 = tf.expand_dims(input_feed_1, 1)
      input_seqs_2 = tf.expand_dims(input_feed_2, 1)
      input_seqs_3 = tf.expand_dims(input_feed_3, 1)
      input_seqs_4 = tf.expand_dims(input_feed_4, 1)

      # No target sequences or input mask in inference mode.
      target_seqs_0 = None
      target_seqs_1 = None
      target_seqs_2 = None
      target_seqs_3 = None
      target_seqs_4 = None

      input_mask_0 = None
      input_mask_1 = None
      input_mask_2 = None
      input_mask_3 = None
      input_mask_4 = None

    else:
      # Prefetch serialized SequenceExample protos.
      input_queue = input_ops.prefetch_input_data(
          self.reader,
          self.config.input_file_pattern,
          is_training=self.is_training(),
          batch_size=self.config.batch_size,
          values_per_shard=self.config.values_per_input_shard,
          input_queue_capacity_factor=self.config.input_queue_capacity_factor,
          num_reader_threads=self.config.num_input_reader_threads)

      self.input_queue = input_queue

      assert self.config.num_preprocess_threads % 2 == 0

      # Image processing and random distortion. Split across multiple threads
      # with each thread applying a slightly different distortion.
      images_and_captions = []
      for thread_id in range(self.config.num_preprocess_threads):
        serialized_sequence_example = input_queue.dequeue()
        encoded_image_0, caption_0, encoded_image_1, caption_1, encoded_image_2, caption_2, encoded_image_3, caption_3, encoded_image_4, caption_4 = input_ops.parse_sequence_example(serialized_sequence_example, image_feature=self.config.image_feature_name, caption_feature=self.config.caption_feature_name)
        image_0 = self.process_image(encoded_image_0, thread_id=thread_id)
        image_1 = self.process_image(encoded_image_1, thread_id=thread_id)
        image_2 = self.process_image(encoded_image_2, thread_id=thread_id)
        image_3 = self.process_image(encoded_image_3, thread_id=thread_id)
        image_4 = self.process_image(encoded_image_4, thread_id=thread_id)
        images_and_captions.append([image_0, caption_0, image_1, caption_1, image_2, caption_2, image_3, caption_3, image_4, caption_4])

      self.images_and_captions = images_and_captions

      queue_capacity = (2 * self.config.num_preprocess_threads * self.config.batch_size) #200
      images_0, input_seqs_0, target_seqs_0, input_mask_0, images_1, input_seqs_1, target_seqs_1, input_mask_1, images_2, input_seqs_2, target_seqs_2, input_mask_2, images_3, input_seqs_3, target_seqs_3, input_mask_3, images_4, input_seqs_4, target_seqs_4, input_mask_4 = (
          input_ops.batch_with_dynamic_pad(images_and_captions,
                                           batch_size=self.config.batch_size,
                                           queue_capacity=queue_capacity))
    self.images_0 = images_0
    self.input_seqs_0 = input_seqs_0
    self.target_seqs_0 = target_seqs_0
    self.input_mask_0 = input_mask_0

    self.images_1 = images_1
    self.input_seqs_1 = input_seqs_1
    self.target_seqs_1 = target_seqs_1
    self.input_mask_1 = input_mask_1

    self.images_2 = images_2
    self.input_seqs_2 = input_seqs_2
    self.target_seqs_2 = target_seqs_2
    self.input_mask_2 = input_mask_2

    self.images_3 = images_3
    self.input_seqs_3 = input_seqs_3
    self.target_seqs_3 = target_seqs_3
    self.input_mask_3 = input_mask_3

    self.images_4 = images_4
    self.input_seqs_4 = input_seqs_4
    self.target_seqs_4 = target_seqs_4
    self.input_mask_4 = input_mask_4

  def build_image_embeddings(self):
      # Get image representation via Inception V3 model
    inception_output_0 = image_embedding.inception_v3(
        self.images_0,
        trainable=self.train_inception,
        is_training=self.is_training(), scope="InceptionV3")
    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    inception_output_1 = image_embedding.inception_v3(
            self.images_1,
            trainable=self.train_inception,
            is_training=self.is_training(), scope="InceptionV3")
    self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    inception_output_2 = image_embedding.inception_v3(
            self.images_2,
            trainable=self.train_inception,
            is_training=self.is_training(), scope="InceptionV3")
    self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    inception_output_3 = image_embedding.inception_v3(
            self.images_3,
            trainable=self.train_inception,
            is_training=self.is_training(), scope="InceptionV3")
    self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    inception_output_4 = image_embedding.inception_v3(
            self.images_4,
            trainable=self.train_inception,
            is_training=self.is_training(), scope="InceptionV3")
    self.inception_variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings_0 = tf.contrib.layers.fully_connected(
          inputs=inception_output_0,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

      scope.reuse_variables()

      image_embeddings_1 = tf.contrib.layers.fully_connected(
          inputs=inception_output_1,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

      image_embeddings_2 = tf.contrib.layers.fully_connected(
          inputs=inception_output_2,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

      image_embeddings_3 = tf.contrib.layers.fully_connected(
          inputs=inception_output_3,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

      image_embeddings_4 = tf.contrib.layers.fully_connected(
          inputs=inception_output_4,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    tf.constant(self.config.embedding_size, name="embedding_size")

    self.image_embeddings_0 = image_embeddings_0
    self.image_embeddings_1 = image_embeddings_1
    self.image_embeddings_2 = image_embeddings_2
    self.image_embeddings_3 = image_embeddings_3
    self.image_embeddings_4 = image_embeddings_4

  def build_seq_embeddings(self):
    with tf.variable_scope("seq_embedding_0"):
      embedding_map_0 = tf.get_variable(
          name="map_0",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings_0 = tf.nn.embedding_lookup(embedding_map_0, self.input_seqs_0)

    with tf.variable_scope("seq_embedding_1"):
      embedding_map_1 = tf.get_variable(
          name="map_1",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings_1 = tf.nn.embedding_lookup(embedding_map_1, self.input_seqs_1)

    with tf.variable_scope("seq_embedding_2"):
      embedding_map_2 = tf.get_variable(
          name="map_2",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings_2 = tf.nn.embedding_lookup(embedding_map_2, self.input_seqs_2)

    with tf.variable_scope("seq_embedding_3"):
      embedding_map_3 = tf.get_variable(
          name="map_3",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings_3 = tf.nn.embedding_lookup(embedding_map_3, self.input_seqs_3)

    with tf.variable_scope("seq_embedding_4"):
      embedding_map_4 = tf.get_variable(
          name="map_4",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings_4 = tf.nn.embedding_lookup(embedding_map_4, self.input_seqs_4)

    self.seq_embeddings_0 = seq_embeddings_0
    self.seq_embeddings_1 = seq_embeddings_1
    self.seq_embeddings_2 = seq_embeddings_2
    self.seq_embeddings_3 = seq_embeddings_3
    self.seq_embeddings_4 = seq_embeddings_4


  def build_model(self):
    # Define cell
    lstm_cell_enc = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, state_is_tuple=True)
    lstm_cell_dec_0 = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, state_is_tuple=True)
    lstm_cell_dec_1 = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, state_is_tuple=True)
    lstm_cell_dec_2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, state_is_tuple=True)
    lstm_cell_dec_3 = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, state_is_tuple=True)
    lstm_cell_dec_4 = tf.contrib.rnn.BasicLSTMCell(num_units=self.config.num_lstm_units, state_is_tuple=True)

    # Adds dropout when training
    if self.mode == "train":
      lstm_cell_enc = tf.contrib.rnn.DropoutWrapper(lstm_cell_enc, input_keep_prob=self.config.lstm_dropout_keep_prob, output_keep_prob=self.config.lstm_dropout_keep_prob)
      lstm_cell_dec_0 = tf.contrib.rnn.DropoutWrapper(lstm_cell_dec_0, input_keep_prob=self.config.lstm_dropout_keep_prob, output_keep_prob=self.config.lstm_dropout_keep_prob)
      lstm_cell_dec_1 = tf.contrib.rnn.DropoutWrapper(lstm_cell_dec_1, input_keep_prob=self.config.lstm_dropout_keep_prob, output_keep_prob=self.config.lstm_dropout_keep_prob)
      lstm_cell_dec_2 = tf.contrib.rnn.DropoutWrapper(lstm_cell_dec_2, input_keep_prob=self.config.lstm_dropout_keep_prob, output_keep_prob=self.config.lstm_dropout_keep_prob)
      lstm_cell_dec_3 = tf.contrib.rnn.DropoutWrapper(lstm_cell_dec_3, input_keep_prob=self.config.lstm_dropout_keep_prob, output_keep_prob=self.config.lstm_dropout_keep_prob)
      lstm_cell_dec_4 = tf.contrib.rnn.DropoutWrapper(lstm_cell_dec_4, input_keep_prob=self.config.lstm_dropout_keep_prob, output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("encoder", initializer=self.initializer) as encoder_scope:
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell_enc.zero_state(batch_size=self.image_embeddings_0.get_shape()[0], dtype=tf.float32)
      _, initial_state_enc = lstm_cell_enc(self.image_embeddings_0, zero_state)

      # Allow the LSTM variables to be reused.
      encoder_scope.reuse_variables()

      encoder_inputs = tf.stack([self.image_embeddings_1, self.image_embeddings_2, self.image_embeddings_3, self.image_embeddings_4], axis=1)

      # Run the batch of sequence embeddings through the LSTM.
      encoder_outputs, encoder_states = tf.nn.dynamic_rnn(cell=lstm_cell_enc,
                                          inputs=encoder_inputs,
                                          initial_state=initial_state_enc,
                                          dtype=tf.float32,
                                          scope=encoder_scope)

    with tf.variable_scope("decoder", initializer=self.initializer) as decoder_scope:
      # Multiple decoders
      with tf.variable_scope("decoder_0", initializer=self.initializer) as decoder_0_scope:
        # Feed the image embeddings to set the initial LSTM state.
        _, initial_state_0 = lstm_cell_dec_0(self.image_embeddings_0, encoder_states)

        # Run the batch of sequence embeddings through the LSTM.
        decoder_0_scope.reuse_variables()

        if self.mode == "inference":
          tf.concat(axis=1, values=initial_state_0, name="initial_state_0")

        # Placeholder for feeding a batch of concatenated states.
          state_feed_0 = tf.placeholder(dtype=tf.float32, shape=[None, sum(lstm_cell_dec_0.state_size)], name="state_feed_0")
          state_tuple_0 = tf.split(value=state_feed_0, num_or_size_splits=2, axis=1)

        # Run a single LSTM step.
          decoder_outputs_0, state_tuple_0 = lstm_cell_dec_0(inputs=tf.squeeze(self.seq_embeddings_0, axis=[1]), state=state_tuple_0)

        # Concatentate the resulting state.
          tf.concat(axis=1, values=state_tuple_0, name="state_0")

        else:
          decoder_outputs_0, _ = tf.nn.dynamic_rnn(cell=lstm_cell_dec_0,
                                          inputs=self.seq_embeddings_0,
                                          initial_state=initial_state_0,
                                          dtype=tf.float32,
                                          scope=decoder_0_scope)

      with tf.variable_scope("decoder_1", initializer=self.initializer) as decoder_1_scope:
        # Feed the image embeddings to set the initial LSTM state.
        _, initial_state_1 = lstm_cell_dec_1(self.image_embeddings_1, encoder_states)

        # Allow the LSTM variables to be reused.
        decoder_1_scope.reuse_variables()

        if self.mode == "inference":
          tf.concat(axis=1, values=initial_state_1, name="initial_state_1")

          # Placeholder for feeding a batch of concatenated states.
          state_feed_1 = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell_dec_1.state_size)],
                                    name="state_feed_1")
          state_tuple_1 = tf.split(value=state_feed_1, num_or_size_splits=2, axis=1)

          # Run a single LSTM step.
          decoder_outputs_1, state_tuple_1 = lstm_cell_dec_1(
            inputs=tf.squeeze(self.seq_embeddings_1, axis=[1]),
            state=state_tuple_1)

          # Concatentate the resulting state.
          tf.concat(axis=1, values=state_tuple_1, name="state_1")

        else:
          decoder_outputs_1, _ = tf.nn.dynamic_rnn(cell=lstm_cell_dec_1,
                                          inputs=self.seq_embeddings_1,
                                          initial_state=initial_state_1,
                                          dtype=tf.float32,
                                          scope=decoder_1_scope)

      with tf.variable_scope("decoder_2", initializer=self.initializer) as decoder_2_scope:
        # Feed the image embeddings to set the initial LSTM state.
        _, initial_state_2 = lstm_cell_dec_2(self.image_embeddings_2, encoder_states)

        # Allow the LSTM variables to be reused.
        decoder_2_scope.reuse_variables()

        if self.mode == "inference":
          tf.concat(axis=1, values=initial_state_2, name="initial_state_2")

          # Placeholder for feeding a batch of concatenated states.
          state_feed_2 = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell_dec_2.state_size)],
                                    name="state_feed_2")
          state_tuple_2 = tf.split(value=state_feed_2, num_or_size_splits=2, axis=1)

          # Run a single LSTM step.
          decoder_outputs_2, state_tuple_2 = lstm_cell_dec_2(
            inputs=tf.squeeze(self.seq_embeddings_2, axis=[1]),
            state=state_tuple_2)

          # Concatentate the resulting state.
          tf.concat(axis=1, values=state_tuple_2, name="state_2")

        else:
          decoder_outputs_2, _ = tf.nn.dynamic_rnn(cell=lstm_cell_dec_2,
                                            inputs=self.seq_embeddings_2,
                                            initial_state=initial_state_2,
                                            dtype=tf.float32,
                                            scope=decoder_2_scope)

      with tf.variable_scope("decoder_3", initializer=self.initializer) as decoder_3_scope:
        # Feed the image embeddings to set the initial LSTM state.
        _, initial_state_3 = lstm_cell_dec_3(self.image_embeddings_3, encoder_states)

        # Allow the LSTM variables to be reused.
        decoder_3_scope.reuse_variables()

        if self.mode == "inference":
          tf.concat(axis=1, values=initial_state_3, name="initial_state_3")

          # Placeholder for feeding a batch of concatenated states.
          state_feed_3 = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell_dec_3.state_size)],
                                    name="state_feed_3")
          state_tuple_3 = tf.split(value=state_feed_3, num_or_size_splits=2, axis=1)

          # Run a single LSTM step.
          decoder_outputs_3, state_tuple_3 = lstm_cell_dec_3(
            inputs=tf.squeeze(self.seq_embeddings_3, axis=[1]),
            state=state_tuple_3)

          # Concatentate the resulting state.
          tf.concat(axis=1, values=state_tuple_3, name="state_3")

        else:
          decoder_outputs_3, _ = tf.nn.dynamic_rnn(cell=lstm_cell_dec_3,
                                            inputs=self.seq_embeddings_3,
                                            initial_state=initial_state_3,
                                            dtype=tf.float32,
                                            scope=decoder_3_scope)

      with tf.variable_scope("decoder_4", initializer=self.initializer) as decoder_4_scope:
        # Feed the image embeddings to set the initial LSTM state.
        _, initial_state_4 = lstm_cell_dec_4(self.image_embeddings_4, encoder_states)

        # Allow the LSTM variables to be reused.
        decoder_4_scope.reuse_variables()

        if self.mode == "inference":
          tf.concat(axis=1, values=initial_state_4, name="initial_state_4")

          # Placeholder for feeding a batch of concatenated states.
          state_feed_4 = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell_dec_4.state_size)],
                                    name="state_feed_4")
          state_tuple_4 = tf.split(value=state_feed_4, num_or_size_splits=2, axis=1)

          # Run a single LSTM step.
          decoder_outputs_4, state_tuple_4 = lstm_cell_dec_4(
            inputs=tf.squeeze(self.seq_embeddings_4, axis=[1]),
            state=state_tuple_4)

          # Concatentate the resulting state.
          tf.concat(axis=1, values=state_tuple_4, name="state_4")

        else:
          decoder_outputs_4, _ = tf.nn.dynamic_rnn(cell=lstm_cell_dec_4,
                                            inputs=self.seq_embeddings_4,
                                            initial_state=initial_state_4,
                                            dtype=tf.float32,
                                            scope=decoder_4_scope)

    # Stack batches vertically.
    decoder_outputs_0 = tf.reshape(decoder_outputs_0, [-1, lstm_cell_dec_0.output_size])
    decoder_outputs_1 = tf.reshape(decoder_outputs_1, [-1, lstm_cell_dec_1.output_size])
    decoder_outputs_2 = tf.reshape(decoder_outputs_2, [-1, lstm_cell_dec_2.output_size])
    decoder_outputs_3 = tf.reshape(decoder_outputs_3, [-1, lstm_cell_dec_3.output_size])
    decoder_outputs_4 = tf.reshape(decoder_outputs_4, [-1, lstm_cell_dec_4.output_size])

    with tf.variable_scope("logits_0") as logits_0_scope:
      logits_0 = tf.contrib.layers.fully_connected(
          inputs=decoder_outputs_0,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_0_scope)

    with tf.variable_scope("logits_1") as logits_1_scope:
      logits_1 = tf.contrib.layers.fully_connected(
          inputs=decoder_outputs_1,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_1_scope)

    with tf.variable_scope("logits_2") as logits_2_scope:
      logits_2 = tf.contrib.layers.fully_connected(
          inputs=decoder_outputs_2,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_2_scope)

    with tf.variable_scope("logits_3") as logits_3_scope:
      logits_3 = tf.contrib.layers.fully_connected(
          inputs=decoder_outputs_3,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_3_scope)

    with tf.variable_scope("logits_4") as logits_4_scope:
      logits_4 = tf.contrib.layers.fully_connected(
          inputs=decoder_outputs_4,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_4_scope)

    if self.mode == "inference":
      tf.nn.softmax(logits_0, name="softmax_0")
      tf.nn.softmax(logits_1, name="softmax_1")
      tf.nn.softmax(logits_2, name="softmax_2")
      tf.nn.softmax(logits_3, name="softmax_3")
      tf.nn.softmax(logits_4, name="softmax_4")

    else:
      targets_0 = tf.reshape(self.target_seqs_0, [-1])
      weights_0 = tf.to_float(tf.reshape(self.input_mask_0, [-1]))

      targets_1 = tf.reshape(self.target_seqs_1, [-1])
      weights_1 = tf.to_float(tf.reshape(self.input_mask_1, [-1]))

      targets_2 = tf.reshape(self.target_seqs_2, [-1])
      weights_2 = tf.to_float(tf.reshape(self.input_mask_2, [-1]))

      targets_3 = tf.reshape(self.target_seqs_3, [-1])
      weights_3 = tf.to_float(tf.reshape(self.input_mask_3, [-1]))

      targets_4 = tf.reshape(self.target_seqs_4, [-1])
      weights_4 = tf.to_float(tf.reshape(self.input_mask_4, [-1]))

      # Compute losses.
      losses_0 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_0, logits=logits_0)
      batch_loss_0 = tf.div(tf.reduce_sum(tf.multiply(losses_0, weights_0)), tf.reduce_sum(weights_0), name="batch_loss_0")

      losses_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_1, logits=logits_1)
      batch_loss_1 = tf.div(tf.reduce_sum(tf.multiply(losses_1, weights_1)), tf.reduce_sum(weights_1), name="batch_loss_1")

      losses_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_2, logits=logits_2)
      batch_loss_2 = tf.div(tf.reduce_sum(tf.multiply(losses_2, weights_2)), tf.reduce_sum(weights_2), name="batch_loss_2")

      losses_3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_3, logits=logits_3)
      batch_loss_3 = tf.div(tf.reduce_sum(tf.multiply(losses_3, weights_3)), tf.reduce_sum(weights_3), name="batch_loss_3")

      losses_4 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets_4, logits=logits_4)
      batch_loss_4 = tf.div(tf.reduce_sum(tf.multiply(losses_4, weights_4)), tf.reduce_sum(weights_4), name="batch_loss_4")

      tf.losses.add_loss(batch_loss_0)
      tf.losses.add_loss(batch_loss_1)
      tf.losses.add_loss(batch_loss_2)
      tf.losses.add_loss(batch_loss_3)
      tf.losses.add_loss(batch_loss_4)

      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/batch_loss_0", batch_loss_0)
      tf.summary.scalar("losses/batch_loss_1", batch_loss_1)
      tf.summary.scalar("losses/batch_loss_2", batch_loss_2)
      tf.summary.scalar("losses/batch_loss_3", batch_loss_3)
      tf.summary.scalar("losses/batch_loss_4", batch_loss_4)
      tf.summary.scalar("losses/total_loss", total_loss)

      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss
      self.target_cross_entropy_losses_0 = losses_0  # Used in evaluation.
      self.target_cross_entropy_losses_1 = losses_1  # Used in evaluation.
      self.target_cross_entropy_losses_2 = losses_2  # Used in evaluation.
      self.target_cross_entropy_losses_3 = losses_3  # Used in evaluation.
      self.target_cross_entropy_losses_4 = losses_4  # Used in evaluation.

      self.target_cross_entropy_loss_weights_0 = weights_0  # Used in evaluation.
      self.target_cross_entropy_loss_weights_1 = weights_1  # Used in evaluation.
      self.target_cross_entropy_loss_weights_2 = weights_2  # Used in evaluation.
      self.target_cross_entropy_loss_weights_3 = weights_3  # Used in evaluation.
      self.target_cross_entropy_loss_weights_4 = weights_4  # Used in evaluation.

  def setup_inception_initializer(self):
    if self.mode != "inference":
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    global_step = tf.Variable(initial_value=0, name="global_step", trainable=False, collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    self.global_step = global_step

  def build(self):
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()
