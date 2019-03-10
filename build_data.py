from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os.path
import random
import sys
import threading
import os

import nltk.tokenize
import numpy as np
import tensorflow as tf

reload(sys)
sys.setdefaultencoding("utf8")

tf.flags.DEFINE_string("train_image_dir", "/tmp/train/",
                       "Training image directory.")
tf.flags.DEFINE_string("val_image_dir", "/tmp/val/",
                       "Validation image directory.")
tf.flags.DEFINE_string("test_image_dir", "/tmp/test/",
                       "Test image directory.")

tf.flags.DEFINE_string("train_stories_file", "/tmp/stories_train.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("val_stories_file", "/tmp/stories_val.json",
                       "Validation captions JSON file.")
tf.flags.DEFINE_string("test_stories_file", "/tmp/stories_test.json",
                       "Test captions JSON file.")

tf.flags.DEFINE_string("output_dir", "/tmp/", "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 122,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 16,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 13,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file", "/tmp/word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 1,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

StoryMetadata = namedtuple("StoryMetadata",
                           ["image_id_0", "filename_0", "captions_0", "album_id_0", "story_id_0", "order_0", "caption_id_0",
                           "image_id_1", "filename_1", "captions_1", "album_id_1", "story_id_1", "order_1", "caption_id_1",
                           "image_id_2", "filename_2", "captions_2", "album_id_2", "story_id_2", "order_2", "caption_id_2",
                           "image_id_3", "filename_3", "captions_3", "album_id_3", "story_id_3", "order_3", "caption_id_3",
                           "image_id_4", "filename_4", "captions_4", "album_id_4", "story_id_4", "order_4", "caption_id_4"])

class Vocabulary(object):
  def __init__(self, vocab, unk_id):
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id

class ImageDecoder(object):
  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _int64_feature(value):
  # Wrapper for inserting an int64 Feature into a SequenceExample proto.
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  # Wrapper for inserting a bytes Feature into a SequenceExample proto.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
  # Wrapper for inserting an int64 FeatureList into a SequenceExample proto.
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  # Wrapper for inserting a bytes FeatureList into a SequenceExample proto.
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(story, decoder, vocab):
  # Builds a SequenceExample proto for an image-caption pair.
  # Args:
  #  image: An ImageMetadata object.
  #  decoder: An ImageDecoder object.
  #  vocab: A Vocabulary object.
  # Returns:
  #  A SequenceExample proto.

  # Open images
  encoded_images = []
  for i in range(1, len(story), 7):
  	image = story[i]
  	with tf.gfile.FastGFile(image, "r") as f:
		encoded_image = f.read()
  	# Try to decode images
  	try:
	    decoder.decode_jpeg(encoded_image)
	    encoded_images.append(encoded_image)
    # Damaged images
  	except (tf.errors.InvalidArgumentError, AssertionError):
		print("Skipping file with invalid JPEG data: %s" % image)

  context = tf.train.Features(feature={
      "image/image_id_0": _int64_feature(story.image_id_0),
      "image/data_0": _bytes_feature(encoded_images[0]),
      "image/album_id_0": _int64_feature(story.album_id_0),
      "image/story_id_0": _int64_feature(story.story_id_0),
      "image/order_0": _int64_feature(story.order_0),
      "image/caption_id_0": _int64_feature(story.caption_id_0),

      "image/image_id_1": _int64_feature(story.image_id_1),
      "image/data_1": _bytes_feature(encoded_images[1]),
      "image/album_id_1": _int64_feature(story.album_id_1),
      "image/story_id_1": _int64_feature(story.story_id_1),
      "image/order_1": _int64_feature(story.order_1),
      "image/caption_id_1": _int64_feature(story.caption_id_1),

      "image/image_id_2": _int64_feature(story.image_id_2),
      "image/data_2": _bytes_feature(encoded_images[2]),
      "image/album_id_2": _int64_feature(story.album_id_2),
      "image/story_id_2": _int64_feature(story.story_id_2),
      "image/order_2": _int64_feature(story.order_2),
      "image/caption_id_2": _int64_feature(story.caption_id_2),

      "image/image_id_3": _int64_feature(story.image_id_3),
      "image/data_3": _bytes_feature(encoded_images[3]),
      "image/album_id_3": _int64_feature(story.album_id_3),
      "image/story_id_3": _int64_feature(story.story_id_3),
      "image/order_3": _int64_feature(story.order_3),
      "image/caption_id_3": _int64_feature(story.caption_id_3),

      "image/image_id_4": _int64_feature(story.image_id_4),
      "image/data_4": _bytes_feature(encoded_images[4]),
      "image/album_id_4": _int64_feature(story.album_id_4),
      "image/story_id_4": _int64_feature(story.story_id_4),
      "image/order_4": _int64_feature(story.order_4),
      "image/caption_id_4": _int64_feature(story.caption_id_4),
  })

  # Extract captions
  captions = []
  for i in range(2, len(story), 7):
  	captions.append(story[i])

  # Convert words to ids
  captions_ids = []
  for caption in captions:
  	for c in caption:
  		captions_ids.append([vocab.word_to_id(word) for word in c])

  feature_lists = tf.train.FeatureLists(feature_list={
      "image/caption_0": _bytes_feature_list(captions[0]),
      "image/caption_ids_0": _int64_feature_list(captions_ids[0]),

      "image/caption_1": _bytes_feature_list(captions[1]),
      "image/caption_ids_1": _int64_feature_list(captions_ids[1]),

      "image/caption_2": _bytes_feature_list(captions[2]),
      "image/caption_ids_2": _int64_feature_list(captions_ids[2]),

      "image/caption_3": _bytes_feature_list(captions[3]),
      "image/caption_ids_3": _int64_feature_list(captions_ids[3]),

      "image/caption_4": _bytes_feature_list(captions[4]),
      "image/caption_ids_4": _int64_feature_list(captions_ids[4])
  })
  # Creates a SequenceExample proto, a pair storing information in context and feature_lists
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example


def _process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards):
  # Processes and stores a subset of sequence of images as TFRecord files in one thread.
  # Args:
  #  thread_index: Integer thread identifier within [0, len(ranges)].
  #  ranges: A list of pairs of integers specifying the ranges of the dataset to
  #    process in parallel.
  #  name: Unique identifier specifying the dataset.
  #  images: List of StoryMetadata.
  #  decoder: An ImageDecoder object.
  #  vocab: A Vocabulary object.
  #  num_shards: Integer number of shards for the output files.

  # Each thread produces N shards where N = num_shards / num_threads. For
  # instance, if num_shards = 128, and num_threads = 2, then the first thread
  # would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)
  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in images_in_shard:
      image = images[i]
      # Obtains the SequenceExample proto for each of the image-caption pair in a StoryMetadata
      sequence_example = _to_sequence_example(image, decoder, vocab)
      if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        shard_counter += 1
        counter += 1
      if not counter % 1000:
        print("%s [thread %d]: Processed %d of %d items in thread batch." %
              (datetime.now(), thread_index, counter, num_images_in_thread))
        sys.stdout.flush()

    writer.close()
    print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
  sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
  # Processes a complete data set and saves it as a TFRecord.
  # Args:
  #  name: Unique identifier specifying the dataset.
  #  images: List of StoryMetadata.
  #  vocab: A Vocabulary object.
  #  num_shards: Integer number of shards for the output files.

  # Shuffle the ordering of images. Make the randomization repeatable.
  random.seed(12345)
  random.shuffle(images)

  # Break the images into num_threads batches. Batch i is defined as
  # images[ranges[i][0]:ranges[i][1]].
  num_threads = min(num_shards, FLAGS.num_threads)
  spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a utility for decoding JPEG images to run sanity checks.
  decoder = ImageDecoder()

  # Launch a thread for each batch.
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in range(len(ranges)):
    args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
    t = threading.Thread(target=_process_image_files, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
        (datetime.now(), len(images), name))


def _create_vocab(captions):
  # Creates the vocabulary of word to word_id.
  # The vocabulary is saved to disk in a text file of word counts. The id of each
  # word in the file is its corresponding 0-based line number.
  # Args:
  #  captions: A list of lists of strings.
  # Returns:
  #  A Vocabulary object.

  print("Creating vocabulary.")
  counter = Counter()
  for c in captions:
    counter.update(c)
  print("Total words:", len(counter))

  # Filter uncommon words and sort by descending count.
  word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  # Write out the word counts file.
  with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
  print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

  # Create the vocabulary dictionary.
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab


def _process_caption(caption):
  # Processes a caption string into a list of tonenized words.
  # Args:
  #  caption: A string caption.
  # Returns:
  #  A list of strings; the tokenized caption.

  tokenized_caption = [FLAGS.start_word]
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))
  tokenized_caption.append(FLAGS.end_word)
  return tokenized_caption

def _load_and_process_metadata(captions_file, image_dir):
  # Loads story metadata from a JSON file and processes the captions.
  # Args:
  #  captions_file: JSON file containing story annotations.
  #  image_dir: Directory containing the image files.
  # Returns:
  #  A list of StoryMetadata.

  with tf.gfile.FastGFile(captions_file, "r") as f:
  	caption_data = json.load(f)

  # Extract the filenames.
  id_to_filename = {x["id"] : [x["file_name"], x["album_id"]] for x in caption_data["images"]}
  # Extract the captions. Each image_id is associated with multiple captions.
  id_to_captions = []

  for story in caption_data["annotations"]:
    image_id_0 = story[0]["image_id"]
    caption_0 = story[0]["caption"]
    story_id_0 = story[0]["story_id"]
    order_0 = story[0]["order"]
    album_id_0 = story[0]["album_id"]
    caption_id_0 = story[0]["id"]

    image_id_1 = story[1]["image_id"]
    caption_1 = story[1]["caption"]
    story_id_1 = story[1]["story_id"]
    order_1 = story[1]["order"]
    album_id_1 = story[1]["album_id"]
    caption_id_1 = story[1]["id"]

    image_id_2 = story[2]["image_id"]
    caption_2 = story[2]["caption"]
    story_id_2 = story[2]["story_id"]
    order_2 = story[2]["order"]
    album_id_2 = story[2]["album_id"]
    caption_id_2 = story[2]["id"]

    image_id_3 = story[3]["image_id"]
    caption_3 = story[3]["caption"]
    story_id_3 = story[3]["story_id"]
    order_3 = story[3]["order"]
    album_id_3 = story[3]["album_id"]
    caption_id_3 = story[3]["id"]

    image_id_4 = story[4]["image_id"]
    caption_4 = story[4]["caption"]
    story_id_4 = story[4]["story_id"]
    order_4 = story[4]["order"]
    album_id_4 = story[4]["album_id"]
    caption_id_4 = story[4]["id"]

    id_to_captions.append((image_id_0, caption_0, story_id_0, order_0, album_id_0, caption_id_0, image_id_1, caption_1, story_id_1, order_1, album_id_1, caption_id_1, image_id_2, caption_2, story_id_2, order_2, album_id_2,  caption_id_2, image_id_3, caption_3, story_id_3, order_3, album_id_3, caption_id_3, image_id_4, caption_4, story_id_4, order_4, album_id_4, caption_id_4))

  print("Loaded caption metadata for %d images from %s" %
        (len(id_to_filename), captions_file))

  # Process the captions and combine the data into a list of StoryMetadata.
  print("Processing captions.")
  stories_metadata = []
  num_captions = 0

  for story in id_to_captions:
    image_id_0 = story[0]
    base_filename = id_to_filename[image_id_0][0]
    filename_0 = os.path.join(image_dir, base_filename)
    album_id_0 = id_to_filename[image_id_0][1]
    captions_0 = [_process_caption(story[1])]
    story_id_0 = story[2]
    order_0    = story[3]
    caption_id_0 = story[5]

    image_id_1 = story[6]
    base_filename = id_to_filename[image_id_1][0]
    filename_1 = os.path.join(image_dir, base_filename)
    album_id_1 = id_to_filename[image_id_1][1]
    captions_1 = [_process_caption(story[7])]
    story_id_1 = story[8]
    order_1    = story[9]
    caption_id_1 = story[11]

    image_id_2 = story[12]
    base_filename = id_to_filename[image_id_2][0]
    filename_2 = os.path.join(image_dir, base_filename)
    album_id_2 = id_to_filename[image_id_2][1]
    captions_2 = [_process_caption(story[13])]
    story_id_2 = story[14]
    order_2    = story[15]
    caption_id_2 = story[17]

    image_id_3 = story[18]
    base_filename = id_to_filename[image_id_3][0]
    filename_3 = os.path.join(image_dir, base_filename)
    album_id_3 = id_to_filename[image_id_3][1]
    captions_3 = [_process_caption(story[19])]
    story_id_3 = story[20]
    order_3    = story[21]
    caption_id_3 = story[23]

    image_id_4 = story[24]
    base_filename = id_to_filename[image_id_4][0]
    filename_4 = os.path.join(image_dir, base_filename)
    album_id_4 = id_to_filename[image_id_4][1]
    captions_4 = [_process_caption(story[25])]
    story_id_4 = story[26]
    order_4    = story[27]
    caption_id_4 = story[29]

    stories_metadata.append(StoryMetadata(image_id_0, filename_0, captions_0, album_id_0, story_id_0, order_0, caption_id_0, image_id_1, filename_1, captions_1, album_id_1, story_id_1, order_1, caption_id_1, image_id_2, filename_2, captions_2, album_id_2, story_id_2, order_2, caption_id_2, image_id_3, filename_3, captions_3, album_id_3, story_id_3, order_3, caption_id_3, image_id_4, filename_4, captions_4, album_id_4, story_id_4, order_4,  caption_id_4))

    num_captions += 5

  print("Finished processing %d captions for %d images in %s" %
        (num_captions, len(id_to_filename), captions_file))

  return stories_metadata


def main(unused_argv):
  def _is_valid_num_shards(num_shards):
    # Returns True if num_shards is compatible with FLAGS.num_threads.
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")
  assert _is_valid_num_shards(FLAGS.test_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.test_shards")

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)
  # Load image metadata from caption files.
  vist_train_dataset = _load_and_process_metadata(FLAGS.train_stories_file, FLAGS.train_image_dir)
  vist_val_dataset = _load_and_process_metadata(FLAGS.val_stories_file, FLAGS.val_image_dir)
  vist_test_dataset = _load_and_process_metadata(FLAGS.test_stories_file, FLAGS.test_image_dir)

  train_captions = []
  for story in vist_train_dataset:
  	for c in story.captions_0:
		train_captions.append(c)
	for c in story.captions_1:
		train_captions.append(c)
	for c in story.captions_2:
		train_captions.append(c)
	for c in story.captions_3:
		train_captions.append(c)
	for c in story.captions_4:
		train_captions.append(c)

  # Create vocabulary from the training captions.
  vocab = _create_vocab(train_captions)

  # Process each dataset
  _process_dataset("train", vist_train_dataset, vocab, FLAGS.train_shards)
  _process_dataset("val", vist_val_dataset, vocab, FLAGS.val_shards)
  _process_dataset("test", vist_test_dataset, vocab, FLAGS.test_shards)

if __name__ == "__main__":
  tf.app.run()
