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

"""Model wrapper class for performing inference with the ContextualizeShowAndTellModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import show_and_tell_model
import inference_wrapper_base


class InferenceWrapper(inference_wrapper_base.InferenceWrapperBase):
  """Model wrapper class for performing inference with the ContextualizeShowAndTellModel."""

  def __init__(self):
    super(InferenceWrapper, self).__init__()

  def build_model(self, model_config):
    model = show_and_tell_model.ShowAndTellModel(model_config, mode="inference")
    model.build()
    return model

  def feed_image(self, sess, encoded_images):
    initial_state_0, initial_state_1, initial_state_2, initial_state_3, initial_state_4 = sess.run(fetches=["decoder/decoder_0/initial_state_0:0", "decoder/decoder_1/initial_state_1:0", "decoder/decoder_2/initial_state_2:0", "decoder/decoder_3/initial_state_3:0", "decoder/decoder_4/initial_state_4:0"], feed_dict={"image_feed_0:0": encoded_images[0], "image_feed_1:0" : encoded_images[1], "image_feed_2:0" : encoded_images[2], "image_feed_3:0" : encoded_images[3], "image_feed_4:0" : encoded_images[4]})
    return initial_state_0, initial_state_1, initial_state_2, initial_state_3, initial_state_4

  def inference_step(self, sess, input_feed, state_feed, order):
    if order == 0:
      softmax_output, state_output = sess.run(fetches=["softmax_0:0", "decoder/decoder_0/state_0:0"], feed_dict={"input_feed_0:0": input_feed,"decoder/decoder_0/state_feed_0:0": state_feed,})
    elif order == 1:
      softmax_output, state_output = sess.run(fetches=["softmax_1:0", "decoder/decoder_1/state_1:0"], feed_dict={"input_feed_1:0": input_feed,"decoder/decoder_1/state_feed_1:0": state_feed,})  
    elif order == 2:
      softmax_output, state_output = sess.run(fetches=["softmax_2:0", "decoder/decoder_2/state_2:0"], feed_dict={"input_feed_2:0": input_feed,"decoder/decoder_2/state_feed_2:0": state_feed,})
    elif order == 3:
      softmax_output, state_output = sess.run(fetches=["softmax_3:0", "decoder/decoder_3/state_3:0"], feed_dict={"input_feed_3:0": input_feed,"decoder/decoder_3/state_feed_3:0": state_feed,})  
    else:
      softmax_output, state_output = sess.run(fetches=["softmax_4:0", "decoder/decoder_4/state_4:0"], feed_dict={"input_feed_4:0": input_feed,"decoder/decoder_4/state_feed_4:0": state_feed,})
    return softmax_output, state_output, None