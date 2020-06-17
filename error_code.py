from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras.layers import LSTM, TimeDistributed
from keras.layers import Bidirectional
from keras_contrib.layers import CRF

import tensorflow as tf
import tensorflow_hub as hub

######################################MAKE ELMO EMBEDDING LAYER##############################################
class ElmoEmbeddingLayer(Layer):
  def __init__(self, **kwargs):
      self.dimensions = 1024
      self.token_length = 10*[1000]
      self.trainable = True
      super().__init__(**kwargs)

  def build(self, input_shape):
      self.elmo = hub.Module('https://tfhub.dev/google/elmo/3', trainable=self.trainable,
                            name="{}_module".format(self.name))
      
      #self._trainable_weights += tf.compat.v1.trainable_variables(scope="^{}_module/.*".format(self.name))

      super().build(input_shape)

  def call(self, x, mask=None):
      result = self.elmo(
                        inputs={
                            "tokens": tf.squeeze(tf.cast(x, tf.string)),
                            "sequence_len": self.token_length
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]
      return result

  
  # def compute_mask(self, inputs, mask=None):
  #   return K.not_equal(inputs, '__PAD__')

  def compute_output_shape(self, input_shape):
      return (input_shape[0], 1000, self.dimensions)
##############################################################################################


######################################MAKE MODEL##############################################
sequence_input = Input(shape=(1000,), dtype = tf.string) #đầu vào tương ứng 1000 từ (đã pad)
embedded_sequences = ElmoEmbeddingLayer()(sequence_input)
x = Bidirectional(LSTM(units=512, return_sequences=True, dropout = 0.1))(embedded_sequences)

dense = TimeDistributed(Dense(100, activation="relu"))(x) 

crf = CRF(3) #đầu ra là vector tương ứng 3 tags B-ENT, I-ENT, O

crf_layer = crf(dense)

model = Model(sequence_input, crf_layer)
##############################################################################################


#############################COMPILE MODEL####################################################
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy, crf_accuracy
model.compile(loss = crf_loss,
              optimizer='rmsprop',
              metrics=[crf_viterbi_accuracy])
##############################################################################################


###########################TRAINING###########################################################
history = model.fit(X_tr, y_tr, batch_size = 10, epochs = 3)
