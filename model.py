
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)


class ElmoEmbeddingLayer(layers.Layer):
    def __init__(self,
                 input_is_tokenized=True,
                 sentence_level=False,
                 **kwargs):
        self.output_dim = 1024
        self.trainable = True
        self.signature = "tokens" if input_is_tokenized else "default"
        self.elmo_output_key = "default" if sentence_level else "elmo"
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2',
                               trainable=self.trainable,
                               name="{}_module".format(self.name))
        self.trainable_weights += tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, seq_lengths=None):
        if self.signature == "tokens":
            inputs = {
                "tokens": x,
                "sequence_len": seq_lengths
            }
        else:
            inputs = x
        result_dict = self.elmo(inputs, as_dict=True, signature=self.signature)
        return result_dict[self.elmo_output_key]

    def compute_mask(self, inputs, mask=None):
        return tf.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_shape


class QQPairsModel(tf.keras.Model):

    def __init__(self):
        super(QQPairsModel, self).__init__(name='qqp_model')
        # Define forward pass
        pass

    def call(self, inputs, **kwargs):
        pass

    def compute_output_shape(self, input_shape):
        pass

