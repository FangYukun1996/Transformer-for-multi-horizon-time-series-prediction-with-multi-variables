"""
## Reference:
   1. Transformer元素中class：https://keras.io/examples/audio/transformer_asr/
   2. Transformer的构建：https://keras.io/examples/nlp/neural_machine_translation_with_transformer/
   3. PositionalEncoding：https://blog.csdn.net/xm961217/article/details/107787737
   4. 产生训练数据时需要padding的思路来自Informer，不然Train不起来，就是要求等长：
      https://arxiv.org/abs/2012.07436
   5. Learning Rate Schedule: https://keras.io/examples/audio/transformer_asr/
      此外，如果想保存模型，自定义的Learning Rate Schedule还得override ‘get_config’：
      https://stackoverflow.com/questions/61557024/notimplementederror-learning-rate-schedule-must-override-get-config
   6. 自定义的Layer，如果想保存模型要override ‘get_config’：
      https://blog.csdn.net/qq_39269347/article/details/111464049
   7. 自定义的Layer，按照6确实是可以保存了，但是很多时候根本就load不了，尤其是自定义的东西特别多的情况。
      面对这种情况，就保存权重(save_weights())和训练模型时的一些配置参数(建立一个参数字典dict，然后np.save('xxx.npy',dict)),
      测试时把模型原模原样搭建一遍，加载这些参数和权重就可以了
      https://blog.csdn.net/qq_37644877/article/details/95722651
   8. 测试的时候，需求是实时预测，就是说给一个输入就给一个输出，但是keras的model.predict()默认batch_size为32，
      所以测试的时候调用predict()时，置batch_size=1
      https://keras.io/api/models/model_training_apis/#predict-method
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Generate Training Data for Time series
# non-padding edition. It do not work in the training
# def generate_training_examples(sequence,input_step,pred_stride):
#     n_patterns=len(sequence)-input_step-pred_stride+2
#     X, y = list(), list()
#     for i in range(n_patterns-1):
#         X.append(sequence[i:i+input_step])
#         y.append(sequence[i+input_step:i+input_step+pred_stride])
#     X_enc = np.array(X)
#     X_dec = X_enc
#     y = np.array(y)
#     y = y[:,:,[0,1]]
#
#     return X_enc, X_dec,y

# padding edition
def generate_examples(sequence,input_step,pred_stride):
    n_patterns=len(sequence)-input_step-pred_stride+2
    X, y = list(), list()
    for i in range(n_patterns-1):
        X.append(sequence[i:i+input_step])
        y.append(sequence[i+pred_stride:i+pred_stride+input_step])
    X_enc = np.array(X)
    X_dec = np.zeros(np.shape(X_enc)) #padding
    X_dec[:,0:input_step-pred_stride,:]=X_enc[:,pred_stride:,:]
    y = np.array(y)
    y = y[:,:,[0,1]]

    return X_enc, X_dec,y


# Construct the TransformerEncoder
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1,**kwargs):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"att": self.att, "ffn": self.ffn,"layernorm1":self.layernorm1,
                  "layernorm2":self.layernorm2,"dropout1":self.dropout1,"dropout2":self.dropout2}
        base_config = super(TransformerEncoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Construct the TransformerDecoder
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1,**kwargs):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(rate)
        self.ffn_dropout = layers.Dropout(rate)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def call(self, enc_out, target):
        target_att = self.self_att(target, target)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"layernorm1":self.layernorm1,"layernorm2":self.layernorm2,"layernorm3":self.layernorm3,
                  "self_att":self.self_att,"enc_att":self.enc_att,"self_dropout":self.self_dropout,
                  "enc_dropout":self.enc_dropout,"ffn_dropout":self.ffn_dropout,"ffn":self.ffn}
        base_config = super(TransformerDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# Define the way for positional encoding
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_len=None, embedding_dim=None, **kwargs):
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        super(PositionalEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        if self.embedding_dim == None:
            self.embedding_dim = int(inputs.shape[-1])

        position_embedding = np.array([
            [pos / np.power(10000, 2. * i / self.embedding_dim) for i in range(self.embedding_dim)]
            for pos in range(self.sequence_len)])

        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  # dim 2i
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])  # dim 2i+1
        position_embedding = np.expand_dims(position_embedding,axis=0)
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        return position_embedding

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"sequence_len": self.sequence_len, "embedding_dim": self.embedding_dim}
        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# Define the Schedule for training
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        init_lr=0.00001,
        lr_after_warmup=0.001,
        final_lr=0.00001,
        warmup_epochs=15,
        decay_epochs=85,
        steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        warmup_lr = (
            self.init_lr
            + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - self.warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / (self.decay_epochs),
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"init_lr": self.init_lr, "lr_after_warmup": self.lr_after_warmup,
                  "final_lr":self.final_lr,"warmup_epochs":self.warmup_epochs,
                  "decay_epochs":self.decay_epochs,"steps_per_epoch":self.steps_per_epoch}

        return config
