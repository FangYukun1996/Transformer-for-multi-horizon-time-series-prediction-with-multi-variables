import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Model,Input
from utils import*

# Generate the training data
root_dir="./data/"
train_csv = "test5_denoise.csv"
csv_file = os.path.join(root_dir, train_csv)
df = pd.read_csv(csv_file,usecols=['latitude','longitude',
                                   'gyro_x','gyro_y','gyro_z',
                                   'acc_x','acc_y','acc_z'])
data=np.array(df).astype(np.float) # keep the precision
mm=MinMaxScaler()
train_data=mm.fit_transform(data)

input_step=30
pred_stride=3
x_enc,x_dec,y_train=generate_examples(train_data,input_step,pred_stride)


# build the model
embed_dim=20
num_heads=2
num_feed_forward=32
input_shape = x_enc.shape[1:]
decoder_input_shape=x_dec.shape[1:]
output_dim=2

# encoder
enc_input=Input(shape=input_shape,name="encoder_inputs")
input_pos_encoding=PositionalEncoding(input_step,embed_dim)(enc_input)
enc_out=TransformerEncoder(embed_dim, num_heads, num_feed_forward)(input_pos_encoding)
encoder=Model(enc_input, enc_out,name="encoder")

# decoder
decoder_inputs = Input(shape=decoder_input_shape,name="decoder_inputs")
encoder_outputs= Input(shape=(30,embed_dim),name="encoder_outputs")
decoder_inputs_pos_encoding = PositionalEncoding(input_step, embed_dim)(decoder_inputs)
dec_out = TransformerDecoder(embed_dim, num_heads, num_feed_forward)(decoder_inputs_pos_encoding,encoder_outputs)
dec_out = layers.Dropout(0.5)(dec_out)
dec_out=layers.Dense(output_dim)(dec_out)
decoder = Model([decoder_inputs,encoder_outputs], dec_out,name='decoder')

# Transformer
decoder_outputs=decoder([decoder_inputs,enc_out])
transformer=Model([enc_input,decoder_inputs],decoder_outputs,name="Transformer")

# Train
loss_fn = losses.MeanSquaredError()
learning_rate = CustomSchedule(
    init_lr=0.001,
    lr_after_warmup=0.01,
    final_lr=0.0001,
    warmup_epochs=5,
    decay_epochs=15,
    steps_per_epoch=100,
)
optimizer = keras.optimizers.Adam(learning_rate)

model=transformer
keras.utils.plot_model(encoder, to_file='./model/Transformer_encoder.png', show_shapes=True)
keras.utils.plot_model(decoder, to_file='./model/Transformer_decoder.png', show_shapes=True)
keras.utils.plot_model(model, to_file='./model/Transformer_regression.png', show_shapes=True)
model.summary()
model.compile(optimizer=optimizer, loss=loss_fn)
callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

history=model.fit(
    [x_enc,x_dec],
    y_train,
    validation_split=0.1,
    epochs=2,
    batch_size=100,
    callbacks=callbacks,
)

# Save the model
model_saved_path='./model/Transformer_regression.h5'
model.save(model_saved_path)
model.save_weights('./model/Transformer_regression_weight.h5')

# Save the model configs
paras={'input_step':input_step,
       'pred_stride':pred_stride,
       'embed_dim':embed_dim,
       'num_heads':num_heads,
       'num_feed_forward':num_feed_forward,
       'input_shape':input_shape,
       'decoder_input_shape':decoder_input_shape,
       'output_dim':output_dim}
np.save('./model/Transformer_regression_weight_configs.npy',paras)

#Plot the model's training and validation loss
metric = "loss"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model MSE loss")
plt.ylabel("MSE loss", fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.savefig('./fig/TS_regression_Transformer_training_loss.png')
plt.show()
plt.close()
