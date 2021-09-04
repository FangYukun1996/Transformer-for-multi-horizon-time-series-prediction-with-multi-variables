import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import Model,Input
from keras.models import load_model
from utils import*

# load the configs of the model
paras=np.load('./model/Transformer_regression_weight_configs.npy',allow_pickle=True)

# Generate the test data
root_dir="./data/"
train_csv = "test5_denoise.csv"
test_csv='test5_denoise_trail.csv'
csv_file_train = os.path.join(root_dir, train_csv)
csv_file_test = os.path.join(root_dir, test_csv)
df_train= pd.read_csv(csv_file_train,usecols=['latitude','longitude',
                                              'gyro_x','gyro_y','gyro_z',
                                              'acc_x','acc_y','acc_z'])
df_test = pd.read_csv(csv_file_test,usecols=['latitude','longitude',
                                             'gyro_x','gyro_y','gyro_z',
                                             'acc_x','acc_y','acc_z'])
data_train=np.array(df_train).astype(np.float) # keep the precision
data_test= np.array(df_test).astype(np.float)

mm=MinMaxScaler()
scaled_training_data=mm.fit_transform(data_train)
data_test=(data_test-mm.data_min_)/(mm.data_max_-mm.data_min_)

input_step=paras.item()['input_step']
pred_stride=paras.item()['pred_stride']
x_enc,x_dec,_=generate_examples(data_test,input_step,pred_stride)

# load model
# model=load_model('./model/Transformer_regression.h5',
#                  custom_objects={'PositionalEncoding': PositionalEncoding,
#                                  'TransformerEncoder': TransformerEncoder,
#                                  'TransformerDecoder': TransformerDecoder})

# embed_dim=20
# num_heads=2
# num_feed_forward=32
# input_shape = x_enc.shape[1:]
# decoder_input_shape=x_dec.shape[1:]
# output_dim=2

embed_dim=paras.item()['embed_dim']
num_heads=paras.item()['num_heads']
num_feed_forward=paras.item()['num_feed_forward']
input_shape = paras.item()['input_shape']
decoder_input_shape=paras.item()['decoder_input_shape']
output_dim=paras.item()['output_dim']

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
model=Model([enc_input,decoder_inputs],decoder_outputs,name="Transformer")
model.load_weights('./model/Transformer_regression_weight.h5')

# predict
# 这里务必注意，由于需求是一个一个预测，需要指定batch_size=1,不然默认batch_size=32，每32个才给出一个预测值
preds=model.predict([x_enc,x_dec],batch_size=1)

preds=preds*[mm.data_max_[0:2]-mm.data_min_[0:2]]+mm.data_min_[0:2]
preds_1=[]
# preds_2=[]
# preds_3=[]
for i in range (preds.shape[0]):
    preds1=preds[i,input_step-pred_stride,:]
    np.reshape(preds1,[1,output_dim])
    preds_1.append(preds1)

    # 多步预测提取数据的示例
    # preds2=preds[i,input_step-pred_stride+1,:]
    # np.reshape(preds2,[1,output_dim])
    # preds_2.append(preds2)
    # preds3=preds[i,input_step-pred_stride+2,:]
    # np.reshape(preds3,[1,output_dim])
    # preds_3.append(preds3)

saved_data = pd.DataFrame(preds_1)
saved_data.to_csv('./predictions/TRIAL.csv',header=None,index=None)

lat_pred=np.array(preds_1)[:,0]
lon_pred=np.array(preds_1)[:,1]
plt.plot(lat_pred)
plt.savefig('./fig/lat_TRIAL.png')
plt.figure()
plt.plot(lon_pred)
plt.savefig('./fig/lon_TRIAL.png')
plt.show()
