import keras
import pandas as pd
from keras import Sequential
from keras.layers import RNN, TimeDistributed, Dense, GlobalMaxPool1D, GlobalAveragePooling1D, Flatten, ConvLSTM1D, Bidirectional, Reshape, LSTM, BatchNormalization, Conv1D, Dropout
from keras.initializers import GlorotNormal
import keras.backend as K
from keras.optimizers import Adam
from keras import callbacks
from data_utils import add_label, compute_fft, compute_kaiman
import tensorflow as tf
from keras_multi_head import MultiHeadAttention
from tensorflow_addons.rnn import PeepholeLSTMCell
import glob, os
import matplotlib
import matplotlib.pyplot as plt
from ml_utils import combined_lr_scheduler, earlystopping, NUM_EPOCHS
from ml_utils import r_squared
from keras.models import load_model
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# matplotlib.use('TkAgg')
# batch_size = 2* 24 #* 60
# initializer = GlorotNormal(seed=42)


# log_dir = "logs/fit/" + "biofreq"
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# optimizer = Adam(learning_rate=0.0005)

# ALTERNATIVE TRAINING SETUP
# reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                               patience=(num_epochs//20)+3, min_lr=0.000000000001)


csv_files = glob.glob(os.path.join("/home/daltonik/Desktop/bioML/data/Input/", "*.csv"), recursive=True)

# Step 3: Iterate over the list of CSV files and read each one into a DataFrame
df_list = []
for file in csv_files:
    try:
        initial_data = pd.read_csv(file, skiprows=2)
        initial_data.columns = ["sec","Volt","Volt","Volt","Volt","Volt","Date","Time"]
        initial_data['datetime'] = initial_data['Date'] + ' ' + initial_data['Time']
        initial_data['datetime'] = pd.to_datetime(initial_data['datetime'], format='%m-%d-%y %H:%M:%S')
        initial_data = initial_data.set_index('datetime')
        initial_data = initial_data[~initial_data.index.duplicated(keep=False)]
        initial_data = initial_data.asfreq('s')
        initial_data = initial_data.drop(columns=['Date', 'Time'])

        initial_data.drop(['sec'], axis=1, inplace=True)
        mapped_columns = ['plant1', 'plant2', 'plant3',
                          'plant4', 'plant5']
        initial_data.columns = mapped_columns
        initial_data = initial_data.astype(float)

        # initial_data = initial_data[['leaves_and_fruit_in_recovery1', 'leaves_and_fruit_wealthy', 'recovery_after_flavescence']]

        initial_data = initial_data.resample(
            'h').median()  # You can replace mean() with another aggregation function as needed

        # initial_data = compute_fft(initial_data)
        # initial_data = compute_kaiman(initial_data)

        if len(initial_data)>0:
            df_list.append(initial_data)

    except Exception as err:
        print(err)


df_list = [df_list[0], df_list[1], df_list[6]]
df_tot = pd.concat(df_list, axis=1, ignore_index=False)
df_tot.dropna(axis=0, inplace=True)
df_tot.columns = list(df_list[0].columns.str.replace("plant", "plant_0_"))+ list(df_list[1].columns.str.replace("plant", "plant_1_")) + \
                 list(df_list[2].columns.str.replace("plant", "plant_2_"))

meteo_data = pd.read_csv("/home/daltonik/Desktop/bioML/data/Output/Hourly_Indexes_Oct22.csv", skiprows=1, encoding='latin', sep=";", decimal=',')
meteo_data.columns = ['data', 'temperatura', 'avg_umidit', 'bagnatura_fogliare', 'pluviometro_mm_h', 'temperatura_rugiada']
# meteo_data.columns = ['data','temperatura', 'temp_max', 'temp_min', 'umidit', 'umidit_max', 'umidit_min', 'bagn_foliare', 'pluviometro']

# meteo_data['datetime'] = pd.to_datetime(meteo_data['data'], format='%d/%m/%Y')
meteo_data['datetime'] = pd.to_datetime(meteo_data['data'], format='%d/%m/%Y %H:%M:%S')
meteo_data = meteo_data.set_index('datetime')
meteo_data = meteo_data.drop(columns=['data'])
meteo_data = meteo_data['temperatura']
meteo_data = meteo_data.astype(float)
meteo_data = meteo_data.asfreq('h').interpolate()
meteo_data = pd.concat([df_tot, meteo_data], axis=1, ignore_index=False)
meteo_data.dropna(inplace=True)
y = meteo_data.iloc[:, -1].values
x = meteo_data[['plant_0_1', 'plant_0_2', 'plant_1_2', 'plant_1_3', 'plant_2_4', 'plant_2_5', 'plant_1_5', 'plant_1_4']]  # Adjust threshold
x = x.values
x = MinMaxScaler().fit_transform(x)
x = np.tile(x, (48, 1, 1))
x = x.reshape((x.shape[1], x.shape[0], x.shape[2]))

# train_x, test_x, train_y, test_y, y_scaler = add_label(df_list, meteo_data, batch_size, chosen_column="temperatura", columns_to_remove=
#     ['avg_umidit', 'bagnatura_fogliare', 'pluviometro_mm_h', 'temperatura_rugiada'], training=False)


model = load_model("/home/daltonik/Desktop/bioML/bio-ml/bioml_48_temperature.h5", custom_objects={'MultiHeadAttention': MultiHeadAttention,
                                                                                      'PeepholeLSTMCell': PeepholeLSTMCell,
                                                                                 'r_squared': r_squared})
y_pred = model.predict(x)
y_pred_flat = y_pred[:, -1, :].flatten()
y_std = (max(y)- y) / (max(y) - min(y))
plt.figure(figsize=(12, 6))
plt.plot(y_std, label='Actual')
plt.plot(y_pred_flat, label='Predicted', alpha=0.7)
plt.legend()
plt.title('Actual vs Predicted Over Time')
plt.show()










