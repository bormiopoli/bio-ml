import keras
import pandas as pd
from keras import Sequential
from keras.layers import RNN
from keras.initializers import GlorotNormal
from keras.optimizers import Adam
from keras import callbacks
from data_utils import add_label, compute_fft, compute_kaiman
from ml_utils import r_squared
import tensorflow as tf
from keras_multi_head import MultiHeadAttention
from tensorflow_addons.rnn import PeepholeLSTMCell
import glob, os
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')
batch_size = 2* 24 #* 60
num_epochs = 1200
initializer = GlorotNormal(seed=42)


log_dir = "logs/fit/" + "biofreq"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


class CombinedLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, max_lr, increase_factor=2, patience=num_epochs//15, plateau_patience=num_epochs//20, min_lr=1e-6):
        super().__init__()
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.increase_factor = increase_factor
        self.patience = patience
        self.plateau_patience = plateau_patience
        self.min_lr = min_lr
        self.wait = 0
        self.plateau_wait = 0
        self.best_loss = float('inf')
        self.best_weights = None

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.initial_lr)

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')

        # Check for improvement
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0  # Reset wait for increasing lr
            self.plateau_wait = 0  # Reset plateau wait
            self.best_weights = self.model.get_weights()  # Save best weights
        else:
            self.wait += 1

            # Increase learning rate if no improvement for 'patience'
            if self.wait >= self.patience:
                new_lr = min(self.model.optimizer.learning_rate * self.increase_factor, self.max_lr)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                print(f"Learning rate increased to: {new_lr:.6f}")
                self.wait = 0  # Reset wait counter

            # Reduce learning rate if no improvement for 'plateau_patience'
            elif self.plateau_wait >= self.plateau_patience:
                new_lr = max(self.model.optimizer.learning_rate / (self.increase_factor/1.5), self.min_lr)
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                print(f"Learning rate reduced to: {new_lr:.6f}")
                self.plateau_wait = 0  # Reset plateau wait

                # Early stopping condition
                if self.plateau_wait >= self.plateau_patience:
                    print("Early stopping triggered.")
                    self.model.set_weights(self.best_weights)  # Restore best weights
                    self.model.stop_training = True  # Stop training

            # Update plateau wait
            self.plateau_wait += 1


combined_lr_scheduler = CombinedLRScheduler(patience=num_epochs//20, plateau_patience=num_epochs//25, initial_lr=0.0005, max_lr=0.01)
earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                        mode ="min", patience = (num_epochs//10)+4,
                                        restore_best_weights = True)
optimizer = Adam(learning_rate=0.0005)

# ALTERNATIVE TRAINING SETUP
# reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
#                               patience=(num_epochs//20)+3, min_lr=0.000000000001)


csv_files = glob.glob(os.path.join("/home/daltonik/Desktop/bio_frequencies_ml/data/", "*.csv"), recursive=True)

# Step 3: Iterate over the list of CSV files and read each one into a DataFrame
df_list = []
for file in csv_files:
    try:
        initial_data = pd.read_csv(file, skiprows=1)
        initial_data['datetime'] = initial_data['Date'] + ' ' + initial_data['Time']
        initial_data['datetime'] = pd.to_datetime(initial_data['datetime'], format='%m-%d-%y %H:%M:%S')
        initial_data = initial_data.set_index('datetime')
        initial_data = initial_data.asfreq('s')
        initial_data = initial_data.drop(columns=['Date', 'Time'])

        initial_data.drop(['sec'], axis=1, inplace=True)
        mapped_columns = ['recovery_after_flavescence', 'leaves_and_fruit_past_recovery', 'branch1',
                          'leaves_and_fruit_in_recovery1', 'flavescence_symptoms_ongoing', 'branch2',
                          'leaves_and_fruit_in_recovery2', 'leaves_and_fruit_wealthy']
        initial_data.columns = mapped_columns
        initial_data = initial_data[~initial_data.index.duplicated(keep=False)]

        # initial_data = initial_data[['leaves_and_fruit_in_recovery1', 'leaves_and_fruit_wealthy', 'recovery_after_flavescence']]

        initial_data = initial_data.resample(
            'h').median()  # You can replace mean() with another aggregation function as needed

        initial_data = compute_fft(initial_data)
        # initial_data = compute_kaiman(initial_data)

        if len(initial_data)>0:
            df_list.append(initial_data)

    except Exception as err:
        print(err)


meteo_data = pd.read_excel("/home/daltonik/Desktop/bio_frequencies_ml/data/Export_Hourly_Adorno_-_Cascina_01_01_2023_18_12_2023.xls")
meteo_data['datetime'] = meteo_data['Data'] + ' ' + meteo_data['Ora']
meteo_data['datetime'] = pd.to_datetime(meteo_data['datetime'], format='%d/%m/%Y %H:%M:%S')
meteo_data = meteo_data.set_index('datetime')
meteo_data = meteo_data.drop(columns=['Data', 'Ora'])


# chosen_column = 'Umidità [ % ]'
chosen_column = 'Temperatura[ °C ]'
# chosen_column = 'Pluviometro[ mm ]'

train_x, test_x, train_y, test_y, y_scaler, index_test_y = add_label(df_list, meteo_data, batch_size, chosen_column=chosen_column)


model = Sequential()

# model = Sequential([
#     Conv1D(32, 3, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])),  # 24 timesteps, 5 features
#     BatchNormalization(),
#     LSTM(32, return_sequences=True),
#     LSTM(32),
#     Dense(32, activation='selu'),
#     Dropout(0.3),
#     Dense(train_x.shape[1], activation='linear')
# ])


model = tf.keras.Sequential([

    # tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(1, n_features)),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Bidirectional(RNN(PeepholeLSTMCell(units=len(initial_data.columns)  * 2, activation='relu',
                          dropout=0.25, recurrent_dropout=0.2, bias_initializer='zeros', kernel_initializer=initializer
                          ), return_sequences=True, name='RNN_Peephole_LSTM_cells')),
    # tf.keras.layers.Conv1D(filters=3, kernel_size=14, padding='same',
    #                      kernel_initializer=initializer,
    #               bias_initializer=keras.initializers.Zeros(), name='Conv1D'),
    MultiHeadAttention(2, name='attention_layer'),
    tf.keras.layers.Dense(32, activation='selu', kernel_initializer=initializer, bias_initializer='zeros'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer, bias_initializer='zeros')
])

model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', r_squared])

history = model.fit(train_x, train_y,
            epochs=num_epochs, batch_size=batch_size, validation_data=(test_x, test_y), verbose=2,
            shuffle=False, callbacks=[
        earlystopping,
        combined_lr_scheduler,
        # reduce_lr,
        tensorboard_callback
    ]
                    # , class_weight=class_weights
                    )


# Step 1: Build predicted DataFrame
yhat = model.predict(test_x, batch_size=batch_size)[:, -1]
yhat = pd.DataFrame.from_dict(yhat.flatten())
yhat['index'] = index_test_y[:, -1]
yhat.set_index("index", inplace=True)

# Step 2: Build observed DataFrame
y_obs = pd.DataFrame.from_dict(test_y[:, -1].flatten())
y_obs['index'] = index_test_y[:, -1]
y_obs.set_index("index", inplace=True)

# Step 3: Convert index to datetime and sort both DataFrames
yhat.index = pd.to_datetime(yhat.index)
y_obs.index = pd.to_datetime(y_obs.index)
yhat.sort_index(inplace=True)
y_obs.sort_index(inplace=True)

# Step 4: Convert index to string after sorting
yhat.index = yhat.index.astype(str)
y_obs.index = y_obs.index.astype(str)

# Step 5: Concatenate the two DataFrames
h = pd.concat([yhat, y_obs], axis=1, ignore_index=True)
h.columns = [f"Predicted {'temperature' if chosen_column == 'Temperatura[ °C ]'  else 'umidity'}",
             f"Observed {'temperature' if chosen_column == 'Temperatura[ °C ]'  else 'umidity'}"]

width_px = 1400
height_px = 1000
dpi = 100  # dots per inch
figsize = (width_px / dpi, height_px / dpi)
h.plot(figsize=figsize)
plt.xlabel("Time")  # X-axis label
plt.ylabel(f"Standardised {'temperature' if chosen_column == 'Temperatura[ °C ]'  else 'umidity'}", fontsize=20)  # Y-axis label
plt.title(f"Predicted vs Observed {'temperature' if chosen_column == 'Temperatura[ °C ]'  else 'umidity'}", fontsize=20)
plt.xticks(fontsize=18, rotation=60)  # Increase x-axis tick label font
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig(f"/home/daltonik/Desktop/bioML_R1/bio-ml/images/predicted_{'temperature' if chosen_column == 'Temperatura[ °C ]'  else 'umidity'}")
model.save(f'bioml_{batch_size}_{chosen_column[:6]}')
