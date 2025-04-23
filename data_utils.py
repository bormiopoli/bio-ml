import pandas as pd
from pandas import concat, DataFrame
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.utils import class_weight
import numpy as np


def series_to_supervised(data, n_in=1, n_out=1, dropnan=False):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def compute_class_weights(train_y, batch_size, class_weight_instance=class_weight):
    num_batches = train_y.shape[0] // batch_size + (train_y.shape[0] % batch_size > 0)
    class_counts = {}

    for i in range(num_batches):
        # Get the current batch
        start_index = i * batch_size
        end_index = min(start_index + batch_size, train_y.shape[0])
        batch_y = train_y[start_index:end_index]

        # Reshape for computing class weights
        batch_y_reshaped = batch_y.reshape(-1)

        # Compute class weights for the current batch
        class_weights = class_weight_instance.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(batch_y_reshaped),
            y=batch_y_reshaped
        )

        # Aggregate class weights
        for j, cls in enumerate(np.unique(batch_y_reshaped)):
            if cls in class_counts:
                class_counts[cls].append(class_weights[j])
            else:
                class_counts[cls] = [class_weights[j]]

    # Compute the mean class weight for each class
    aggregated_class_weights = {cls: np.mean(weights) for cls, weights in class_counts.items()}

    return aggregated_class_weights


def compute_kaiman(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from filterpy.kalman import KalmanFilter

    initial_covariance = df.cov().values

    measurement_diffs = df.diff().dropna()
    # Estimate measurement noise covariance matrix R
    R = np.cov(measurement_diffs, rowvar=False)
    smoothed_data = df.rolling(window=3).mean().dropna()
    # Calculate differences in the smoothed data to estimate process noise
    process_diffs = smoothed_data.diff().dropna()
    # Estimate process noise covariance matrix Q
    Q = np.cov(process_diffs, rowvar=False)

    # Initialize Kalman Filter for eight signals
    kf = KalmanFilter(dim_x=len(df.columns), dim_z=len(df.columns))
    kf.x = np.zeros(len(df.columns))  # Initial state (zeros for all signals)
    kf.F = np.eye(len(df.columns))  # State transition matrix (identity matrix)
    kf.H = np.eye(len(df.columns))  # Measurement function (identity matrix)
    kf.P = initial_covariance  # Covariance matrix (sample covariance)
    kf.R = R  # Measurement noise (scaled sample covariance)
    kf.Q = Q  # Process noise (scaled sample covariance)

    np_df = df.values
    # Apply Kalman filter to each row in the DataFrame
    filtered_values = np.zeros_like(np_df)
    for i in range(len(np_df)):
        kf.predict()
        kf.update(np_df[i])
        filtered_values[i] = kf.x

    df = pd.DataFrame(filtered_values, columns=df.columns, index=df.index)
    return df


def butter_lowpass_filter(data, cutoff, fs, order=5):
    from scipy.signal import butter, filtfilt
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def compute_fft(df):
    # Apply FFT
    import matplotlib
    matplotlib.use("TkAgg")
    import numpy as np
    import matplotlib.pyplot as plt
    for column in df.columns:
        n = len(df[column])
        # fft_values = np.fft.fft(df[column])
        fft_freq = np.fft.fftfreq(n, d=60.0)  # d is the sampling interval (1 second)

        # Only keep the positive frequencies and corresponding values
        positive_freqs = fft_freq[:n // 2]
        # positive_fft_values = np.abs(fft_values[:n // 2])

        # Sampling frequency (1 hour = 1/3600 Hz)
        fs = 1 / 60.0  # Hz
        cutoff = pd.Series(positive_freqs).quantile(0.8)  # Cutoff frequency in Hz

        # Apply the low-pass filter to the original signal
        filtered_signal = butter_lowpass_filter(df[column].values, cutoff, fs, order=2)
        # Plot the frequency spectrum
        # plt.figure(figsize=(12, 6))
        # plt.plot(df[column].values, filtered_signal)
        # plt.title('Filtered with lowpass')
        # plt.xlabel('Observation')
        # plt.ylabel('dV')
        # plt.grid(True)
        # plt.show()
        df[column] = filtered_signal

    return df

def split_data(train_x, train_y, indexes, factor=8):
    test_x = train_x[factor*(train_x.shape[0]//10):]
    train_x = train_x[:factor*(train_x.shape[0]//10)]
    test_y = train_y[factor*(train_y.shape[0]//10):]
    train_y = train_y[:factor*(train_y.shape[0]//10)]

    index_test_y = indexes[-test_y.flatten().shape[0]:]
    index_train_y = indexes[:train_y.flatten().shape[0]]

    return train_x, test_x, train_y, test_y, index_train_y, index_test_y


def add_label(dfs, fun_meteo_data, batch_size):

    trains_y = []
    trains_x = []
    tests_y = []
    tests_x = []
    indices = []
    fun_meteo_data = fun_meteo_data[~fun_meteo_data.index.duplicated(keep=False)]
    fun_meteo_data = fun_meteo_data.asfreq("h")
    # fun_meteo_data = fun_meteo_data.resample('min').interpolate(method='time')
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()

    for df in dfs:
        beginning = df.index[0];
        end = df.index[-1];
        df = df[~df.index.duplicated(keep=False)]
        temp_meteo_data = fun_meteo_data[(fun_meteo_data.index >= beginning) & (fun_meteo_data.index <= end)]
        complete_data = pd.concat([df, temp_meteo_data], axis=1)
        complete_data.sort_index(inplace=True)
        complete_data = complete_data.interpolate(method='time')
        complete_data.dropna(how='any', inplace=True)

        chosen_column =  'Temperatura[ °C ]'
        # chosen_column = 'Anemometro[ km/h ]'
        columns_to_remove = ['Pluviometro[ mm ]', 'Umidità [ % ]', 'Anemometro[ km/h ]',
           'Bagnatura fogliare stimata[ Bagnatura ]',
           'Bagnatura fogliare[ Bagnatura ]', 'T Bulbo umido[ °C ]']
        # columns_to_remove = []
        complete_data = complete_data[pd.Index(
            sorted([col for col in complete_data.columns if chosen_column not in col and col not in columns_to_remove])  +  [col for col in complete_data.columns if
                                                                          chosen_column in col])]

        # WE MAKE A BATCH SERIES OF BATCH_SIZE LENGTH FOR EACH POINT OF THE SUBDATASET
        num_samples = len(complete_data) - batch_size
        X = []
        Y = []
        scaled_data = scaler.fit_transform(complete_data)
        complete_data = pd.DataFrame(scaled_data, columns=complete_data.columns)

        for i in range(num_samples):

            x = complete_data[complete_data.columns[:-1]].iloc[i:i + batch_size].values
            y = complete_data[complete_data.columns[-1]].iloc[i:i+batch_size].values
            # x = scaler.fit_transform(x)
            # y = scaler.fit_transform(y.reshape(-1, 1)).reshape(1, -1)

            x = x.reshape(1, x.shape[0], x.shape[1])

            X.append(x)
            Y.append(y.reshape(1, -1))

        # batch_size = 1
        try:

            X = np.concatenate(X)
            Y = np.concatenate(Y)

            train_x = X[:(X.shape[0] // 10) * 8]
            test_x = X[(X.shape[0] // 10) * 8:]

            train_y = Y[:(Y.shape[0] // 10) * 8]
            test_y = Y[(Y.shape[0] // 10) * 8 :]

            trains_y.append(train_y)
            trains_x.append(train_x)
            tests_x.append(test_x)
            tests_y.append(test_y)

            indices.append(complete_data.index[-(len(complete_data.index) // batch_size) * batch_size:])


        except Exception as err:
            print("Not enough length for test or train data   ", err)

    trains_x = np.concatenate(trains_x)
    trains_y = np.concatenate(trains_y)
    tests_x = np.concatenate(tests_x)
    tests_y = np.concatenate(tests_y)

    return trains_x, tests_x, trains_y, tests_y, scaler


# def add_label_categorise(df, fun_meteo_data, batch_size, is_tf=True):
#
#     if type(df) == list:
#         df = pd.concat(df)
#         df = df.sort_index()
#
#     trains_y_l = []
#     trains_x_l = []
#     indices = []
#     fun_meteo_data = fun_meteo_data[~fun_meteo_data.index.duplicated(keep=False)]
#     fun_meteo_data = fun_meteo_data.asfreq("h")
#     # fun_meteo_data = fun_meteo_data.resample('min').interpolate(method='time')
#
#     beginning = df.index[0];
#     end = df.index[-1];
#     df = df[~df.index.duplicated(keep=False)]
#     temp_meteo_data = fun_meteo_data[(fun_meteo_data.index >= beginning) & (fun_meteo_data.index <= end)]
#     complete_data = pd.concat([df, temp_meteo_data], axis=1)
#     complete_data.sort_index(inplace=True)
#     complete_data = complete_data.interpolate(method='time')
#     complete_data.dropna(how='any', inplace=True)
#
#     chosen_column = 'Temperatura[ °C ]'
#     # chosen_column = 'Anemometro[ km/h ]'
#     columns_to_remove = ['Pluviometro[ mm ]', 'Umidità [ % ]', 'Anemometro[ km/h ]',
#                          'Bagnatura fogliare stimata[ Bagnatura ]',
#                          'Bagnatura fogliare[ Bagnatura ]', 'T Bulbo umido[ °C ]']
#     # columns_to_remove = []
#     complete_data = complete_data[pd.Index(
#         sorted([col for col in complete_data.columns if chosen_column not in col and col not in columns_to_remove]) + [
#             col for col in complete_data.columns if
#             chosen_column in col])]
#
#     for i, column in enumerate(df.columns):
#         if 'Temperatura[ °C ]' not in column:
#             new_df = pd.DataFrame()
#             new_df['signal'] = complete_data[column]
#             new_df['temp'] = complete_data['Temperatura[ °C ]']
#             new_df['y'] = str(column)
#
#             original_columns = new_df.columns
#
#             if is_tf:
#
#                 previous_steps = 48
#                 reframed = series_to_supervised(new_df, n_in=previous_steps, dropnan=True)
#                 # test_dfs = series_to_supervised(test_dfs, n_in=previous_steps, dropnan=True)
#
#                 for j in range(0, previous_steps + 1, 1):
#                     if j != 0:
#                         reframed.drop([f'var{len(original_columns)}(t{f"-{j}" if j > 0 else ""})'], axis=1, inplace=True)
#                         # test_dfs.drop([f'var1(t{f"-{i}" if i > 0 else ""})'], axis=1, inplace=True)
#                         # train_dfs.drop([f'var{len(original_columns)}(t{f"-{i}" if i > 0 else ""})'], axis=1, inplace=True)
#                         # train_dfs.drop([f'var1(t{f"-{i}" if i > 0 else ""})'], axis=1, inplace=True)
#
#                 reframed = reframed[pd.Index(
#                     [col for col in reframed.columns if f'var{len(original_columns)}(t)' not in col] + [col for col in
#                                                                                                         reframed.columns if
#                                                                                                         re.match(
#                                                                                                             f'var{len(original_columns)}' + r'\(t\)',
#                                                                                                             col)])]
#             else:
#                 reframed = new_df
#
#             train_dfs = reframed.dropna()
#
#             train_x = train_dfs.iloc[:, :-1].to_numpy()
#             train_x = train_x[:(len(train_x)//batch_size) * batch_size]
#
#             train_y = train_dfs.iloc[:, -1].to_numpy()
#             train_y = train_y[: (len(train_y)//batch_size)*batch_size]
#
#             train_x = train_x.reshape(train_x.shape[0]//batch_size, batch_size, -1)
#             train_y = train_y.reshape(train_y.shape[0]//batch_size, batch_size)
#
#             trains_x_l.append(train_x);  trains_y_l.append(train_y)
#
#             indices.append(complete_data.index[-(len(complete_data.index) // batch_size) * batch_size:])
#
#     trains_x = np.concatenate(trains_x_l)
#     trains_y = np.concatenate(trains_y_l)
#
#     label_encoder = LabelEncoder()
#     shape0 = trains_y.shape[0]; shape1 = trains_y.shape[1]
#     trains_y = label_encoder.fit_transform(trains_y.flatten())
#
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     trains_x = scaler.fit_transform(trains_x.reshape(-1, 1))
#     trains_x = trains_x.reshape(shape0, shape1, -1)
#     trains_y = scaler.fit_transform(trains_y.reshape(shape0, shape1))
#
#     indices = np.concatenate(indices)
#
#     trains_x, tests_x, trains_y, tests_y, index_train_y, index_y = split_data(trains_x, trains_y, indices)
#
#     return trains_x, tests_x, trains_y, tests_y, index_train_y, index_y
