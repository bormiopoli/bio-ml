from pandas import concat, DataFrame
import tensorflow as tf
import keras.backend as K
from keras import callbacks

NUM_EPOCHS = 1200


def r_squared(y_true, y_pred):
    """
    Calculate the R-squared (R2) metric.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        R-squared value.
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - (SS_res / (SS_tot + K.epsilon())))


class CombinedLRScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, max_lr, increase_factor=2, patience=NUM_EPOCHS//15, plateau_patience=NUM_EPOCHS//20, min_lr=1e-6):
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


combined_lr_scheduler = CombinedLRScheduler(patience=NUM_EPOCHS//20, plateau_patience=NUM_EPOCHS//25, initial_lr=0.0005, max_lr=0.01)


earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                        mode ="min", patience = (NUM_EPOCHS//10)+4,
                                        restore_best_weights = True)

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


# def compute_class_weights(train_y, batch_size, class_weight_instance=class_weight):
#     num_batches = train_y.shape[0] // batch_size + (train_y.shape[0] % batch_size > 0)
#     class_counts = {}
#
#     for i in range(num_batches):
#         # Get the current batch
#         start_index = i * batch_size
#         end_index = min(start_index + batch_size, train_y.shape[0])
#         batch_y = train_y[start_index:end_index]
#
#         # Reshape for computing class weights
#         batch_y_reshaped = batch_y.reshape(-1)
#
#         # Compute class weights for the current batch
#         class_weights = class_weight_instance.compute_class_weight(
#             class_weight='balanced',
#             classes=np.unique(batch_y_reshaped),
#             y=batch_y_reshaped
#         )
#
#         # Aggregate class weights
#         for j, cls in enumerate(np.unique(batch_y_reshaped)):
#             if cls in class_counts:
#                 class_counts[cls].append(class_weights[j])
#             else:
#                 class_counts[cls] = [class_weights[j]]
#
#     # Compute the mean class weight for each class
#     aggregated_class_weights = {cls: np.mean(weights) for cls, weights in class_counts.items()}
#
#     return aggregated_class_weights