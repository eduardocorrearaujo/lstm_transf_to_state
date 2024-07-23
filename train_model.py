import numpy as np
import pandas as pd
import preprocess_data as prep
from keras.optimizers import Adam
from models import build_bi_lstm, train_model_using_cross_val, build_baseline, make_predictions, build_lstm 
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

'''
This script is used to train the model for a specific STATE and forecast the cases on a 
specific year (TEST_YEAR). The model is trained  with the regional health data before the year selected. 
'''

STATE = 'MG'
TEST_YEAR = 2023

df = prep.load_cases_data()
df = df.loc[df.uf == STATE]
enso = prep.load_enso_data()

# generate the samples to train and test based on the regional data 
X_train, y_train = prep.generate_regional_train_samples(df, enso, TEST_YEAR)

LOSS = 'msle'
batch_size = 4
model_name = 'baseline'

model = build_baseline(hidden=64, features=5, predict_n=52, look_back=52, loss =LOSS, optimizer = 'adam',  stateful = False, batch_size = batch_size)

model = train_model_using_cross_val(model, X_train, y_train, n_splits=4, epochs = 150,
                                verbose = 0,
                                batch_size = batch_size, 
                                monitor = 'val_loss',
                                min_delta = 0,
                                patience = 20)


# Retreinando com os dados do estado e aplicando para o ano de teste
df_w = prep.aggregate_data(df)

data = df_w.merge(enso[['enso']], left_index = True, right_index = True)

X_train, y_train, norm_values = prep.get_train_data(data.loc[data.year < TEST_YEAR])

model.compile(loss=LOSS, optimizer = Adam(learning_rate = 0.0001), metrics=["accuracy", "mape", "mse"])

TB_callback = TensorBoard(
            log_dir="./tensorboard",
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            # embeddings_freq=10
        )

hist = model.fit(
            X_train,
            y_train,
            batch_size=1,
            epochs=25,
            verbose=0,
            callbacks=[TB_callback]
        )

# save the model
model.save(f'saved_models/model_{STATE}_{TEST_YEAR-1}_{model_name}.keras')

# retreinando os modelos usando os dados de 2023
#regional samples
X_train_reg, y_train_reg = prep.generate_regional_train_samples(df, enso, TEST_YEAR, min_year = 2023)

# state samples
df_w = prep.aggregate_data(df)

data = df_w.merge(enso[['enso']], left_index = True, right_index = True)

X_train_state, y_train_state, norm_values = prep.get_train_data(data.loc[data.year < TEST_YEAR], min_year = 2023)

X_test, y_test = prep.get_test_data(norm_values,data, year = TEST_YEAR)

X_train = np.append(X_train_reg, X_train_state, axis = 0)

y_train = np.append(y_train_reg, y_train_state, axis = 0)

model.compile(loss=LOSS, optimizer = Adam(learning_rate = 0.0001), metrics=["accuracy", "mape", "mse"])

TB_callback = TensorBoard(
            log_dir="./tensorboard",
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            # embeddings_freq=10
        )

hist = model.fit(
            X_train,
            y_train,
            batch_size=4,
            epochs=100,
            verbose=0,
            callbacks=[TB_callback, EarlyStopping(monitor='loss', min_delta=0, patience=20)]
        )

# save the model
model.save(f'saved_models/model_{STATE}_{TEST_YEAR-1}_{model_name}.keras')
