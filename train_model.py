import time
import numpy as np
import pandas as pd
import preprocess_data as prep
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split 
from models import train_model, train_model_using_cross_val, schedule, build_baseline, build_comb_lstm_att, build_lstm_att_3, sum_regions_predictions 
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

'''
This script is used to train the model for a specific STATE and forecast the cases on a 
specific year (TEST_YEAR). The model is trained with the regional health data before the year selected. 
'''

# Load the cases and enso data
df_all = prep.load_cases_data()
enso = prep.load_enso_data()

# flag to decide if the model will be applied or not
apply = False

start_time = time.time()  

for STATE in ['BA']:

    TEST_YEAR = 2023 

    if STATE == 'PR': 

        min_year = 2019

    else: 

        min_year = 2013

    #columns used in the model
    cols_to_norm = ['casos','epiweek', 'enso']#,  'R0', 'total_cases',
                         # 'peak_week', 'perc_geocode'] 

    #cols_to_norm = ['casos','epiweek', 'enso', 'ampsas', 'amptrend', 'ST', 'R0', 'total_cases',
     #                       'peak_week', 'perc_geocode']
    print(STATE)
    print(TEST_YEAR)

    df = df_all.loc[df_all.uf == STATE]

    # generate the samples to train and test based on the regional data 
    X_train, y_train = prep.generate_regional_train_samples(df, enso, TEST_YEAR, columns_to_normalize=cols_to_norm,
                                                             episcanner = True,
                                                            min_year = min_year)

    # parameters of the model
    LOSS = 'msle'
    batch_size = 4
    model_name = 'baseline'

    #create model
    model = build_baseline(hidden=64, features=4, predict_n=52, look_back=89, loss=LOSS, 
                    stateful = False, batch_size = batch_size,  optimizer = 'adam', activation = 'sigmoid')

    # train model 
    model = train_model_using_cross_val(model, X_train, y_train, n_splits=4, epochs = 150,
                                    verbose = 0,
                                    batch_size = batch_size, 
                                    monitor = 'val_loss',
                                    min_delta = 0,
                                    patience = 20)

    #model = train_model(model, X_train, y_train, epochs = 500,
    #                                verbose = 0,
    #                                batch_size = batch_size, 
    #                                monitor = 'val_loss',
    #                                min_delta = 0,
    #                                patience = 30)


    # save model 
    model.save(f'saved_models/model_{STATE}_{TEST_YEAR-1}_{model_name}.keras')

    if apply:
        df_preds = sum_regions_predictions(model, df, enso, TEST_YEAR, cols_to_norm, True, False)
        df_preds['adm_1'] = STATE
        df_preds['adm_0'] = 'BR'
        df_preds['adm_2'] = pd.NA
        df_preds.to_csv(f'./predictions/preds_{STATE}_{TEST_YEAR}_{model_name}.csv', index = False)

    TEST_YEAR = 2024 
    print(TEST_YEAR)

    # save the model
    # retreinando os modelos usando os dados de 2023
    #regional samples
    X_train, y_train = prep.generate_regional_train_samples(df, enso, TEST_YEAR,cols_to_norm, True, False, min_year = 2023)

    model.compile(loss=LOSS, optimizer = Adam(learning_rate = 0.0005), metrics=["accuracy", "mape", "mse"])

    TB_callback = TensorBoard(
                log_dir="./tensorboard",
                histogram_freq=0,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                # embeddings_freq=10
            )

    #lr_scheduler = LearningRateScheduler(schedule)


    if STATE != 'DF':
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=42)

        hist = model.fit(
                    X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=100,
                    verbose=0,
                    validation_data=(X_val, y_val),
                    callbacks=[TB_callback, EarlyStopping(monitor='val_loss',
                                                           min_delta=0, patience=15)]
                )
        
    else: 
        hist = model.fit(
                X_train,
                y_train,
                batch_size=1,
                epochs=100,
               verbose=0,
                callbacks=[TB_callback, EarlyStopping(monitor='loss', min_delta=0, patience=10)]
            )
    

    # save the model
    model.save(f'saved_models/model_{STATE}_{TEST_YEAR-1}_{model_name}.keras')

    if apply: 
        df_preds = sum_regions_predictions(model, df, enso, TEST_YEAR, cols_to_norm, True, False)
        df_preds['adm_1'] = STATE
        df_preds['adm_0'] = 'BR'
        df_preds['adm_2'] = pd.NA
        df_preds.to_csv(f'./predictions/preds_{STATE}_{TEST_YEAR}_{model_name}.csv', index = False)

end_time = time.time()

execution_time = end_time - start_time
print(f"Tempo de execução: {execution_time} segundos")