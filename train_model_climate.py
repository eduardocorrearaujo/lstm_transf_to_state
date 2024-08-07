import time 
import numpy as np
import pandas as pd
import preprocess_data as prep
from keras.optimizers import Adam
from models import train_model_using_cross_val, schedule, build_baseline, build_lstm_att, build_comb_lstm_att, sum_regions_predictions
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping,  LearningRateScheduler
from keras.models import load_model
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

'''
This script is used to train the model for a specific STATE and forecast the cases on a 
specific year (TEST_YEAR). The model is trained with the regional health data before the year selected. 
This notebook considers the use of climate data. 
'''

lr_scheduler = LearningRateScheduler(schedule)

state_to_code = {'RJ': 33, 'ES': 32, 'PR': 41, 'CE': 23, 'MA': 21,
                 'MG': 31, 'SC': 42, 'PE': 26, 'PB': 25, 'RN': 24,
                 'PI': 22, 'AL': 27, 'SE': 28, 'SP': 35, 'RS': 43,
                 'PA': 15, 'AP': 16, 'RR': 14, 'RO': 11, 'AM': 13,
                 'AC': 12, 'MT': 51, 'MS': 50, 'GO': 52, 'TO': 17,
                 'DF': 53, 'BA': 29}

df_all = prep.load_cases_data()
df_clima_all = prep.load_climate_data()

# flag to decide if the model will be applied or not
apply = True 

start_time = time.time()  
for STATE in ['AM', 'CE', 'GO', 'MG', 'PR']:
    
    TEST_YEAR = 2023 

    if STATE == 'PR': 

        min_year = 2019

    else: 

        min_year = 2013

    print(STATE)
    print(TEST_YEAR)

    df = df_all.loc[df_all.uf == STATE]

    df_clima = df_clima_all.loc[df_clima_all.geocode.astype(str).str[:2] == str(state_to_code[STATE])]

    df_end = df.merge(df_clima, left_on = ['date', 'epiweek', 'geocode'], right_on = ['date', 'epiweek', 'geocode'])

    df_end.date = pd.to_datetime(df_end.date)

    df_end.set_index('date', inplace = True)

    enso = prep.load_sea_indicators()

    columns_to_normalize = ['casos', 'epiweek', 'temp_med', 'temp_amp', 'rel_humid_med', 'precip_tot', 'enso', 'iod', 'pdo',
                            'R0', 'total_cases',
                             'peak_week', 'perc_geocode']

    X_train, y_train = prep.generate_regional_train_samples(df_end, enso, TEST_YEAR, columns_to_normalize, True, True, min_year =min_year)

    LOSS = 'mse'
    batch_size = 4 
    activation = 'relu'
    model_name = 'baseline'
    model = model =  build_baseline(hidden=64, features=14, predict_n=52, look_back=89, loss =LOSS, optimizer = 'adam',  
                                   stateful = False, batch_size = batch_size, activation = activation)

    model = train_model_using_cross_val(model, X_train, y_train, n_splits=4, epochs = 150,
                                    verbose = 0,
                                    batch_size = batch_size, 
                                    monitor = 'val_loss',
                                    min_delta = 0,
                                    patience = 20)

    # save the model
    model.save(f'./saved_models/model_climate_{STATE}_{TEST_YEAR-1}_{model_name}.keras')

    if apply:
        df_preds_23 = sum_regions_predictions(model, df_end, enso, TEST_YEAR, columns_to_normalize, True, True)

        df_preds_23['adm_1'] = STATE
        df_preds_23['adm_0'] = 'BR'
        df_preds_23['adm_2'] = pd.NA

        df_preds_23.to_csv(f'./predictions/preds_climate_{STATE}_{TEST_YEAR}_{model_name}.csv', index = False)

    # retreinando os modelos usando os dados de 2023
    #regional samples
    TEST_YEAR = 2024
    X_train_23, y_train_23 = prep.generate_regional_train_samples(df_end, enso, TEST_YEAR, columns_to_normalize, True, True, min_year = 2023)

    model.compile(loss=LOSS, optimizer = Adam(learning_rate = 0.0005), metrics=["accuracy", "mape", "mse"])

    TB_callback = TensorBoard(
                log_dir="./tensorboard",
                histogram_freq=0,
                write_graph=True,
                write_images=True,
                update_freq='epoch',
                # embeddings_freq=10
            )

    hist = model.fit(
                X_train_23,
                y_train_23,
                batch_size=4,
                epochs=100,
                verbose=0,
                callbacks=[TB_callback, EarlyStopping(monitor='loss', min_delta=0, patience=20)]
            )

    model.save(f'./saved_models/model_climate_{STATE}_{TEST_YEAR-1}_{model_name}.keras')

    if apply: 
        df_preds_24 = sum_regions_predictions(model, df_end, enso, TEST_YEAR, columns_to_normalize, True, True)

        df_preds_24['adm_1'] = STATE
        df_preds_24['adm_0'] = 'BR'
        df_preds_24['adm_2'] = pd.NA

        df_preds_24.to_csv(f'./predictions/preds_climate_{STATE}_{TEST_YEAR}_{model_name}.csv', index = False)

end_time = time.time()

execution_time = end_time - start_time
print(f"Tempo de execução: {execution_time} segundos")