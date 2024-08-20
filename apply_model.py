import numpy as np
import pandas as pd
import preprocess_data as prep
from keras.optimizers import Adam
from models import build_bi_lstm, train_model_using_cross_val
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from models import sum_regions_predictions
from keras.models import load_model

# Load the model
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

'''
This script is used to train the model for a specific STATE and forecast the cases on a 
specific year (TEST_YEAR). The model is trained  with the regional health data before the year selected. 
'''

df_all = prep.load_cases_data()
enso = prep.load_enso_data()

for state, model_name in zip(['PR'], 
                        ['baseline']):
    
    print(state)
    print(model_name)
    
    for year in [2023, 2024]:
        print(year)

        df = df_all.loc[df_all.uf == state]

        cols_to_norm = ['casos','epiweek', 'enso']#,  'R0', 'total_cases',
                                #'peak_week', 'perc_geocode'] 

        # save the model
        model = load_model(f'./saved_models/model_{state}_{year-1}_{model_name}.keras')

        df_preds = sum_regions_predictions(model, df, enso, year, cols_to_norm, episcanner=True, 
                                                                                clima = False, percentile_90 = True)
        df_preds['adm_1'] = state
        df_preds['adm_0'] = 'BR'
        df_preds['adm_2'] = pd.NA

        df_preds.to_csv(f'./predictions/preds_90_{state}_{year}_{model_name}.csv', index = False)