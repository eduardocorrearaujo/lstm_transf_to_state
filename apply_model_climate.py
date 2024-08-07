import time
import numpy as np
import pandas as pd
import preprocess_data as prep
from models import sum_regions_predictions
from keras.models import load_model
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

STATE = 'MG'
model_name = 'baseline'
test_year = 2023

state_to_code = {'RJ': 33, 'ES': 32, 'PR': 41, 'CE': 23, 'MA': 21,
                 'MG': 31, 'SC': 42, 'PE': 26, 'PB': 25, 'RN': 24,
                 'PI': 22, 'AL': 27, 'SE': 28, 'SP': 35, 'RS': 43,
                 'PA': 15, 'AP': 16, 'RR': 14, 'RO': 11, 'AM': 13,
                 'AC': 12, 'MT': 51, 'MS': 50, 'GO': 52, 'TO': 17,
                 'DF': 53, 'BA': 29}

df = prep.load_cases_data()
df = df.loc[df.uf == STATE]

df_clima = prep.load_climate_data()

df_clima = df_clima.loc[df_clima.geocode.astype(str).str[:2] == str(state_to_code[STATE])]

df_end = df.merge(df_clima, left_on = ['date', 'epiweek', 'geocode'], right_on = ['date', 'epiweek', 'geocode'])

df_end.date = pd.to_datetime(df_end.date)

df_end.set_index('date', inplace = True)

enso = prep.load_sea_indicators()
columns_to_normalize = ['casos', 'epiweek', 'temp_med', 'temp_amp', 'rel_humid_med', 'precip_tot', 'enso', 'iod', 'pdo',
                            'R0', 'total_cases',
                             'peak_week', 'perc_geocode']

model = load_model(f'./saved_models/model_climate_{STATE}_{test_year-1}_{model_name}.keras') 

df_preds = sum_regions_predictions(model, df_end, enso, test_year, columns_to_normalize)
df_preds['adm_1'] = STATE
df_preds['adm_0'] = 'BR'
df_preds['adm_2'] = pd.NA

df_preds.to_csv(f'./predictions/preds_climate_{STATE}_{test_year}_{model_name}.csv', index = False)
