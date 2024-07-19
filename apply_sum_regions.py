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
STATE = 'MG'
TEST_YEAR = 2024

df = prep.load_cases_data()
df = df.loc[df.uf == STATE]

df_clima = prep.load_climate_data()

df_clima = df_clima.loc[df_clima.geocode.astype(str).str[:2] == '31']

df_end = df.merge(df_clima, left_on = ['date', 'epiweek', 'geocode'], right_on = ['date', 'epiweek', 'geocode'])

df_end.date = pd.to_datetime(df_end.date)

df_end.set_index('date', inplace = True)

enso = prep.load_sea_indicators()

columns_to_normalize = ['casos', 'temp_med', 'temp_amp', 'rel_humid_med', 'precip_tot', 'enso', 'iod', 'pdo']

X_train, y_train = prep.generate_regional_train_samples(df_end, enso, TEST_YEAR, columns_to_normalize, True)

# save the model
model = load_model(f'./saved_models/region_model_{STATE}_{TEST_YEAR-2}.keras')

df_preds = sum_regions_predictions(model, df_end, enso, TEST_YEAR, columns_to_normalize)
df_preds['adm_1'] = STATE
df_preds['adm_0'] = 'BR'
df_preds['adm_2'] = pd.NA

df_preds.to_csv(f'./predictions/preds_region_{STATE}_{TEST_YEAR}_.csv', index = False)