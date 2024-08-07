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
STATE = 'CE'
TEST_YEAR = 2024
MODEL_NAME = 'baseline'

df_all = prep.load_cases_data()
enso = prep.load_enso_data()

df = df_all.loc[df_all.uf == STATE]

cols_to_norm = ['casos','epiweek', 'enso',  'R0', 'total_cases',
                          'peak_week', 'perc_geocode'] 

# save the model
model = load_model(f'./saved_models/model_{STATE}_{TEST_YEAR-1}_{MODEL_NAME}.keras')

df_preds = sum_regions_predictions(model, df, enso, TEST_YEAR, cols_to_norm, episcanner=True, 
                                                                        clima = False)
df_preds['adm_1'] = STATE
df_preds['adm_0'] = 'BR'
df_preds['adm_2'] = pd.NA

df_preds.to_csv(f'./predictions/preds_{STATE}_{TEST_YEAR}_{MODEL_NAME}.csv', index = False)