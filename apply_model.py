import numpy as np
import pandas as pd
import preprocess_data as prep
from keras.optimizers import Adam
from models import build_bi_lstm, train_model_using_cross_val, build_baseline, make_predictions 
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from keras.models import load_model

STATE = 'MG'

df = prep.load_cases_data()
df = df.loc[df.uf == STATE]
enso = prep.load_enso_data()

model_name = 'baseline'

for test_year in [2023, 2024]:
    model = load_model(f'./saved_models/model_{STATE}_{test_year-1}_{model_name}.keras')

    df_w = prep.aggregate_data(df)

    data = df_w.merge(enso[['enso']], left_index = True, right_index = True)

    X_train, y_train, norm_values = prep.get_train_data(data.loc[data.year < test_year])

    X_test, y_test = prep.get_test_data(norm_values,data, year = test_year)


    # save preds
    df_preds = make_predictions(model, X_test, norm_values, dates = prep.gen_forecast_dates(test_year))

    df_preds['adm_1'] = STATE
    df_preds['adm_0'] = 'BR'
    df_preds['adm_2'] = pd.NA

    df_preds.to_csv(f'./predictions/preds_{STATE}_{test_year}_{model_name}.csv', index = False)
