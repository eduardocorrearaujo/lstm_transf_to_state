import numpy as np
import pandas as pd
from epiweeks import Week

# this list will be used for multiple functions 
list_of_enso_indicators = ['enso', 'iod', 'pdo']

def transform_epiweek_label(ep_label):
  '''
  Function to change the epiweeks 41-40 into 1-52.
  '''
  year_ep_label = int(ep_label[:4])
  week_ep_label = int(ep_label[4:])

  if week_ep_label <=40:
    new_week = 12 + week_ep_label
    new_year = year_ep_label

  else:
    new_week = week_ep_label - 40
    new_year = year_ep_label +1

  if new_week <=9:
    week_str = f'0{new_week}'
  else:
    week_str = str(new_week)

  new_ep_label = f'{new_year}{week_str}'

  return new_ep_label

def add_epiweek_label(df_w):
    '''
    This function assumes that the dataframe has a datetime index
    and add the epiweek and year value
    '''

    df_w['epiweek_label'] = [Week.fromdate(x) for x in df_w.index]

    df_w['epiweek_label'] = df_w['epiweek_label'].astype(str)

    df_w = df_w.loc[df_w.epiweek_label.str[-2:].astype(int) != 53]

    df_w['epiweek_label'] = df_w['epiweek_label'].apply(transform_epiweek_label)

    df_w['epiweek'] = df_w['epiweek_label'].astype(str).str[-2:].astype(int)
    df_w['year'] = df_w['epiweek_label'].astype(str).str[:4].astype(int)

    return df_w

def aggregate_data(df, geocode = None, column = 'geocode'):
  '''
  Função para agregar os dados a partir de um geocode específico, se o geocode não
  é fornecido os dados são agregados para todo o estado.
  '''

  if geocode is not None:

    df = df.loc[df[column] == geocode]

  df_w = df[['casos']]

  df_w = df_w.resample('W-SUN').sum()

  df_w = add_epiweek_label(df_w)

  return df_w


def aggregate_data_clima(df, geocode = None, column = 'geocode'):
    '''
    Função para agregar os dados a partir de um geocode específico, se o geocode não
    é fornecido os dados são agregados para todo o estado.
    '''

    if geocode is not None:

        df = df.loc[df[column] == geocode]

    df_w = df[['casos', 'temp_med', 'temp_amp', 'rel_humid_med', 'precip_tot']]

    df_w = df_w.resample('W-SUN').agg({'casos': 'sum',  
                                      'temp_med':'mean', 
                                      'temp_amp':'mean', 
                                      'rel_humid_med':'mean',
                                      'precip_tot': 'sum'})
    df_w = add_epiweek_label(df_w)

    return df_w



def get_train_data(df_w, columns_to_normalize = ['casos', 'enso'], min_year=None):
    '''
    Função para obter os dados de treinamento.
    O shape dos dados de input é: (n_rows, 52, 5)
    O shape dos dados de output é: (n_rows, 52)
    '''

    norm_values = df_w.max()[columns_to_normalize]
    
    df_w[columns_to_normalize] = df_w[columns_to_normalize]/norm_values
    
    if min_year is None:
        min_year = df_w.index.year.min() + 4
    else: 
        min_year = max(min_year, df_w.index.year.min() + 4) 

    indicators = [item for item in list_of_enso_indicators if item in columns_to_normalize]

    X_train = np.empty((0, 52, (len(columns_to_normalize))*3 - 2*len(indicators) + 1 ))

    y_train = np.empty((0, 52))
        
    for year in np.arange(min_year, df_w.index.year.max()+1): 
        
        last_values = df_w.loc[df_w.year <= year - 1][['epiweek', 'casos']].groupby('epiweek').mean().values.reshape(1, -1,1)

        for col in columns_to_normalize: 

            if (col == 'enso') or (col == 'iod') or (col == 'pdo'):

                last_  = df_w.loc[df_w.year== year-1][col].values.reshape(1, -1,1)

            else:
        
                last_3 = df_w.loc[df_w.year == year-3][col].values.reshape(1, -1,1)
                last_2 = df_w.loc[df_w.year == year-2][col].values.reshape(1, -1,1)
                last_1 = df_w.loc[df_w.year == year-1][col].values.reshape(1, -1,1)

                last_ = np.concatenate((last_1, last_2, last_3), axis=2)

            last_values = np.concatenate((last_values, last_), axis=2)
    
        X_train = np.append(X_train, last_values, axis = 0)
            
        y_train = np.append(y_train, df_w.loc[df_w.year == year][['casos']].values.reshape(1,-1),
                                   axis = 0)


    return X_train.astype(np.float32), y_train.astype(np.float32), norm_values 

def get_test_data(norm_values, df_w, year, columns_to_normalize = ['casos','enso']): 
    '''
    Function that returns the test data for a specific year. It will be used to test
    the models 
    '''
    df_w[columns_to_normalize] = df_w[columns_to_normalize]/norm_values

    indicators = [item for item in list_of_enso_indicators if item in columns_to_normalize]

    X_test = np.empty((0, 52, (len(columns_to_normalize))*3 - 2*len(indicators) + 1 ))
    
    y_test = np.empty((0,df_w.loc[df_w.year == year][['casos']].values.shape[0]))

    last_values = df_w.loc[df_w.year <= year - 1][['epiweek', 'casos']].groupby('epiweek').mean().values.reshape(1, -1,1)

    for col in columns_to_normalize: 

        if (col == 'enso') or (col == 'iod') or (col == 'pdo'):

                last_  = df_w.loc[df_w.year== year-1][col].values.reshape(1, -1,1)

        else:
        
            last_3 = df_w.loc[df_w.year == year-3][col].values.reshape(1, -1,1)
            last_2 = df_w.loc[df_w.year == year-2][col].values.reshape(1, -1,1)
            last_1 = df_w.loc[df_w.year == year-1][col].values.reshape(1, -1,1)

            last_ = np.concatenate((last_1, last_2, last_3), axis=2)

        last_values = np.concatenate((last_values, last_), axis=2)

    X_test = np.append(X_test, last_values, axis = 0)
        
    y_test = np.append(y_test, df_w.loc[df_w.year == year][['casos']].values.reshape(1,-1),
                               axis = 0)

    return X_test.astype(np.float32), y_test.astype(np.float32)

def gen_forecast_dates(year):
    '''
    Function to gen the date of the forecasted 41-40 weeks.
    '''

    dates = []
    for y, week in zip(np.concatenate((np.repeat(year-1, 12), np.repeat(year, 40))),
                        np.concatenate((np.arange(41,53), np.arange(1, 41)))):

        dates.append(Week(y, week).startdate())

    return dates 


def load_cases_data():
    '''
    Function that load the dataset of cases 
    '''
    df = pd.read_csv('./data/dengue.csv.gz')

    df.date = pd.to_datetime(df.date)

    df.set_index('date', inplace = True)

    return df 

def load_climate_data(): 
    '''
    Function that the load the climate data 
    '''
    df_climate = pd.read_csv('./data/climate.csv.gz', usecols = [
    'date', 'epiweek', 'geocode', 'temp_min', 'temp_med', 'temp_max', 'rel_humid_med', 'precip_tot']) 

    df_climate.date = pd.to_datetime(df_climate.date)

    df_climate = df_climate.loc[df_climate.epiweek.astype(str).str[-2:].astype(int) != 53]

    df_climate['temp_amp'] = df_climate['temp_max'] - df_climate['temp_min']

    return df_climate

#download enso data
def load_enso_data():
    '''
    Function that loads the enso data 
    '''
    enso = pd.read_csv('data/enso.csv.gz')

    enso.date = pd.to_datetime(enso.date)

    enso.set_index('date', inplace = True)

    enso = enso.loc[enso.index.year >= 2010]

    enso = enso.resample('W-SUN').mean()  # Resample to monthly frequency and calculate mean

    enso = enso.ffill()

    enso = add_epiweek_label(enso)

    return enso 

def load_sea_indicators():
    '''
    Load the 3 sea indicators made available by the sprint: iod, enso, pdo. 
    '''
    enso = pd.read_csv('https://raw.githubusercontent.com/eduardocorrearaujo/baseline_week/main/data/enso.csv')

    iod = pd.read_csv('https://raw.githubusercontent.com/eduardocorrearaujo/baseline_week/main/data/iod.csv', index_col = 'Unnamed: 0')

    enso = enso.merge(iod, right_on = 'date', left_on = 'date')

    pdo = pd.read_csv('https://raw.githubusercontent.com/eduardocorrearaujo/baseline_week/main/data/pdo.csv', index_col = 'Unnamed: 0')

    enso = enso.merge(pdo, right_on = 'date', left_on = 'date')

    enso.date = pd.to_datetime(enso.date)

    enso.set_index('date', inplace = True)

    enso = enso.loc[enso.index.year >= 2010]

    enso = enso.resample('W-SUN').mean()  # Resample to monthly frequency and calculate mean
    enso = enso.ffill()

    enso = add_epiweek_label(enso)

    return enso
   

def generate_regional_train_samples(df, enso, test_year, columns_to_normalize = ['casos', 'enso'], clima = False, min_year = None):
    '''
    Generate the train date from all the health regions 
    '''

    indicators = [item for item in list_of_enso_indicators if item in columns_to_normalize]

    features = len(columns_to_normalize)*3 - 2*len(indicators) + 1 

    X_train = np.empty((0, 52, features))
    y_train = np.empty((0, 52))

    for geo in df.regional_geocode.unique():
        
        if clima:
            df_w = aggregate_data_clima(df, geo, column = 'regional_geocode')
        else: 
            df_w = aggregate_data(df, geo, column = 'regional_geocode')

        data = df_w.merge(enso[indicators], left_index = True, right_index = True)

        X_train_, y_train_, norm_values_ = get_train_data(data.loc[data.year < test_year], columns_to_normalize= columns_to_normalize, min_year = min_year)

        X_train = np.append(X_train, X_train_, axis = 0)

        y_train = np.append(y_train, y_train_ , axis = 0)
    
    return X_train, y_train