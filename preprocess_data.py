import numpy as np
import pandas as pd
from epiweeks import Week

# this list will be used for multiple functions 

df_pop_region = pd.read_csv('./data/pop_regional.csv')
#episcanner parameters
df_all_epi = pd.read_csv('./data/episcanner_regional.csv.gz')


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

def get_train_data(df_w, columns_to_normalize = ['casos', 'epiweek', 'enso'], min_year=None, target_1 = False):
    '''
    Função para obter os dados de treinamento.
    O shape dos dados de input é: (n_rows, 52, 5)
    O shape dos dados de output é: (n_rows, 52)
    '''

    norm_values = df_w.max()[columns_to_normalize]
    
    df_w[columns_to_normalize] = df_w[columns_to_normalize]/norm_values

    if min_year is None:
        min_year = df_w.index.year.min() + 3
    else: 
        min_year = max(min_year, df_w.index.year.min() + 3) 

    X_train = np.empty((0,89, len(columns_to_normalize)+1))

    if target_1:
        y_train = np.empty((0, 26))

    else: 
        y_train = np.empty((0, 52))
    
    columns_to_normalize_ = columns_to_normalize + ['pop_norm']
    
    for year in np.arange(min_year, df_w.index.year.max()+1): 

        #print(year)

        last_values = np.empty((1, 89, 0))
        
        for col in columns_to_normalize_: 

            #print(col)

            last_ = df_w.loc[( (df_w.year< year-1) & (df_w.year>= year-2) ) | ((df_w.year== year-1) & (df_w.epiweek <=37/52))].sort_index()[col].values.reshape(1, -1,1)

            last_values = np.concatenate((last_values, last_), axis=2)
        #print(columns_to_normalize_)
        #print(X_train.shape)
        #print(last_values.shape)
        #print(last_values.shape)
        X_train = np.append(X_train, last_values, axis = 0)

        if target_1: 
            y_train = np.append(y_train, df_w.loc[(df_w.year == year) & (df_w.epiweek <= 26/52)][['casos']].values.reshape(1,-1),
                                   axis = 0)

        else:
            y_train = np.append(y_train, df_w.loc[df_w.year == year][['casos']].values.reshape(1,-1),
                                   axis = 0)

    return X_train.astype(np.float32), y_train.astype(np.float32), norm_values 

def get_test_data(norm_values, df_w, year, columns_to_normalize = ['casos','epiweek', 'enso'], target_1 = False): 
    '''
    Function that returns the test data for a specific year. It will be used to test
    the models 
    '''
    
    df_w[columns_to_normalize] = df_w[columns_to_normalize]/norm_values

    X_test = np.empty((0, 89, len(columns_to_normalize)+1))
    
    y_test = np.empty((0,df_w.loc[df_w.year == year][['casos']].values.shape[0]))

    last_values = np.empty((1, 89, 0))

    columns_to_normalize_ = columns_to_normalize + ['pop_norm']

    for col in columns_to_normalize_: 

        last_  = df_w.loc[( (df_w.year< year-1) & (df_w.year>= year-2) ) | ((df_w.year== year-1) & (df_w.epiweek <=37/52))].sort_index()[col].values.reshape(1, -1,1)

        last_values = np.concatenate((last_values, last_), axis=2)
        
    X_test = np.append(X_test, last_values, axis = 0)

    if  target_1:

        y_test = np.append(y_test, df_w.loc[(df_w.year == year) & (df_w.epiweek <= 26/52)][['casos']].values.reshape(1,-1),
                               axis = 0)

    else:
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

    enso.enso = enso.enso + 2

    enso = add_epiweek_label(enso)

    return enso 

def load_sea_indicators():
    '''
    Load the 3 sea indicators made available by the sprint: iod, enso, pdo. 
    '''
    enso = pd.read_csv('./data/enso.csv.gz')

    iod = pd.read_csv('./data/iod.csv.gz')

    enso = enso.merge(iod, right_on = 'date', left_on = 'date')

    pdo = pd.read_csv('./data/pdo.csv.gz')

    enso = enso.merge(pdo, right_on = 'date', left_on = 'date')

    enso.date = pd.to_datetime(enso.date)

    enso.set_index('date', inplace = True)

    enso = enso.loc[enso.index.year >= 2010]

    enso = enso.resample('W-SUN').mean()  # Resample to monthly frequency and calculate mean
    enso = enso.ffill()

    enso = add_epiweek_label(enso)

    return enso
   

def generate_regional_train_samples(df, enso, test_year, columns_to_normalize = ['casos', 'epiweek', 'enso'], episcanner = False, clima = False, min_year = None):
    '''
    Generate the train date from all the health regions 
    '''

    features = len(columns_to_normalize)+1
    X_train = np.empty((0, 89, features))
    y_train = np.empty((0, 52))

    list_of_enso_indicators = ['enso', 'iod', 'pdo']

    indicators = [item for item in list_of_enso_indicators if item in columns_to_normalize]
 

    for geo in df.regional_geocode.unique():
        
        if clima:
            df_w = aggregate_data_clima(df, geo, column = 'regional_geocode')
        else: 
            df_w = aggregate_data(df, geo, column = 'regional_geocode')

        
        #df_w['inc'] = 10*df_w['casos']/df_pop_region.loc[df_pop_region.regional_geocode==geo]['pop'].values[0]
        df_w['pop_norm'] = df_pop_region.loc[df_pop_region.regional_geocode==geo]['pop_norm'].values[0]

        if episcanner: 

            df_w = df_w.reset_index().merge( df_all_epi.loc[df_all_epi.code_region == geo][['year',
                                                          'R0', 
                                                          'peak_week',
                                                          'total_cases', 
                                                          'perc_geocode']], how = 'left', left_on = 'year', right_on = 'year').set_index('date')

            df_w = df_w.fillna(0)
        
        data = df_w.merge(enso[indicators], left_index = True, right_index = True)

        X_train_, y_train_, norm_values_ = get_train_data(data.loc[data.year < test_year], columns_to_normalize= columns_to_normalize, min_year = min_year)

        X_train = np.append(X_train, X_train_, axis = 0)

        y_train = np.append(y_train, y_train_ , axis = 0)
    
    return X_train, y_train