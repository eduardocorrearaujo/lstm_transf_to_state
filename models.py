'''
This module contains the functions used to crate and train the neural network models
'''
import math
import numpy as np
import pandas as pd 
import tensorflow as tf
import preprocess_data as prep
import tensorflow.keras as keras
from keras.optimizers import Adam
from tensorflow.keras.activations import gelu
from tensorflow.keras.layers import (LSTM,  Dense, Dropout, Conv1D, Bidirectional, TimeDistributed, Flatten, RepeatVector, Activation,
                                     Permute, Multiply, Lambda, Reshape, Concatenate)

from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from sklearn.model_selection import KFold

from keras_multi_head import MultiHeadAttention
from keras_self_attention import SeqSelfAttention

df_pop_region = pd.read_csv('./data/pop_regional.csv')
#episcanner parameters
df_all_epi = pd.read_csv('./data/episcanner_regional.csv.gz')
# trend and sazonal descriptors: 
df_s_reg = pd.read_csv('./data/desc_ST_regional.csv', sep = ';')

df_s_reg = df_s_reg.replace({',': '.'}, regex=True)

df_s_reg = df_s_reg.rename(columns = {'ano_e':'year'})

df_s_reg['year'] = df_s_reg['year'] + 1

for col in ['ampsas', 'amptrend', 'ST']: 
    
    df_s_reg[col] = df_s_reg[col].astype(float)

def build_baseline(hidden=8, features=100, predict_n=4, look_back=4, loss ='msle', optimizer = 'adam',  stateful = False, batch_size = 1,
                   activation = 'relu'):
    
    '''
    Baseline model with two LSTM layers
    '''

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features)
    )

    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,

        return_sequences=True,
    )(inp, training=True)

    x = Dropout(0.2, name='dropout_1')(x, training=True)

    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        activation = gelu,
        return_sequences=False,
    )(x, training=True)

    x = Dropout(0.2, name='dropout_2')(x, training=True)


    out = Dense(
        predict_n,
        activation=activation 
    )(x)
        #activity_regularizer=regularizers.L2(l2) )(x)
    model = keras.Model(inp, out)

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model


def build_baseline_multi(hidden=8, features=100, predict_n=4, look_back=4, loss ='msle', optimizer = 'adam',  stateful = False, batch_size = 1,
                   activation = 'relu'):
    
    '''
    Baseline model with two LSTM layers
    '''

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features))
    
    att = MultiHeadAttention(
                        head_num=features,
                        name='Multi-Head',
                        #activation='sigmoid',
                        history_only = True
                    )(inp, training = True)

    att = Dropout(0.2, name='dropout_1')(att, training=True) 

    concat = Concatenate(axis=-1)([att, inp])

    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,

        return_sequences=True,
    )(concat, training=True)

    x = Dropout(0.2, name='dropout_2')(x, training=True)

    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        activation = gelu,
        return_sequences=False,
    )(x, training=True)

    x = Dropout(0.2, name='dropout_3')(x, training=True)


    out = Dense(
        predict_n,
        activation=activation 
    )(x)
        #activity_regularizer=regularizers.L2(l2) )(x)
    model = keras.Model(inp, out)

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model


def build_bi_lstm( hidden=8, features=100, predict_n=4, look_back=4, loss='msle', 
                  stateful = False, batch_size = 1,  optimizer = Adam(learning_rate=0.001), 
                  activation = 'relu'):
    '''
    Model with one bidirectional lstm layers and two LSTM layers after that
    '''

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features)
    )

    x = Bidirectional(LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful=stateful,
        return_sequences=True,
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-5),
        #bias_regularizer=regularizers.L2(1e-5),
        # activity_regularizer=regularizers.L2(1e-5),
        #activation=f_act_1,
        dropout=0.1,
        recurrent_dropout=0,
        implementation=2,
        unit_forget_bias=True,
    ), merge_mode='ave', name='bidirectional_1')(inp, training=True)

    x = Dropout(0.2)(x, training=True)    
    
    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        activation = gelu,
        
        return_sequences=False,
    )(x, training=True)

    x = Dropout(0.2)(x, training=True) 

    #x = BatchNormalization()(x, training = True)

    out = Dense(
        predict_n,
        activation = activation
    )(x)
       
    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model


def build_lstm_multi( hidden=8, features=100, predict_n=4, look_back=4, loss='msle', stateful = False, batch_size = 1,
                optimizer = Adam(learning_rate=0.001), activation = 'relu'):

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features))
    
    att = MultiHeadAttention(
                        head_num=features,
                        name='Multi-Head',
                        #activation='sigmoid',
                        history_only = True
                    )(inp, training = True)

    att = Dropout(0.2)(att, training=True) 

    concat = Concatenate(axis=-1)([att, inp])
    
    lstm_1 = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        
        return_sequences=False,
    )(concat, training=True)

    lstm_1 = Dropout(0.2)(lstm_1, training=True) 
    
    out = Dense(
        predict_n,
        activation=activation 
    )(lstm_1)

    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model


def build_lstm_multi_art( hidden=8, features=100, predict_n=4, look_back=4, loss='msle', stateful = False, batch_size = 1,
                optimizer = Adam(learning_rate=0.001), activation = 'relu'):

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features))
    
    att = MultiHeadAttention(
                        head_num=features,
                        name='Multi-Head',
                        #activation='sigmoid',
                        history_only = True
                    )(inp, training = True)

    att = Dropout(0.2, )(att, training=True)

    flat_att = Flatten()(att)

    lstm_1 = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,

        return_sequences=False,
    )(inp, training=True)

    lstm_1 = Dropout(0.2)(lstm_1, training=True) 

    flat_lstm = Flatten()(lstm_1)

    flat_inp = Flatten()(inp)

    concat = Concatenate(axis=-1)([flat_lstm, flat_att, flat_inp])

    out = Dense(
        predict_n,
        activation=activation 
    )(concat)

    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model



def build_lstm_att_comb( hidden=8, features=100, predict_n=4, look_back=4, loss='msle', stateful = False, batch_size = 1,
                optimizer = Adam(learning_rate=0.001), activation = 'relu'):

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features))
    
    att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       history_only= True,
                       name='Attention')(inp, training = True)
    

    out_att = Dense(
        hidden,
        activation='relu'
    )(att)
    
    lstm_1 = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        
        return_sequences=True,
    )(inp, training=True)

    lstm_1 = Dropout(0.2)(lstm_1, training=True) 

    concat = Concatenate(axis=-1)([out_att, lstm_1])

    x = LSTM(
        2*hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        #activation = gelu,
        return_sequences=False,
    )(concat, training=True)

    x = Dropout(0.2, )(x, training=True)

    #flatten = tf.keras.layers.Flatten()(concat)
    
    out = Dense(
        predict_n,
        activation=activation 
    )(x)

    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model


def build_comb_lstm_att( hidden=8, features=100, predict_n=4, look_back=4, loss='msle', stateful = False, batch_size = 1,
                optimizer = Adam(learning_rate=0.001), activation = 'relu'):

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features))
    
    att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(inp, training = True )

    att = Dropout(0.2)(att, training=True) 

    concat = Concatenate(axis=-1)([att, inp])
    
    lstm_1 = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        
        return_sequences=False,
    )(concat, training=True)

    lstm_1 = Dropout(0.2)(lstm_1, training=True) 
    
    out_1 = Dense(
        predict_n,
        activation=activation 
    )(lstm_1)

    out_1 = Dropout(0.2)(out_1, training=True) 

    lstm_2_1 = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,

        return_sequences=True,
    )(inp, training=True)

    lstm_2_1 = Dropout(0.2)(lstm_2_1, training=True)

    lstm_2_2 = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        activation = gelu,
        return_sequences=False,
    )(lstm_2_1, training=True)

    lstm_2_2 = Dropout(0.2)(lstm_2_2, training=True)


    out_2 = Dense(
        predict_n,
        activation=activation 
    )(lstm_2_2)

    out_2 = Dropout(0.2)(out_2, training=True) 

    concat = Concatenate(axis=1)([out_1, out_2])

    out = Dense(
        predict_n,
        activation=activation 
    )(concat)

    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model


def build_lstm_att( hidden=8, features=100, predict_n=4, look_back=4, loss='msle', stateful = False, batch_size = 1,
                optimizer = Adam(learning_rate=0.001), activation = 'relu'):

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features))
    
    att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(inp, training = True )

    att = Dropout(0.2)(att, training=True) 

    concat = Concatenate(axis=-1)([att, inp])
    
    lstm_1 = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        
        return_sequences=False,
    )(concat, training=True)

    lstm_1 = Dropout(0.2)(lstm_1, training=True) 
    
    out = Dense(
        predict_n,
        activation=activation 
    )(lstm_1)

    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model

def build_lstm_att_2( hidden=8, features=100, predict_n=4, look_back=4, loss='msle', stateful = False, batch_size = 1,
                optimizer = Adam(learning_rate=0.001), activation = 'relu'):

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features))
    
    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        
        return_sequences=True,
    )(inp, training=True)

    x = Dropout(0.2)(x, training=True)

    x = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(x, training = True )
    
    x = Dropout(0.2)(x, training=True)

    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        
        return_sequences=False,
    )(x, training=True)  
    
    x = Dropout(0.2)(x, training=True)

    out = Dense(
        predict_n,
        activation=activation
    )(x)
  
    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model


def build_lstm_att_3( hidden=8, features=100, predict_n=4, look_back=4, loss='msle', stateful = False, batch_size = 1,
                optimizer = Adam(learning_rate=0.001), activation = 'relu'):

    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features))

    x = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(inp, training = True )
    
    x = Dropout(0.2)(x, training=True)
    
    flatten = tf.keras.layers.Flatten()(x)

    out = Dense(
        predict_n,
        activation=activation
    )(flatten)
  
    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model


def build_conv_lstm( hidden=8, features=100, predict_n=4, look_back=4, loss='msle',
                  stateful = False, batch_size = 1,  optimizer = Adam(learning_rate=0.001), filters  =32):
    '''
    Model with one convolutional layer followed by two lstm layers
    '''
    inp = keras.Input(
        #shape=(look_back, features),
        batch_shape=(batch_size, look_back, features)
    )

    x = Conv1D(filters=filters, kernel_size=4,
                input_shape=(batch_size, look_back, features), padding = 'same')(inp, training=True)
    #x = ConvLSTM1D(filters=filters, kernel_size=4,
    #            input_shape=(look_back, features), padding = 'same', return_sequences = True)(inp, training=True)

    x = Dropout(0.2)(x, training=True)

    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        activation = gelu,

        return_sequences=True,
    )(x, training=True)


    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful = stateful,
        activation = gelu,

        return_sequences=False,
    )(x, training=True)

    x = Dropout(0.2)(x, training=True)

    x = BatchNormalization()(x, training = True)

    out = Dense(
        predict_n,
        activation='relu',
    )(x)

    model = keras.Model(inp, out)

    #optimizer = RMSprop(learning_rate=0.001, momentum= 0.5)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print(model.summary())
    return model
    

def custom_loss(y_true, y_pred):
    '''
    Test of a custom loss function
    '''
    first_log = tf.math.log(y_pred + 1.0)
    second_log = tf.math.log(y_true + 1.0)

    sum_values = abs(tf.reduce_sum(y_pred) - tf.reduce_sum(y_true))

    peak_mag = abs(tf.reduce_max(y_pred) - tf.reduce_max(y_true))

    peak_pos = abs(tf.argmax(y_pred) - tf.argmax(y_true))

    sum_values = tf.cast(sum_values, tf.float32)
    peak_mag = tf.cast(peak_mag, tf.float32)
    peak_pos = tf.cast(peak_pos, tf.float32)

    return tf.reduce_mean(tf.square(first_log - second_log))  + sum_values

def schedule(epoch, lr):
    return lr * math.exp(-0.1)

def train_model_using_cross_val(model, X_train, y_train, n_splits=4, epochs = 5,
                                verbose = 0,
                                batch_size = 4, 
                                monitor = 'val_loss',
                                min_delta = 0,
                                patience = 20):
    '''
    Function to training the model using cross-validation. The number of split is defined
    by the n_splits parameter. 
    '''

    lr_scheduler = LearningRateScheduler(schedule)

    TB_callback = TensorBoard(
            log_dir="./tensorboard",
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            # embeddings_freq=10
        )

    seed = 7

    # Definição das camadas de validação cruzada
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_no = 1
    for train_index, val_index in kf.split(X_train):
        
        print(f'Training fold {fold_no}...')

        # Split data
        X_train_, X_val_ = X_train[train_index], X_train[val_index]
        y_train_, y_val_ = y_train[train_index], y_train[val_index]

        hist = model.fit(
                    X_train_,
                    y_train_,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=verbose,
                    validation_data=(X_val_, y_val_),
                    callbacks=[TB_callback, EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)]
                )
        
        fold_no = fold_no + 1

    return model 

def make_predictions(model, X_test, norm, dates):
    '''
    Função para retornar as previsões do modelo em um DataFrame
    '''
    predicted = np.stack([model(X_test, training =True) for i in range(100)], axis=2)

    df_pred = pd.DataFrame(np.percentile(predicted, 50, axis=2))
    df_pred25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2))
    df_pred975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2))

    df_preds = pd.DataFrame()

    df_preds['lower'] = df_pred25.values.reshape(1,-1)[0]
    df_preds['preds'] = df_pred.values.reshape(1,-1)[0]
    df_preds['upper'] = df_pred975.values.reshape(1,-1)[0]

    df_preds = df_preds*norm['casos']

    df_preds['dates'] = pd.to_datetime(dates)

    return df_preds[['dates', 'lower', 'preds', 'upper']]

def sum_regions_predictions(model, df, enso, test_year, columns_to_normalize, episcanner = True, clima = True):
    '''
    Função que aplica o modelo para todas as regionais de saúde e retorna a soma,
    que representa a função para o estado no formato de um dataframe. Não sei se existem formas de
    otimizar esse loop for. 
    '''
    dates = prep.gen_forecast_dates(test_year)

    list_of_enso_indicators = ['enso', 'iod', 'pdo']

    indicators = [item for item in list_of_enso_indicators if item in columns_to_normalize]
 
    predicted = np.zeros((1,52,100))
    for geo in df.regional_geocode.unique():

        if clima:
            df_w = prep.aggregate_data_clima(df, geo, column = 'regional_geocode')
        else: 
            df_w = prep.aggregate_data(df, geo, column = 'regional_geocode')

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

        data = data.dropna()

        X_train, y_train, norm_values = prep.get_train_data(data.loc[data.year < test_year], columns_to_normalize= columns_to_normalize)

        X_test, y_test = prep.get_test_data(norm_values, data, test_year, columns_to_normalize)

        predicted_ = np.stack([model(X_test.astype(np.float32), training =True) for i in range(100)], axis=2)

        predicted_ = predicted_*norm_values['casos']
        #predicted_ = predicted_*df_pop_region.loc[df_pop_region.regional_geocode==geo]['pop'].values[0]/10

        predicted = predicted + predicted_

    df_pred = pd.DataFrame(np.percentile(predicted, 50, axis=2))
    df_pred25 = pd.DataFrame(np.percentile(predicted, 2.5, axis=2))
    df_pred975 = pd.DataFrame(np.percentile(predicted, 97.5, axis=2))

    df_preds = pd.DataFrame()

    df_preds['lower'] = df_pred25.values.reshape(1,-1)[0]
    df_preds['preds'] = df_pred.values.reshape(1,-1)[0]
    df_preds['upper'] = df_pred975.values.reshape(1,-1)[0]

    df_preds['dates'] = pd.to_datetime(dates)

    return df_preds[['dates', 'lower', 'preds', 'upper']]