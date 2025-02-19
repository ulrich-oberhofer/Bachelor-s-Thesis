import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import L1Loss, MSELoss
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score, MeanAbsolutePercentageError
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import uniform, randint
from scipy.ndimage import gaussian_filter1d
# MLP imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
import sys
sys.path.append('..')
from utils import ml_parameters
from utils import settings as s
from utils.helper_functions import get_frequency_data, to_angular_freq
from utils.km_functions import km_get_drift, km_get_primary_control, km_get_diffusion
import csv
# define global imputer and scaler for preprocessing
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
min_max_scaler = MinMaxScaler()

import matplotlib.pyplot as plt

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()


def fit_linear_regression(X_train: np.array, y_train: np.array) -> np.array:
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    print('max and min: ', np.max(X_train), np.min(X_train))
    print('not nans: ', np.count_nonzero(~np.isnan(X_train)))
    print('nans: ', np.count_nonzero(np.isnan(X_train)))

    return model


def fit_mlp(X_train: np.array, y_train: np.array, parameters: dict, do_grid_search: bool) -> MLPRegressor:
    best_parameters = parameters

    if do_grid_search:
        params = {
            'activation': ['relu', 'identity'],
            'solver': ['adam', 'lbfgs', 'sgd'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
            'hidden_layer_sizes': [
                # (10, 100,),
                # (10, 100, 10,),
                # (10, 50, 10,),
                # (20, 100, 20,),
                # (50, 50, 50, 50,),
                # (10, 100, 100, 10,),
                # (20, 100, 100, 20,),
                # (10, 50, 50, 10,),
                # (10, 50, 100, 50, 10,),
                # (64, 64, 64),
                (64, 128, 64),
                # (128,128, 128),
                (128, 256, 128),
                (128, 256, 128, 64),
                (128, 256, 128, 64, 32),
                # (128, 256, 128, 64, 32, 16),
                # (128, 256, 128, 64, 32, 8),
                (16),
                (16,16),
                (32,16),
                (32),
                (64),
                (128),
                (64,32)
            ],
        }
        grid_search = GridSearchCV(
            estimator=MLPRegressor(
                random_state=42,
                learning_rate='constant',
                max_iter=1000,
                early_stopping=True, # added early stopping to prevent overfitting
            ),
            param_grid=params,
            cv=5,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_parameters = grid_search.best_estimator_.get_params()
        print('Best parameters: ', best_parameters)
    model = MLPRegressor(
        random_state=42,
        solver=best_parameters['solver'], #'lbfgs',#'sgd',#'adam',
        learning_rate='constant',
        max_iter=1000,
        activation=best_parameters['activation'],
        alpha=best_parameters['alpha'],
        learning_rate_init = best_parameters['learning_rate_init'],
        hidden_layer_sizes = best_parameters['hidden_layer_sizes'],
        early_stopping=True,
        #batch_size=128,
    )

    model.fit(X_train, y_train)
    print(best_parameters['activation'])
    return model


def impute_scale(x: np.array) -> np.array:
    #x = imp.transform(x) # !!! left that out -> we don't want to impute the test data -> actually need to rename the funnction
    
    #x = min_max_scaler.transform(x)
    x = standard_scaler.transform(x)
    return x

def impute_scale_standard(x: np.array) -> np.array: #!!! newly added
    #x = imp.transform(x) # !!! left that out -> we don't want to impute the test data -> actually need to rename the funnction
    x = standard_scaler.transform(x)
    return x


def evaluate_model(y_true: np.array, y_pred: np.array, model_type: str, dict_eval = None, set: str = 'test') -> None:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if dict_eval is not None:
        dict_eval[model_type] = {
            'Mean Squared Error': mse,
            'Mean Absolute Error': mae,
            'Mean Absolute Percentage Error': mape,
            'r2-Score': r2
        }
    print('{}_{}Mean Squared Error: '.format(model_type, set), mse)
    print('{}_{}Mean Absolute Error: '.format(model_type, set), mae)
    print('{}_{}Mean Absolute Percentage Error: '.format(model_type, set), mape)
    print('{}_{}r2-Score: '.format(model_type, set), r2)

    with open(f'../results/{files_prefix}model_errors_{set}_v3.txt', mode='a') as file:
        file.write(f'\n{model_type}\n')
        file.write(f'Mean Squared Error: {mse}\n')
        file.write(f'Mean Absolute Error: {mae}\n')
        file.write(f'Mean Absolute Percentage Error: {mape}\n')
        file.write(f'r2-Score: {r2}\n')


def save_model_parameters(model, model_type: str) -> None:
    params_model = model.get_params()

    with open(f'../results/{files_prefix}model_parameters_v3.txt', mode='a') as file:
        file.write(f'\n{model_type}\n')
        for key, value in params_model.items():
            if value is not None:
                file.write(f'{key}: {value}\n')


files_prefix = 'ml_v3/'
if s.ml['random noise']:
    files_prefix = 'ml_v3_random_noise/'
elif s.ml['knockout']:
    files_prefix = 'ml_v3_knockout/'

# clear model parameters and errors files
open(f'../results/{files_prefix}model_parameters_v3.txt', 'w').close()
open(f'../results/{files_prefix}model_errors_v3.txt', 'w').close()
open(f'../results/{files_prefix}model_errors_train_v3.txt', 'w').close()
open(f'../results/{files_prefix}model_errors_test_v3.txt', 'w').close()

dict_eval = {}
for area in  ['AUS']: #['AUS', 'CE']: # !!! just for CE at the moment !!!
    dict_eval[area] = {}
    y_complete = pd.DataFrame()
    y_complete_all = pd.DataFrame()

    data = pd.read_hdf(f'../results/prepared_features/{area}_detrended_ml.h5')

    feature_columns = [column for column in data.columns if column not in ['drift', 'diffusion']]
    X = data[feature_columns].copy()

    if s.ml['random noise']:
        np.random.seed(42)
        X['random_noise'] = np.random.rand(X.shape[0])

    print('Share of nans: ', X.isna().any(axis=1).sum()/len(X))
    for target in ['drift', 'diffusion']:
        dict_eval[area][target] = {}
        y = data[target]

        ''' !!! ADDITION: Keep only data without NaNs'''
        valid_ind = ~pd.concat([X, y], axis=1).isnull().any(axis=1)
        X, y = X[valid_ind], y[valid_ind]


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=s.ml['test size'], shuffle = True, random_state=2)

        # block_size = '4d'
        # masker = [pd.Series(g.index) for n, g in X.groupby(pd.Grouper(freq=block_size))]
        # train_mask, test_mask = train_test_split(masker, test_size = 0.2, random_state=2)
        # X_train = X.loc[pd.concat(train_mask)]
        # y_train = y.loc[pd.concat(train_mask)]
        # X_test = X.loc[pd.concat(test_mask)]
        # y_test = y.loc[pd.concat(test_mask)]
        
        # fit imputer and scaler
        imp.fit(X_train.drop(columns=s.top_features[area][target]['mlp']) if s.ml['knockout'] else X_train)
        min_max_scaler.fit(X_train.drop(columns=s.top_features[area][target]['mlp']).values if s.ml[
            'knockout'] else X_train.values)
        standard_scaler.fit(X_train.drop(columns=s.top_features[area][target]['mlp']).values if s.ml[
            'knockout'] else X_train.values)
        

        # # fit models (gbt: gradient boosted tree, rf: random forest, mlp: multi layer perceptron)
        mlp_model = fit_mlp(
            impute_scale_standard(X_train.drop(columns=s.top_features[area][target]['mlp']) if s.ml['knockout'] else X_train.values), #!!!
            y_train,
            ml_parameters.parameters_v3[area][target]['mlp'],
            do_grid_search = True #s.ml['grid search mlp']
        )
        lin_reg_model = fit_linear_regression(
            impute_scale(X_train.drop(columns=s.top_features[area][target]['lin_reg']) if s.ml['knockout'] else X_train.values),
            y_train
        )

        #!!! min_max_scaler might be missing here!
        # predict models
        y_pred_mlp = mlp_model.predict(impute_scale_standard( #!!!
            X_test.drop(columns=s.top_features[area][target]['mlp']) if s.ml['knockout'] else X_test.values
        ))
        y_pred_lin_reg = lin_reg_model.predict(
            impute_scale(X_test.drop(columns=s.top_features[area][target]['lin_reg']) if s.ml['knockout'] else X_test.values
        ))

        # In order to calculate the training errors, we also need to predict on the training data
        y_pred_mlp_train = mlp_model.predict(impute_scale_standard( #!!!
            X_train.drop(columns=s.top_features[area][target]['mlp']) if s.ml['knockout'] else X_train.values
        ))
        y_pred_lin_reg_train = lin_reg_model.predict(
            impute_scale(X_train.drop(columns=s.top_features[area][target]['lin_reg']) if s.ml['knockout'] else X_train.values
        ))



        # evaluate models
        model_description = f'{area}: Detrended {target.capitalize()}'
        evaluate_model(y_test, y_pred_mlp, f'{model_description} (Multi Layer Perceptron)' )
        evaluate_model(y_test, y_pred_lin_reg, f'{model_description} (Linear Regression)', dict_eval[area][target])

        # Evaluate the training errors
        evaluate_model(y_train, y_pred_mlp_train, f'{model_description} (Multi Layer Perceptron)', set='train')
        evaluate_model(y_train, y_pred_lin_reg_train, f'{model_description} (Linear Regression)', dict_eval[area][target], set='train')

        # Do the same for the training data

        # # model parameters
        save_model_parameters(
            mlp_model,
            f'{model_description} (Multi Layer Perceptron)'
        )
        save_model_parameters(
            lin_reg_model,
            f'{model_description} (Linear Regression)'
        )

        # store y values for box plot
        y_complete[f'{target}_true'] = y_test
        y_complete[f'{target}_mlp'] = y_pred_mlp
        y_complete[f'{target}_lin_reg'] = y_pred_lin_reg

        # store y values for all the data
        y_complete_all[f'{target}_true_all'] = y
        y_complete_all[f'{target}_mlp_all'] = mlp_model.predict(
            impute_scale_standard(X.drop(columns=s.top_features[area][target]['mlp']) if s.ml['knockout'] else X.values #!!!
                         ))
        y_complete_all[f'{target}_lin_reg_all'] = lin_reg_model.predict(
            impute_scale(X.drop(columns=s.top_features[area][target]['lin_reg']) if s.ml['knockout'] else X.values
        ))
    # save all predictions and true values
    y_complete.to_hdf(f'../results/{files_prefix}predictions/{area}_detrended_predictions.h5', key='df', mode='w')
    y_complete_all.to_hdf(
        f'../results/{files_prefix}predictions/{area}_detrended_all_predictions.h5', key='df', mode='w'
    )

# dict_eval.to_csv(f'../results/dict_{files_prefix}model_errors_v3.csv')
# with open(f'../results/dict_{files_prefix}model_errors_v3.csv', 'w') as csv_file:  
#     writer = csv.writer(csv_file)
#     for key, value in dict_eval.items():
#        writer.writerow([key, value])
(pd.DataFrame.from_dict(data=dict_eval, orient='index').to_json(f'../results/dict_model_errors_v3.json'))