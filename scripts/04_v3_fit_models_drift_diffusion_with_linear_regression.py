# This script is used because I did not want to replace the old script in case I needed the results again.
# It removes estimating on original data, but instead adds new models.

import lightgbm as lgb
import numpy as np
import xgboost as xgb
import shap
import joblib as jbl
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import uniform, randint

# MLP imports
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neural_network import MLPRegressor
import sys
sys.path.append('..')
from utils import ml_parameters
from utils import settings as s
import csv
# define global imputer and scaler for preprocessing
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

def fit_gbdt_lgb(X_train: np.array, y_train: np.array, parameters: dict, rand_search: bool) -> lgb.LGBMRegressor:
    best_parameters = parameters

    if rand_search:
        param_distributions = {
            'num_leaves': randint(10, 100),
            'learning_rate': uniform(0.01, 0.5),
            'max_depth': randint(3, 12),
            'n_estimators': randint(100, 1000),
            'min_child_samples': randint(20, 80),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
        }

        random_search = RandomizedSearchCV(
            estimator=lgb.LGBMRegressor(
                #random_state=42, 
                boosting_type='gbdt', n_jobs=1),
            param_distributions=param_distributions,
            n_iter=s.ml['random search iterations'],
            cv=5,
            n_jobs=-1,
            # random_state=42
        )

        random_search.fit(X_train, y_train)
        best_parameters = random_search.best_estimator_.get_params()
        print('best parameters: ', best_parameters)
        # save best parameters
        with open(f'../results/{files_prefix}gbdt_lgb.txt', mode='a') as file:
            writer = csv.writer(file)
            for key, value in best_parameters.items():
                writer.writerow([key, value])

    # fit model
    model = lgb.LGBMRegressor(
        random_state=42,
        boosting_type='gbdt',
        num_leaves=best_parameters['num_leaves'],
        learning_rate=best_parameters['learning_rate'],
        max_depth=best_parameters['max_depth'],
        n_estimators=best_parameters['n_estimators'],
        min_child_samples=best_parameters['min_child_samples'],
        reg_alpha=best_parameters['reg_alpha'],
        reg_lambda=best_parameters['reg_lambda']
    )
    model.fit(X_train, y_train)
    return model


def fit_rf_lgb(X_train: np.array, y_train: np.array, parameters: dict, rand_search: bool) -> lgb.LGBMRegressor:
    best_parameters = parameters

    if rand_search:
        param_distributions = {
            'num_leaves': randint(10, 100),
            'max_depth': randint(3, 12),
            'n_estimators': randint(100, 1000),
            'min_child_samples': randint(20, 80),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
            'bagging_freq': randint(0, 10),
            'bagging_fraction': uniform(0, 1)
        }

        random_search = RandomizedSearchCV(
            estimator=lgb.LGBMRegressor(
                #random_state=42, 
                boosting_type='rf', n_jobs=1),
            param_distributions=param_distributions,
            n_iter=s.ml['random search iterations'],
            cv=5,
            n_jobs=-1,
            # random_state=42
        )

        random_search.fit(X_train, y_train)
        best_parameters = random_search.best_estimator_.get_params()

    # fit model
    model = lgb.LGBMRegressor(
        random_state=42,
        boosting_type='rf',
        num_leaves=best_parameters['num_leaves'],
        max_depth=best_parameters['max_depth'],
        n_estimators=best_parameters['n_estimators'],
        min_child_samples=best_parameters['min_child_samples'],
        reg_alpha=best_parameters['reg_alpha'],
        reg_lambda=best_parameters['reg_lambda'],
        bagging_freq=best_parameters['bagging_freq'],
        bagging_fraction=best_parameters['bagging_fraction']
    )
    model.fit(X_train, y_train)
    return model


def fit_xgb(
        X_train: np.array, y_train: np.array, parameters: dict, objective: str, rand_search: bool
) -> xgb.XGBRegressor:
    best_parameters = parameters

    if rand_search:
        param_distributions = {
            'eta': uniform(0.01, 0.5),
            'max_depth': randint(3, 12),
            'n_estimators': randint(100, 1000),
            'min_child_weight': randint(1, 30),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1),
        }

        random_search = RandomizedSearchCV(
            estimator=xgb.XGBRegressor(
                #random_state=42, 
                objective=objective, n_jobs=1),
            param_distributions=param_distributions,
            n_iter=s.ml['random search iterations'],
            cv=5,
            n_jobs=-1,
            # random_state=42
        )

        random_search.fit(X_train, y_train)
        best_parameters = random_search.best_estimator_.get_params()

    model = xgb.XGBRegressor(
        random_state=42,
        objective=objective,
        eta=best_parameters['eta'],
        max_depth=best_parameters['max_depth'],
        n_estimators=best_parameters['n_estimators'],
        min_child_weight=best_parameters['min_child_weight'],
        reg_alpha=best_parameters['reg_alpha'],
        reg_lambda=best_parameters['reg_lambda']
    )
    model.fit(X_train, y_train)
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
                # random_state=42,
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
for area in ['CE']: #['AUS', 'CE']: # !!! just for CE at the moment !!!
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
    for target in ['diffusion']: #['drift', 'diffusion']:
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
        gbt_lgb_model = fit_gbdt_lgb(
            X_train.drop(columns=s.top_features[area][target]['gbt_lgb']) if s.ml['knockout'] else X_train,
            y_train,
            ml_parameters.parameters_v3[area][target]['gbt_lgb'],
            s.ml['random search gbt_lgb']
        )
        gbt_xgb_squarederror_model = fit_xgb(
            (X_train.drop(columns=s.top_features[area][target]['gbt_xgb_squarederror']) if s.ml[
                'knockout'] else X_train),
            y_train,
            ml_parameters.parameters_v3[area][target]['gbt_xgb_squarederror'],
            'reg:squarederror',
            s.ml['random search gbt_xgb_squarederror']
        )
        gbt_xgb_absoluteerror_model = fit_xgb(
            X_train.drop(columns=s.top_features[area][target]['gbt_xgb_absoluteerror']) if s.ml[
                'knockout'] else X_train,
            y_train,
            ml_parameters.parameters_v3[area][target]['gbt_xgb_absoluteerror'],
            'reg:absoluteerror',
            s.ml['random search gbt_xgb_absoluteerror']
        )
        rf_lgb_model = fit_rf_lgb(
            X_train.drop(columns=s.top_features[area][target]['rf_lgb']) if s.ml['knockout'] else X_train,
            y_train,
            ml_parameters.parameters_v3[area][target]['rf_lgb'],
            s.ml['random search rf_lgb']
        )
        mlp_model = fit_mlp(
            impute_scale_standard(X_train.drop(columns=s.top_features[area][target]['mlp']) if s.ml['knockout'] else X_train), #!!!
            y_train,
            ml_parameters.parameters_v3[area][target]['mlp'],
            do_grid_search = s.ml['grid search mlp']
        )
        lin_reg_model = fit_linear_regression(
            impute_scale(X_train.drop(columns=s.top_features[area][target]['lin_reg']) if s.ml['knockout'] else X_train),
            y_train
        )

        #!!! min_max_scaler might be missing here!
        # predict models
        y_pred_gbt_lgb = gbt_lgb_model.predict(
            X_test.drop(columns=s.top_features[area][target]['gbt_lgb']) if s.ml['knockout'] else X_test
        )
        y_pred_gbt_xgb_squarederror = gbt_xgb_squarederror_model.predict(
            (X_test.drop(columns=s.top_features[area][target]['gbt_xgb_squarederror']) if s.ml['knockout'] else X_test)
        )
        y_pred_gbt_xgb_absoluteerror = gbt_xgb_absoluteerror_model.predict(
            X_test.drop(columns=s.top_features[area][target]['gbt_xgb_absoluteerror']) if s.ml['knockout'] else X_test
        )
        y_pred_rf_lgb = rf_lgb_model.predict(
            X_test.drop(columns=s.top_features[area][target]['rf_lgb']) if s.ml['knockout'] else X_test
        )
        y_pred_mlp = mlp_model.predict(impute_scale_standard( #!!!
            X_test.drop(columns=s.top_features[area][target]['mlp']) if s.ml['knockout'] else X_test
        ))
        y_pred_lin_reg = lin_reg_model.predict(
            impute_scale(X_test.drop(columns=s.top_features[area][target]['lin_reg']) if s.ml['knockout'] else X_test
        ))

        # In order to calculate the training errors, we also need to predict on the training data
        y_pred_gbt_lgb_train = gbt_lgb_model.predict(
            X_train.drop(columns=s.top_features[area][target]['gbt_lgb']) if s.ml['knockout'] else X_train
        )
        y_pred_gbt_xgb_squarederror_train = gbt_xgb_squarederror_model.predict(
            (X_train.drop(columns=s.top_features[area][target]['gbt_xgb_squarederror']) if s.ml['knockout'] else X_train)
        )
        y_pred_gbt_xgb_absoluteerror_train = gbt_xgb_absoluteerror_model.predict(
            X_train.drop(columns=s.top_features[area][target]['gbt_xgb_absoluteerror']) if s.ml['knockout'] else X_train
        )
        y_pred_rf_lgb_train = rf_lgb_model.predict(
            X_train.drop(columns=s.top_features[area][target]['rf_lgb']) if s.ml['knockout'] else X_train
        )
        y_pred_mlp_train = mlp_model.predict(impute_scale_standard( #!!!
            X_train.drop(columns=s.top_features[area][target]['mlp']) if s.ml['knockout'] else X_train
        ))
        y_pred_lin_reg_train = lin_reg_model.predict(
            impute_scale(X_train.drop(columns=s.top_features[area][target]['lin_reg']) if s.ml['knockout'] else X_train
        ))



        # evaluate models
        model_description = f'{area}: Detrended {target.capitalize()}'
        evaluate_model(y_test, y_pred_gbt_lgb, f'{model_description} (Gradient Boosted Tree, Mean Squared Error, LightGMB)')
        evaluate_model(y_test, y_pred_gbt_xgb_squarederror, f'{model_description} (Gradient Boosted Tree, Squared Error, XGBoost)' )
        evaluate_model(y_test, y_pred_gbt_xgb_absoluteerror,f'{model_description} (Gradient Boosted Tree, Absolute Error, XGBoost)')
        evaluate_model(y_test, y_pred_rf_lgb,f'{model_description} (Random Forest, LightGBM)')
        evaluate_model(y_test, y_pred_mlp, f'{model_description} (Multi Layer Perceptron)' )
        evaluate_model(y_test, y_pred_lin_reg, f'{model_description} (Linear Regression)', dict_eval[area][target])

        # Evaluate the training errors
        evaluate_model(y_train, y_pred_gbt_lgb_train, f'{model_description} (Gradient Boosted Tree, Mean Squared Error, LightGMB)', set='train')
        evaluate_model(y_train, y_pred_gbt_xgb_squarederror_train, f'{model_description} (Gradient Boosted Tree, Squared Error, XGBoost)', set='train')
        evaluate_model(y_train, y_pred_gbt_xgb_absoluteerror_train, f'{model_description} (Gradient Boosted Tree, Absolute Error, XGBoost)', set='train')
        evaluate_model(y_train, y_pred_rf_lgb_train, f'{model_description} (Random Forest, LightGBM)', set='train')
        evaluate_model(y_train, y_pred_mlp_train, f'{model_description} (Multi Layer Perceptron)', set='train')
        evaluate_model(y_train, y_pred_lin_reg_train, f'{model_description} (Linear Regression)', dict_eval[area][target], set='train')

        # Do the same for the training data

        # # model parameters
        save_model_parameters(
            gbt_lgb_model,
            f'{model_description} (Gradient Boosted Tree, Mean Squared Error, LightGMB)'
        )
        save_model_parameters(
            gbt_xgb_squarederror_model,
            f'{model_description} (Gradient Boosted Tree, Squared Error, XGBoost)'
        )
        save_model_parameters(
            gbt_xgb_absoluteerror_model,
            f'{model_description} (Gradient Boosted Tree, Absolute Error, XGBoost)'
        )
        save_model_parameters(
            rf_lgb_model,
            f'{model_description} (Random Forest, LightGBM)'
        )
        save_model_parameters(
            mlp_model,
            f'{model_description} (Multi Layer Perceptron)'
        )
        save_model_parameters(
            lin_reg_model,
            f'{model_description} (Linear Regression)'
        )

        '''Don't calculate SHAP values for now'''
        # shap feature importance
        gbt_lgb_explainer = shap.Explainer(gbt_lgb_model)
        gbt_xgb_squarederror_explainer = shap.Explainer(gbt_xgb_squarederror_model)
        gbt_xgb_absoluteerror_explainer = shap.Explainer(gbt_xgb_absoluteerror_model)
        rf_lgb_explainer = shap.Explainer(rf_lgb_model)
        # The MLP explainer is currently not implemented due to a large computing time
        # mlp_explainer = shap.KernelExplainer(mlp_model.predict, impute_scale(X_train))

        # save shap explainers
        path_prefix = f'../results/{files_prefix}explainers/{area}_detrended_{target}'
        with open(f'{path_prefix}_gbt_lgb_shap_values', 'wb') as file:
            jbl.dump(gbt_lgb_explainer, file)
        with open(f'{path_prefix}_gbt_xgb_squarederror_shap_values', 'wb') as file:
            jbl.dump(gbt_xgb_squarederror_explainer, file)
        with open(f'{path_prefix}_gbt_xgb_absoluteerror_shap_values', 'wb') as file:
            jbl.dump(gbt_xgb_absoluteerror_explainer, file)
        with open(f'{path_prefix}_rf_lgb_shap_values', 'wb') as file:
            jbl.dump(rf_lgb_explainer, file)
        # The MLP explainer is currently not implemented due to a large computing time
        # with open(f'{path_prefix}_mlp_shap_values', 'wb') as file:
        # jbl.dump(mlp_explainer, file)
        ''' End of SHAP calue calculation'''

        # store y values for box plot
        y_complete[f'{target}_true'] = y_test
        y_complete[f'{target}_gbt_lgb'] = y_pred_gbt_lgb
        y_complete[f'{target}_gbt_xgb_squarederror'] = y_pred_gbt_xgb_squarederror
        y_complete[f'{target}_gbt_xgb_absoluteerror'] = y_pred_gbt_xgb_absoluteerror
        y_complete[f'{target}_rf_lgb'] = y_pred_rf_lgb
        y_complete[f'{target}_mlp'] = y_pred_mlp
        y_complete[f'{target}_lin_reg'] = y_pred_lin_reg

        # store y values for all the data
        y_complete_all[f'{target}_true_all'] = y
        y_complete_all[f'{target}_gbt_lgb_all'] = gbt_lgb_model.predict(
            X.drop(columns=s.top_features[area][target]['gbt_lgb']) if s.ml['knockout'] else X
        )
        y_complete_all[f'{target}_gbt_xgb_squarederror_all'] = gbt_xgb_squarederror_model.predict(
            impute_scale(X.drop(columns=s.top_features[area][target]['gbt_xgb_squarederror']) if s.ml['knockout'] else X)
        )
        y_complete_all[f'{target}_gbt_xgb_absoluteerror_all'] = gbt_xgb_absoluteerror_model.predict(
            X.drop(columns=s.top_features[area][target]['gbt_xgb_absoluteerror']) if s.ml['knockout'] else X
        )
        y_complete_all[f'{target}_rf_lgb_all'] = rf_lgb_model.predict(
            X.drop(columns=s.top_features[area][target]['rf_lgb']) if s.ml['knockout'] else X
        )
        y_complete_all[f'{target}_mlp_all'] = mlp_model.predict(
            impute_scale_standard(X.drop(columns=s.top_features[area][target]['mlp']) if s.ml['knockout'] else X #!!!
                         ))
        y_complete_all[f'{target}_lin_reg_all'] = lin_reg_model.predict(
            impute_scale(X.drop(columns=s.top_features[area][target]['lin_reg']) if s.ml['knockout'] else X
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
(pd.DataFrame.from_dict(data=dict_eval, orient='index')
   .to_json(f'../results/dict_model_errors_v3.json'))