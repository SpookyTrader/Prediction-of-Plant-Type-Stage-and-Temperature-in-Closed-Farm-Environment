import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error, make_scorer
import yaml

def hyperparameter_tuning(X_train, y_train):

    # function for loading and read in config file
    def load_config(config_file):
        with open(config_file) as file:
            config = yaml.safe_load(file)
        return config
    # Config file with specifications of test size and random state
    config = load_config("src/config.yaml")
    
    rmse = make_scorer(root_mean_squared_error)
    
    XgBoost = xgb.XGBRegressor(random_state=config['random_state2'])
    
    params1 = {'max_depth': np.arange(3,15,config['step_max_depth2']),
              'learning_rate': np.arange(0.01,0.2,config['step_learning_rate2']),
              'subsample': np.arange(0.5, 1.0, config['step_subsample2']),
              'colsample_bytree': np.arange(0.5, 1.0, config['step_colsample_bytree2']),
              'colsample_bylevel': np.arange(0.5, 1.0, config['step_colsample_bylevel2']),
              'n_estimators': np.arange(100,1000,config['step_n_estimators2'])}
    
    best_tree1 = RandomizedSearchCV(estimator=XgBoost, param_distributions=params1, scoring=rmse, cv=config['cv2'], n_iter=config['n_iter2'], n_jobs=-1, 
                                   verbose=True, random_state=config['random_state2'])
    best_tree1.fit(X_train, y_train)

    best_params_xgb = best_tree1.best_params_
    best_score_xgb = best_tree1.best_score_
    
    print('\nBest parameters for XgBoost:', best_params_xgb)
    print('\nBest RMSE score for XgBoost:', best_score_xgb)

    RandomForest = RandomForestRegressor(random_state=config['random_state2'])

    params2 = {'max_depth': np.arange(3,15,config['Step_max_depth4']),
              'max_features':['sqrt','log2',None],
              'min_samples_split': np.arange(2, 20, config['step_min_samples_split4']),
              'n_estimators': np.arange(100,1000,config['step_n_estimators4'])}
    
    best_tree2 = RandomizedSearchCV(estimator=RandomForest, param_distributions=params2, scoring=rmse, cv=config['cv2'], n_iter=config['n_iter2'], n_jobs=-1, 
                                    verbose=True, random_state=config['random_state2'])
    best_tree2.fit(X_train, y_train)

    best_params_rf = best_tree2.best_params_
    best_score_rf = best_tree2.best_score_
    
    print('\nBest parameters for Random Forest:', best_params_rf)
    print('\nBest RMSE score for Random forest:', best_score_rf)

    return best_params_rf, best_params_xgb, XgBoost, RandomForest

def prediction_results(X_train, X_test, y_train, y_test, best_params_rf, best_params_xgb, XgBoost, RandomForest, charts=False):

    i = 0
    model_name = ['XgBoost', 'RandomForest']
    parameters = [best_params_xgb, best_params_rf]
    
    for m in [XgBoost, RandomForest]:
    
        m.set_params(**parameters[i])
    
        m.fit(X_train, y_train)
    
        print(f'\nResults for {model_name[i]}:')

        y_predict = m.predict(X_test)
        print('\nPrediction on test data:', y_predict)

        score_rmse = root_mean_squared_error(y_test,y_predict)
        print('\nRMSE on test dataset: ', score_rmse)
        
        score_rmsle = root_mean_squared_log_error(y_test,y_predict)
        print('\nRMSLE on test dataset: ', score_rmsle)

        fsi = pd.Series(m.feature_importances_, index=X_train.columns)
        fsi_sorted = (fsi/fsi.sum()).sort_values(ascending=False)
        if charts:
            plt.figure(figsize=(13, 8))
            plt.bar(fsi_sorted.index, fsi_sorted.values, color='blue')
            plt.xlabel('Features')
            plt.ylabel('Relative Importance')
            plt.xticks(rotation=90, fontsize=10)
            plt.yticks(fontsize=10)
            plt.title("Ranking of feature's importance")
            plt.grid(False)
            plt.tight_layout()
            plt.savefig('reg_feature_ranking_'+model_name[i]+'.png')
            plt.close()
        fsi_df = fsi_sorted.to_frame().reset_index().rename(columns={'index':'Feature',0:'Relative Importance'})
        fsi_df.index += 1
        print(f'\nFeatures ranked by importance:\n{fsi_df}')

        i+=1
        
    return None
    