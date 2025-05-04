import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, accuracy_score, f1_score, make_scorer
import yaml

def hyperparameter_tuning(X_train, y_train):

    # function for loading and read in config file
    def load_config(config_file):
        with open(config_file) as file:
            config = yaml.safe_load(file)
        return config
    # Config file with specifications of test size and random state
    config = load_config("src/config.yaml")
    
    XgBoost = xgb.XGBClassifier(random_state=config['random_state1'], objective='multi:softmax',num_class=12)
    
    params1 = {'max_depth': np.arange(3,15,config['step_max_depth1']),
              'learning_rate': np.arange(0.01,0.2,config['step_learning_rate1']),
              'subsample': np.arange(0.5, 1.0, config['step_subsample1']),
              'colsample_bytree': np.arange(0.5, 1.0, config['step_colsample_bytree1']),
              'colsample_bylevel': np.arange(0.5, 1.0, config['step_colsample_bylevel1']),
              'n_estimators': np.arange(100,1000,config['step_n_estimators1'])}
    
    best_tree1 = RandomizedSearchCV(estimator=XgBoost, param_distributions=params1, scoring=config['metric1'], cv=config['cv1'], n_iter=config['n_iter1'],
                                    n_jobs=-1, verbose=True, random_state=config['random_state1'])
    best_tree1.fit(X_train, y_train)

    best_params_xgb = best_tree1.best_params_
    best_score_xgb = best_tree1.best_score_
    
    print('\nBest parameters for XgBoost:', best_params_xgb)
    print('\nBest f1 score for XgBoost:', best_score_xgb)

    RandomForest = RandomForestClassifier(random_state=config['random_state1'])

    params2 = {'max_depth': np.arange(3,15,config['Step_max_depth3']),
              'criterion':config['criterion3'],
              'max_features':config['max_features3'],
              'min_samples_split': np.arange(2, 20,config['step_min_samples_split3']),
              'n_estimators': np.arange(100,1000,config['step_n_estimators3'])}
    
    best_tree2 = RandomizedSearchCV(estimator=RandomForest, param_distributions=params2, scoring=config['metric1'], cv=config['cv1'], n_iter=config['n_iter1'],
                                    n_jobs=-1, verbose=True, random_state=config['random_state1'])
    best_tree2.fit(X_train, y_train)

    best_params_rf = best_tree2.best_params_
    best_score_rf = best_tree2.best_score_
    
    print('\nBest parameters for Random Forest:', best_params_rf)
    print('\nBest f1 score for Random forest:', best_score_rf)

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
        
        y_predict_proba = m.predict_proba(X_test)
        print('\nPredicted probabilities:',y_predict_proba)
        
        confusion = metrics.confusion_matrix(y_test, y_predict)
        print('\nConfusion matrix:\n', confusion)
        if charts:
            confusion_df = pd.DataFrame(confusion, columns=np.unique(y_test), index = np.unique(y_test))
            plt.figure(figsize = (10,10))
            plt.rcParams.update({'font.size': 15})
            sns.heatmap(confusion_df, cmap = 'Blues', annot=True, fmt='g', square=True, linewidths=.5, cbar=False)
            plt.xlabel('Predicted')
            plt.ylabel('Expected')
            plt.tight_layout()
            plt.savefig('confusion_matrix_'+model_name[i]+'.png')
            plt.close()

        print('\nPrediction Scores:\n', classification_report(y_test, y_predict))

        overallAUC = roc_auc_score(y_test, y_predict_proba, multi_class='ovr')
        print('AUC:',overallAUC)
        if charts:
            plt.figure(figsize=(13, 6))
            colors = ['blue', 'red', 'green', 'yellow', 'orange', 'black', 'grey', 'cyan', 'magenta', 'violet', 'brown','lime']
            type_stage = ['Fruiting Vegetables-Maturity', 'Leafy Greens-Seedling','Vine Crops-Seedling', 'Fruiting Vegetables-Vegetative',
                          'Herbs-Maturity', 'Vine Crops-Vegetative', 'Herbs-Vegetative','Vine Crops-Maturity', 'Herbs-Seedling', 
                          'Leafy Greens-Maturity','Leafy Greens-Vegetative', 'Fruiting Vegetables-Seedling']
            for n in range(len(type_stage)):
                
                y_true = y_test.values.tolist()
                
                for j in range(len(y_true)):
                    if y_true[j] == n:
                        y_true[j]=1
                    else:
                        y_true[j]=0
                        
                fpr, tpr, thresholds = roc_curve(y_true, y_predict_proba[:,n].tolist())
                plt.plot(fpr, tpr, color=colors[n], lw=2, label=f'Class {type_stage[n]} (AUC={auc(fpr, tpr):.5f})')
            
            plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Multiclass ROC Curve - One vs Rest\n(Overall AUC={overallAUC:.5f})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('ROC_'+model_name[i]+'.png')
            plt.close()

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
            plt.savefig('feature_ranking_'+model_name[i]+'.png')
            plt.close()
        fsi_df = fsi_sorted.to_frame().reset_index().rename(columns={'index':'Feature',0:'Relative Importance'})
        fsi_df.index += 1
        print(f'\nFeatures ranked by importance:\n{fsi_df}')

        i+=1
        
    return None
