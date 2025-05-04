from data_cleaning_preprocessing import data_cleaning_processing
from data_prep_for_ML import train_test_data
from Classification_models import hyperparameter_tuning as hypertune_class 
from Classification_models import prediction_results as pred_class
from Regression_models import hyperparameter_tuning as hypertune_reg 
from Regression_models import prediction_results as pred_reg

# Call data_cleaning_processing function to clean and process raw data.
df = data_cleaning_processing()

print('\n\nData preparation for Machine Learning in progress.....')

# Call train_test_data function to process and split cleaned data (df) into training and test set.
X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_data(df, 'Plant Type-Stage')
X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_data(df, 'Temperature')

print('\nCompleted!!!')

print('\n\nRunning hyperparameter tuning .......')

# Hyperparameter tuning of models for classification task.
print('\n\nClassification: Plant Type-Stage')
best_params_rf_plant, best_params_xgb_plant, xgb_plant, rf_plant = hypertune_class(X_train_plant, y_train_plant)

# Hyperparameter tuning of models for regression task.
print('\n\nRegression: Temperature')
best_params_rf_temp, best_params_xgb_temp, xgb_temp, rf_temp = hypertune_reg(X_train_temp, y_train_temp)

print('\nCompleted!!!')

print('\n\nRunning training and prediction .......')

# Prediction of Plant Type-Stage (classification)
print('\nPlant Type-Stage Prediction:\n')
pred_class(X_train_plant, X_test_plant, y_train_plant, y_test_plant, best_params_rf_plant, best_params_xgb_plant, xgb_plant, rf_plant, charts=True)


# Prediction of Temperature (regression)
print('\nTemperature Prediction:\n')
pred_reg(X_train_temp, X_test_temp, y_train_temp, y_test_temp, best_params_rf_temp, best_params_xgb_temp, xgb_temp, rf_temp, charts=True )

