## Introduction
This project is originally a technical assessment for AI Singapore Apprenticeship programme. The required task involves building a machine learning
pipeline, consisting of classification and regression models, for predicting Plant Type-Stage and Temperature based on senor data of a farm's
closed environment. Such predictions will support farm's strategic planning and resource allocation to improve crop management, optimise resource 
usage and increase yield predictability. Details of the task can be found in the PDF file named "AIAP 19 Technical Assessment" above. Datasets can be 
accessed in the "data" folder in "src" directory. Here, a machine learning pipeline consisting of 2 classifiers and 2 regressors are developed that 
can predict Plant Type-Stage and Temperature with good accuracy. 

## Methodology
Both Xgboost and Random Forest are used for the classification and regression tasks. The macro average f1 score and overall AUC are used as the 
performance evaluation metrices for classification. RMSE and RMSLE are used as the performance evaluation metrices for regression.

## Key Results

### Prediction of Plant Type-Stage (Classification)
The key prediction results from the 2 algorithms for the test dataset are as summarized below in Table 1. 

<p align="center"><strong>Table 1: Prediction Performances of 2 Different Models.</strong></p>
<table align="center">
  <tr>
    <th>Model</th>
    <th>Macro Average F1 score</th>
    <th>Overall AUC</th>
  </tr>
  <tr>
    <td>XgBoost (1)</td>
    <td>0.8300</td>
    <td>0.9869</td>
  </tr>
  <tr>
    <td>Random Forest (2)</td>
    <td>0.8400</td>
    <td>0.9869</td>
  </tr>
</table>

Figure 1 and 2 show the ROC curves and feature ranking results produced from XgBoost.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a726453e-dcda-44dc-a37c-5e970834f53a" alt="Diagram" width="700" height='300'/>
</p>
<p align="center"><em>Figure 1: Receiver Operating Curve - Prediction Performance of XgBoost.</em></p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/a2afa167-0b2f-4e34-9df3-0a68c7b64b10" alt="Diagram" width="1200" height='400'/>
</p>
<p align="center"><em>Figure 2: Ranking of Features in Order of Descending Importance By XgBoost.</em></p>

### Prediction of Temperature (Regression)
The key prediction results from the 2 algorithms for the test dataset are as summarized below in Table 2.

<p align="center"><strong>Table 2: Prediction Performances of 2 Different Models.</strong></p>
<table align="center">
  <tr>
    <th>Model</th>
    <th>RMSE</th>
    <th>RMSLE</th>
  </tr>
  <tr>
    <td>XgBoost (1)</td>
    <td>1.0916</td>
    <td>0.0442</td>
  </tr>
  <tr>
    <td>Random Forest (2)</td>
    <td>1.1686</td>
    <td>0.0475</td>
  </tr>
</table>

Figure 3 show the feature ranking results produced from XgBoost.

<p align="center">
  <img src="https://github.com/user-attachments/assets/5013d7b0-f68d-4998-b992-71c450d6e1d6" alt="Diagram" width="1200" height='400'/>
</p>
<p align="center"><em>Figure 3: Ranking of Features in Order of Descending Importance By XgBoost.</em></p>


## Conclusion
Our machine learning analysis shows that Light Intensity, Nutrient K, N and P are 4 environmental variables that consistently ranked among the top 5 features 
most predictive of (and associated with) Plant Type-Stage across the 3 different classifier models. However, only Light Intensity and Nutrient K are among the
top 5 features most predictive of (and associated with) Temperature across the 2 regressor models. The results provide us with insights on which environmental 
factors in the farms are most predictive of Plant Type-Stage and Temperature. 





