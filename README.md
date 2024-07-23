## Models for sprint 2024/25

This repo contains the models used to forecast the dengue cases in the 2024/2025. 

To use the codes in this repository is necessary to download all the `.csv` made available by the sprint organizers and save it in the data folder. 

## Data preprocessing 

The functions related to preprocessing the data are saved in the `preprocess_data.py`. Between then as the model's output, according to the sprint, should be weeks 41-40 there is a function to redefine this interval as 1-52. Therefore, this function transforms a label from epidemiological week '201041' into week '201101' called `transform_epiweek_label`. Also, there are functions to load the datasets and to train the model using cross-validation. Besides, weeks 53 have been removed from the data.

## Models
In the `models.py` module there are functions to apply different NN architectures using LSTM layers. 

## training methodologies
By now there are implemented two methodologies to forecast the cases in a state:

1. A neural network model that uses as input the dengue cases in weeks 1-52 of the previous 3 years (y-1) to predict weeks 1-52 of the next year. In addition, the model uses as a feature the enso time series over the last year and the average number of cases per epidemiological week over the last few years. Therefore, the model uses 5 features as input.

**The normalized data from all the regional health departments will be used as training data. In order to carry out the state-level forecast, the model will be retrained using the state data with few epochs and a low value for the learning rate. (In this last step, it is as if the transfer learning technique were being applied.)**

This methodology is applied in the notebook: `baseline_model.ipynb`. To train models for any state using this methodology just run the `train_model.py` and to apply the trained models to gen predictions use `apply_model.py`. 
The models trained are saved in the `saved_models` folder and the predictions in the `predictions` folder. The `custom_loss.`ipynb` notebook contains a function to apply a custom loss for training the model to improve the performance of the baseline model. This custom loss still must be improved.  

2. A neural network model that uses the value of the following variables over the last 3 years as features:

* `cases`;
* `temp_med`;
* `temp_amp`;
* `rel_humid_med`;
* `precip_tot`.

And the previous year's value of the following variables:
* `enso`;
* `iod`;
* `pdo`.

**The normalized data from all the regional health departments will be used as training data. In order to carry out the forecast at state level, the output of the forecasts after applying the model to the data from each of the health regions was added up.**

This methodology is applied in the notebook: `revised_model_region_in_2023.ipynb`. To train models for any state using this methodology just run the `train_model_region.py` and to apply the trained models to gen predictions use `apply_model_region.py`. 
The models trained are saved in the `saved_models` folder and the predictions in the `predictions` folder. The models and predictions created using this methodology contain the string `region` in the name. To see a performance of a specific region by a trained model take a look at the `apply_the_model_in_a_specific_region.ipynb` notebook. 





