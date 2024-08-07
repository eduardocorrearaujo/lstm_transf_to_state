## Models for sprint 2024/25

This repo contains the models used to forecast the dengue cases in the 2024/2025. 

To use the codes in this repository is necessary to download all the `.csv` and `.csv.gz` files made available by the sprint organizers and save it in the data folder. 

## Data preprocessing 

The functions related to preprocessing the data are saved in the `preprocess_data.py`. Between then, as the model's output, according to the sprint, should be weeks 41-40, there is a function to redefine this interval as 1-52. Therefore, this function (`transform_epiweek_label`) transforms a label from epidemiological week '201041' into week '201101'. The epiweeks `53` have been removed from the data.

The neural network uses as input the dengue cases in transformer epiweeks 1-52 of the year before last (y-2) and the epiweeks 1-37 of the previous year to predict weeks 1-52 of the next year. In addition, the model uses as a feature the enso time series over the last year and the average number of cases per epidemiological week over the last few years. Therefore, the model uses 5 features as input. **The transformed epiweek 37 refers to the week 25 (considering the first week in January) it is done since requested by the sprint rules**

**The normalized data from all the regional health departments will be used as training data. The data of each regional health is normalized considering the data of the regional health.** 

Also, there are functions to load the datasets and to train the model using cross-validation. The training and test samples contain normalized data from all the regional healths of the state. The models were trained using two different datasets, one containing climate data and the other just with the time series of cases and ENSO index. The models and predictions generated using the climate data contain the label `climate` on the name.

![Preprocessing data](./figures/preprocess_data.png)

## Models
In the `models.py` module there are functions to apply different NN architectures using LSTM layers they are apresented in the figure below:

![NN architectures](./figures/comp_models.png)

## Training methodology

![Trainign methodology](./figures/training_workflow.png)

This methodology is applied in the notebook: `baseline_model.ipynb`. To train models for any state using this methodology just run the `train_model.py` and to apply the trained models to gen predictions use `apply_model.py`. 
The models trained are saved in the `saved_models` folder and the predictions in the `predictions` folder. The `custom_loss.`ipynb` notebook contains a function to apply a custom loss for training the model to improve the performance of the baseline model. This custom loss still must be improved.  

2. A neural network model that uses the value of the following variables over the last 3 years as features:

* `cases`;
* `temp_med`;
* `temp_amp`;
* `rel_humid_med`;
* `precip_tot`.

The state output is computed applying the model in every single regional health and summing the output.**

And the previous year's value of the following variables:
* `enso`;
* `iod`;
* `pdo`.

**The normalized data from all the regional health departments will be used as training data. In order to carry out the forecast at state level, the output of the forecasts after applying the model to the data from each of the health regions was added up.**

This methodology is applied in the notebook: `revised_model_region_in_2023.ipynb`. To train models for any state using this methodology just run the `train_model_region.py` and to apply the trained models to gen predictions use `apply_model_region.py`. 
The models trained are saved in the `saved_models` folder and the predictions in the `predictions` folder. The models and predictions created using this methodology contain the string `region` in the name. To see a performance of a specific region by a trained model take a look at the `apply_the_model_in_a_specific_region.ipynb` notebook. 

'casos', 'epiweek', 'temp_med', 'temp_amp', 'rel_humid_med', 'precip_tot', 'enso', 'iod', 'pdo',
                            'R0', 'total_cases',
                             'peak_week', 'perc_geocode'


| Feature       | Description                                                                                                    |
|---------------|----------------------------------------------------------------------------------------------------------------|
|`casos`        | Time series of cases aggregate by regional health                                                              | 
|`epiweek`      | The epiweek value                                                                                              |
|`pop_norm`     | The population of each regional health normalized by the biggest regional health population in the state       |
|`R0`           | The mean reproductive number of the epidemics notified by the Episcanner for the cities in the regional health | 
|`peak_week`    | The mean peak week of the epidemics notified by the Episcanner for the cities in the regional health           |
|`total_cases`  | The sum of the total cases of the epidemics notified by the Episcanner for the cities in the regional health   |
|`perc_geocode` | The fraction of cities in the regional health that had an epidemic identified by the Episcanner                | 
|`temp_med`     | Time series of average temperature aggregate by regional health                                                |
|`temp_amp`     | Time series of thermal amplitude aggregate by regional health                                                  |
|`rel_humid_med`| Time series of relative average humidity aggregate by regional health                                          |
|`precip_tot`   | Time series of total precipitation aggregate by regional health                                                |
|`enso`         | Time series of the enso indicator                                                                              |
|`iod`          | Time series of the iod indicator                                                                               |
|`pdo`          | Time series of the pdo indicator                                                                               |
|---------------|----------------------------------------------------------------------------------------------------------------|



