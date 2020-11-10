# Thinkful-Supervised-Learning-Electricity-Prices
## Predicting Retail Electricity Prices with Supervised Learning Approaches - Data from January 2008 to July 2020
  
[Open the Presentation Summary](https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/Supervised%20Learning%20-%20Presentation%20Summary.pdf)

## ReadMe Table of Contents

* [Project Information](#project-information)
* [Technologies](#technologies)
* [Key Data Visualizations](#key-data-visualizations)
* [Geographic Regions](#geographic-regions)


## Project Information

This project utilized supervised machine learning methods to predict retail electricity price data by US state over the past 12 years and determine if fossil fuel costs, without the costs of renewable or nuclear power, can reliably predict overall electricity prices.  Being able to predict changes in electricity prices relative to changes in fossil fuel markets improves strategic visibility for regional acquisitions and could improve risk management by quantifying incoming market impacts from upstream events and improving calibration of hedging and swap portfolios.  The power generation industry operates on long-term contracts from ten to thirty years in tenure, is a rapidly changing industry technologically and in regulation, and is exposed to significant fluctations in energy prices in merchant markets.

Supervised machine learning methods applied include:
 * __Linear Regression__
 * __Ridge Regression with cross validation__
 * __Lasso Regression with cross validation__
 * __ElasticNet Regression with cross validation__
 * __Support Vector Machine (SVM) Regression without hyperparameter tuning__
 * __Gradient Boosting Regression without hyperparameter tuning__
 * __Random Forest Regression without hyperparameter tuning__
 * __Random Forest Regressions with Random Search and Grid Search cross validation hyperparameter tuning__

The data was pulled from the Energy Information Administration (EIA) and includes 7,701 observations categorized by state location, energy type, and month.  The supervised machine learning models were applied to the entire fifty-state data set with the location field removed to start with a generalized approach.  Data of 6,160 observations in the training set and 1,541 observations in the test set.

Electricity prices were predictable from fossil fuel statistics, which represent a source of volatility and risk, and power generation volume, which is related to infrastructure that is easier to forecast.  __The best-performing model was the Random Forest Regression with Grid Search hyperparameter tuning, resulting in an adjusted R-squared value of 0.99 with a root mean squared error of `$`0.47 and a mean absolute error of `$`0.29, on an average electricity price of `$`10.65 across all time periods.__
	
## Technologies

Project is created with the following Python technologies and libraries:
 * Python version: 3.7.7
 * Numpy version: 1.18.1
 * Pandas version: 1.0.3
 * SciPy version: 1.4.1
 * StatsModels version: 0.11.0
 * Matplotlib version: 3.1.3
 * Seaborn version: 0.10.1

## Key Data Visualizations

### Supervised Machine Learning Regression Results

![alt text](<https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/images/supervised_learning_results.png>)

## Random Forest Regressions

![alt text](<https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/images/random_forest_regr_grid.png>)

![alt text](<https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/images/random_forest_regr_plain.png>)

## Gradient Boosting Regression

![alt text](<https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/images/gradient_boost_regr_plain.png>)

## Linear Regression

![alt text](<https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/images/linear_regr.png>)

## Variable Correlation Matrix

![alt text](<https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/images/correlations.png>)

### Variable Distributions

![alt text](<https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/images/variable_distributions.png>)

### Chronological Variable Data

![alt text](<https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/images/variables_over_time.png>)

## Category Feature Variable Counts

![alt text](<https://github.com/DataCodeFu/Thinkful-Supervised-Learning-Electricity-Prices/blob/main/images/category_counts.png>)

## Geographic Regions

![alt text](<https://github.com/DataCodeFu/Thinkful-Experimental-Design-Electricity-Prices/blob/master/EIA%20data%20sets/region_map.jpg>)
