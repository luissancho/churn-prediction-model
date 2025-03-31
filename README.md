# Churn WTTE Ensemble: Advanced Customer Churn Prediction Model

Churn WTTE Ensemble is a sophisticated machine learning solution designed to predict customer churn with high accuracy by combining multiple predictive models in an ensemble approach. This repository contains the implementation of a hybrid churn prediction system that leverages both survival analysis and gradient boosting techniques to forecast customer departure behavior.

## Table of Contents

- [Introduction to Churn Prediction](#introduction-to-churn-prediction)
- [Model Architecture](#model-architecture)
- [WTTE Model](#wtte-model)
- [XGB Model](#xgb-model)
- [Ensemble Model (Churn WTTE Ensemble)](#ensemble-model-churn-wtte-ensemble)
- [Performance Metrics](#performance-metrics)
- [Features](#features)
- [Model Output](#model-output)
- [Code Reference](#code-reference)
- [Contributing](#contributing)
- [License](#license)

### Introduction to Churn Prediction

Customer churn, the phenomenon where customers cease their relationship with a business, represents a significant challenge across industries. Predicting which customers are at risk of churning enables proactive retention strategies, ultimately improving customer lifetime value and business sustainability.

Churn WTTE Ensemble approaches this prediction challenge by combining multiple model architectures to create a robust system that can:

- Predict the probability of churn within different time horizons
- Estimate the expected time until churn occurs
- Identify key factors contributing to churn risk
- Provide actionable insights for customer retention efforts

### Model Architecture

Churn WTTE Ensemble is a hybrid model that combines two powerful approaches to churn prediction:

1. `WTTE (Weibull Time To Event)`: A survival analysis model specialized in predicting time-to-event phenomena.
2. `XGB (XGBoost)`: A gradient boosting framework optimized for classification tasks.

These models are combined through a carefully weighted ensemble method that leverages the strengths of each approach to produce more accurate and reliable predictions than either model could achieve independently.

### WTTE Model

Weibull Time To Event Recurrent Neural Network (WTTE-RNN) is a deep learning model for customer churn prediction. Based on the excellent work by Egil Martinsson [1].

This model focuses on predicting not just if, but when a customer might stop using a service. It uses the special characteristics of the Weibull distribution, a statistical approach that helps estimate the time until a particular event occurs (in this case, churn). This model is particularly useful because it provides a timeline, helping businesses understand the urgency of customer retention efforts.

It is particularly well-suited for churn prediction because it can handle:

- `Censored data`: Where we know a customer hasn't churned yet, but don't know when they might in the future.
- `Time-varying covariates`: Customer features that change over time.
- `Varying observation windows`: Different customer histories of different lengths.

At its core, the model uses a recurrent neural network (RNN) architecture that processes sequential customer data, including both profile information and usage patterns tracked over time. This RNN analyzes these temporal sequences to estimate the two key parameters (α, β) that define a Weibull distribution, which is widely used in survival analysis to model time-to-event scenarios. These parameters then allow the model to predict the expected time until a customer churns (Time To Event or TTE).

- `Alpha (α)`: The scale parameter, which affects the characteristic life of the distribution
- `Beta (β)`: The shape parameter, which determines the failure rate behavior

The neural network architecture outputs these two parameters (α, β) for each customer, which together define the probability distribution of the time until churn.

Key features of our WTTE implementation:

- `Recurrent layers`: LSTM or GRU cells to capture temporal patterns in customer behavior.
- `Masking layers`: To handle variable-length sequences and missing data.
- `Custom loss function`: Based on the Weibull log-likelihood for censored data.
- `Positive output activation`: Ensuring valid Weibull parameters through exponential or softplus activations.

The resulting WTTE model excels at providing not just churn probabilities, but also confidence intervals for when churn is likely to occur.

[1] [WTTE-RNN: Weibull Time To Event Recurrent Neural Network (Egil Martinsson, 2016)](https://odr.chalmers.se/items/a48cefc9-f542-4280-ba0d-c0a8f5106b9d)

### XGB Model

eXtreme Gradient Boosting (XGBoost) is an open-source library that provides a gradient-boosted decision tree (GBDT) framework. Based on the work by Tianqi Chen and Carlos Guestrin [1].

XGBoost is a highly efficient and flexible model that uses decision trees and works well for a wide range of prediction tasks. In the context of churn prediction, it helps identify the patterns and characteristics of customers who are likely to churn based on historical data.

While WTTE-RNN focuses on the temporal aspects of churn, XGBoost excels at:

- `Feature importance identification`: Discovering which customer attributes most strongly predict churn
- `Handling various data types`: Effectively processing numerical, categorical, and interaction features
- `Non-linear pattern recognition`: Capturing complex relationships in customer data

Our XGBoost implementation for churn prediction includes:

- `Feature engineering pipeline`: Transforming raw customer data into predictive features.
- `Hyperparameter optimization`: Using Bayesian optimization to tune model parameters.
- `Class imbalance handling`: Through weighted classes or sampling techniques.
- `Regularization techniques`: L1/L2 regularization to prevent overfitting.
- `Cross-validation framework`: For robust model evaluation.

For the purpose of churn prediction, the resulting XGB model is used to predict the probability of churn for each customer in a sliding window fashion, which is a common approach in time-to-event analysis. Instead of trying to predict the TTE directly, we predict whether an event will happen within a preset timeframe, and use the probability of churn as a proxy for the probability of the event. For example: will the customer churn within the next N months?

[1] [XGBoost: A Scalable Tree Boosting System (T. Chen & C. Guestrin, 2016)](https://arxiv.org/abs/1603.02754)

### Ensemble Model (Churn WTTE Ensemble)

By combining and integrating these two submodels, the global model leverages the strengths of both: the time-sensitive predictions of WTTE and the pattern recognition capabilities of XGB. This ensemble approach not only predicts if a customer will churn but also provides insights about when this might happen based on their information and usage over time.

Our model processes customer data, which includes application usage sequences over time, to predict potential churn. This allows businesses to intervene at the right time with targeted actions to retain customers, ultimately optimizing their strategies and improving customer satisfaction. It is a robust tool for businesses looking to enhance their customer retention strategies by predicting churn more accurately and timely.

This approach provides several advantages:

- `Complementary strengths`: WTTE excels at temporal patterns and survival dynamics, while XGB captures complex feature interactions.
- `Robust predictions`: Reduced variance and improved stability compared to single models.
- `Multi-dimensional insights`: Provides both churn probability and time-to-event estimates.
- `Graceful degradation`: Maintains reasonable performance even when one component model encounters unfamiliar patterns.

### Performance Metrics

The model is evaluated on multiple metrics to ensure comprehensive performance assessment:

- `AUC-ROC`: measures the model's ability to distinguish between churning and non-churning customers across different probability thresholds. Our model achieves scores above 0.9, where 1.0 represents perfect discrimination and 0.5 represents random guessing.
- `Precision/Recall`: Measures the model's accuracy in identifying churning customers (precision) and its ability to find all actual churning customers (recall) across different probability thresholds.
- `F1 Score`: Harmonic mean of precision and recall which is used as the main metric to select the best probability threshold for the model.

![Model Performance Metrics](docs/ensemble-scores.png)

Our benchmark tests show that the model typically outperforms single models by:

- 5-12% improvement in AUC-ROC over standalone XGBoost.
- 8-15% improvement in calibration over standalone WTTE-RNN.
- 10-20% reduction in prediction variance across customer segments.

### Features

In order to predict churn probability, the model uses a series of features that are computed from the customer information and the software usage data. Each row represents a customer at a specific point in time (monthly granularity).

**Index / Keys**

- `account_id`: Customer ID
- `period`: Current period (month)

**Customer Main Features**

- `plan`: Current purchased plan (0-5 scale based on MRR ranges)
- `interval`: Current billing interval (monthly, yearly, etc.) encoded as number of months between payments
- `country_es`: Is located in Spain?
- `country_mx`: Is located in Mexico?
- `country_latam`: Is located in Latam?
- `employees`: Number of employees (minimum 1)

The `plan` feature is encoded on a 0-5 scale based on Monthly Recurring Revenue (MRR) ranges:

- `0`: MRR < 1€
- `1`: 1€ ≤ MRR < 14€  
- `2`: 14€ ≤ MRR < 34€
- `3`: 34€ ≤ MRR < 64€
- `4`: 64€ ≤ MRR < 94€
- `5`: MRR ≥ 94€

**Customer History Features**

- `months`: Number of months that the customer has been with the company
- `gateway_auto`: Is auto-renewal enabled?
- `paid_periods`: Number of pays made by the customer over its lifetime
- `failed`: Has the account payments failed at any point?
- `failed_ratio`: Ratio of failed over total payments

**Software Usage Features**

- `usage`: Software usage level in each period (0-5 scale based on software usage events)
- `usage_groups`: Software `groups` section usage level in each period (0-5)
- `usage_payments`: Software `payments` section usage level in each period (0-5)
- `usage_avg`: Moving average of software usage level over the customer's lifetime until the current period (0-5)
- `usage_momentum`: The current trend/momentum of software usage over last periods

The `usage` feature represents the overall usage of the software. It is defined with the following scale ranges:

- `0`: 0-4 events / month
- `1`: 5-59 events / month
- `2`: 60-179 events / month
- `3`: 180-359 events / month
- `4`: 360-719 events / month
- `5`: 720+ events / month

The `usage_groups` and `usage_payments` features represent the usage of the `groups` and `payments` sections of the software, respectively. They are defined with the following scale ranges:

- `0`: 0-4 events / month
- `1`: 5-29 events / month
- `2`: 30-89 events / month
- `3`: 90-179 events / month
- `4`: 180-359 events / month
- `5`: 360+ events / month

The `usage_momentum` feature represents the current trend/momentum of software usage. It measures the rate of change of a triple Exponential Weighted Moving Average (EWMA), also called 'Trix' (from 'triple exponential') in technical analysis, applied over the customer's software usage events series until the current month. A rising or falling line is an uptrend or downtrend and Momentum shows the slope of that line, so it's positive for a steady uptrend, negative for a downtrend, and a crossing through zero is a trend-change, i.e. a peak or trough in the underlying average.

![Customers Usage Momentum](docs/usage-momentum.png)

### Model Output

![Model Prediction Distributions](docs/model-prediction-distributions.png)

The model output is a dataset containing all the predictions for each customer, each row having the following fields:

- `account_id`: Account ID
- `probability` (0 to 1): Probability of the account to churn in the next N months (defined by the `min_tte` parameter)
- `target` (0 or 1): Binary classification target (1 if the account will churn in the next N months, 0 otherwise)
- `segment` (1 to 5): Customers are grouped into 5 segments/clusters based on their probability of churning.
- `wtte_alpha` (0 to ∞): Weibull Alpha (α) parameter, which represents the predicted number of periods until the customer churns (the lower the value, the sooner the customer will churn)
- `wtte_beta` (0 to ∞): Weibull Beta (β) parameter, which represents the confidence in the prediction (the greater the value, the more confident the model is in the prediction)
- `usage_momentum` (-100 to 100): The current trend/momentum of software usage over last periods, described in the 'Software Usage Features' section above
- `updated_at`: The timestamp of the last prediction made for this customer

By default, the `min_tte` parameter is set to 1, which means that `tte` <= `min_tte`, meaning that the model predicts the probability of the customer to churn in the current period or the next one.

The `segment` field is computed by clustering the customers based on their probability of churning, using the K-Means algorithm. It is useful to identify different customer segments and to tailor retention strategies accordingly. This is the segmentation scale:

- `1` or `A`: Very low churn risk (most stable customers)
- `2` or `B`: Low churn risk
- `3` or `C`: Moderate churn risk
- `4` or `D`: High churn risk
- `5` or `E`: Very high churn risk (most likely to churn)

The `wtte_alpha` and `wtte_beta` fields are the parameters of the Weibull distribution that represent the predicted number of periods until the customer churns and the confidence the model has in this prediction, respectively. This values cannot be interpreted as a robust prediction, but they are useful to see each customer's probability distribution shape and thus identify the ones that are more likely to churn.

![Customers WTTE Weibull PDF/CDF](docs/wtte-pdf-cdf.png)

## Contributing

Contributions to improve ChurnEnsemble are welcome! Please see CONTRIBUTING.md for details on our code of conduct and submission process.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
