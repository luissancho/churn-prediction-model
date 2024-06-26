import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from typing import Optional
from typing_extensions import Self

from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from .WTTE import WTTE
from .XGB import XGB
from .utils import format_number, get_color


class ChurnEnsemble(object):
    """
    Churn WTTE Ensemble model for customer churn prediction.

    This model combines two powerful predictive techniques, the WTTE (Weibull Time To Event) model
    and the XGBoost model, to predict customer churn effectively.

    WTTE Time To Event model:
    This model focuses on predicting not just if, but when a customer might stop using a service.
    It uses the Weibull distribution, a statistical approach that helps estimate the time until
    a particular event occurs (in this case, churn). This model is particularly useful because it
    provides a timeline, helping businesses understand the urgency of customer retention efforts.

    XGBoost model:
    XGBoost stands for eXtreme Gradient Boosting. It is a highly efficient and flexible model
    that uses decision trees and works well for a wide range of prediction tasks.
    In the context of churn prediction, it helps identify the patterns and characteristics of customers
    who are likely to churn based on historical data.

    By combining and integrating these two models, the global model leverages the strengths of both:
    the time-sensitive predictions of WTTE and the pattern recognition capabilities of XGBoost.
    This ensemble approach not only predicts if a customer will churn but also pinpoints
    when this might happen based on their information and usage over time.

    The model processes customer data, which includes usage sequences over time, to predict potential churn.
    This allows businesses to intervene at the right time with targeted actions to retain customers,
    ultimately optimizing their strategies and improving customer satisfaction.

    This ensemble model is a robust tool for businesses looking to enhance their customer retention
    strategies by predicting churn more accurately and timely.

    Parameters
    ----------
    min_tte : int, optional
        Minimum time to event for binary classification (positive if `tte` <= `min_tte`).
        Defaults to 1 (churn if customer churns in the current period or the next).
    max_sl : int, optional
        Maximum sequence length.
        If 0, the maximum sequence length will be set to the maximum sequence length from data.
    seed : int, optional
        Base random seed for reproducibility.
    verbose : int, optional
        Verbosity level.
        If 0, no output will be printed.
        If 1, only important messages will be printed.
        If 2, all messages will be printed (including fit and predict outputs).
    path : str, optional
        Path to store model files and images.
    kwargs : dict, optional
        Additional parameters to be used in the model.
    """

    def __init__(
        self,
        min_tte: int = 1,
        max_sl: int = 0,
        seed: int = 42,
        verbose: int = 0,
        path: Optional[str] = None,
        **kwargs
    ):
        # Column names
        self.id_col = 'id'  # Sequence ID
        self.tp_col = 'tp'  # Period (month)
        self.ts_col = 'ts'  # Start date
        self.te_col = 'te'  # End date
        self.tfs_col = 'tfs'  # Periods from start date
        self.tte_col = 'tte'  # Periods until end date
        self.cid_col = 'cid'  # Customer ID

        # Weibull parameters
        self.wa_col = 'wa'  # Alpha
        self.wb_col = 'wb'  # Beta

        # Execution params
        self.min_tte = min_tte
        self.max_sl = max_sl
        self.seed = seed
        self.verbose = verbose
        self.path = self.set_path(path)

        # Model params
        self.params = {
            'wtte': {
                'features': kwargs.get('wtte', {}).get('features', []),
                'params': kwargs.get('wtte', {}).get('params', {})
            },
            'xgb': {
                'features': kwargs.get('xgb', {}).get('features', []),
                'params': kwargs.get('xgb', {}).get('params', {})
            }
        }

        self.data = pd.DataFrame()  # Input data
        self.dtrain = pd.DataFrame()  # Training data split
        self.dtest = pd.DataFrame()  # Test data split

        # Model instances
        self.wtte = None  # Weibull Time To Event model
        self.xgb = None  # XGBoost model

        # Results
        self.results = pd.DataFrame()  # Predicted results

        # Model scores
        self.thr = .5  # Minimum probability threshold to classify an observation as churned
        self.scores = {
            'acc': .0,  # Accuracy
            'auc': .0,  # Area under the ROC curve
            'f1': .0,  # F1 score
            'precision': .0,  # Precision
            'recall': .0  # Recall
        }

    def get_params(self) -> dict:
        """
        Get model parameters.

        Returns
        -------
        dict
            Model parameters.
        """
        return {
            'wtte': self.wtte.params if self.wtte is not None else self.params['wtte']['params'],
            'xgb': self.xgb.params if self.xgb is not None else self.params['xgb']['params']
        }
    
    def set_data(
        self,
        data: pd.DataFrame,
        test_size: Optional[float] = None
    ) -> Self:
        """
        Set the data for the ensemble and split into training and test sets.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        test_size : float, optional
            Percentage of the data to use for test/validation.
            If provided, the data is split into training and test sets.
        """
        self.data = data.sort_values([self.id_col, self.tfs_col])

        for col in [self.tp_col, self.ts_col, self.te_col]:
            self.data[col] = pd.to_datetime(self.data[col])

        if self.verbose > 0:
            cs = (
                self.data.sort_values([self.id_col, self.tfs_col]).groupby(self.id_col)[self.tte_col].last() < 0
            ).value_counts().sort_index().astype(float)
            print('Total Customers: {} | Censored: {} | Non-censored: {} | Censored Rate {}%'.format(
                format_number(cs.sum()),
                format_number(cs[1]),
                format_number(cs[0]),
                format_number(100 * cs[1] / cs.sum(), 2)
            ))
        
        if test_size is not None:
            d_split = self.data.sort_values([self.id_col, self.tfs_col]).groupby(self.id_col)[self.tte_col].last().reset_index()
            d_split['censored'] = d_split[self.tte_col] < 0

            d_train, d_test = train_test_split(
                d_split,
                test_size=test_size,
                shuffle=True,
                stratify=d_split['censored'].astype(int),
                random_state=self.seed
            )

            if self.verbose > 0:
                cs_train = d_train['censored'].value_counts().sort_index().astype(float)
                cs_test = d_test['censored'].value_counts().sort_index().astype(float)
                print('Train: {} ({}%) | Test: {} ({}%)'.format(
                    format_number(len(d_train)),
                    format_number(100 * cs_train[1] / cs_train.sum(), 2),
                    format_number(len(d_test)),
                    format_number(100 * cs_test[1] / cs_test.sum(), 2)
                ))
            
            self.dtrain = self.data[
                self.data[self.id_col].isin(d_train[self.id_col])
            ].sort_values([self.id_col, self.tfs_col])

            self.dtest = self.data[
                self.data[self.id_col].isin(d_test[self.id_col])
            ].sort_values([self.id_col, self.tfs_col])

        return self

    def build_model(self) -> Self:
        """
        Build the ensemble model.
        """
        self.wtte = self.create_wtte(self.params['wtte'])
        self.xgb = self.create_xgb(self.params['xgb'])

        return self

    def create_wtte(
        self,
        config: dict
    ) -> WTTE:
        """
        Instantiate the Weibull Time To Event model.

        Parameters
        ----------
        config : dict
            Model configuration.

        Returns
        -------
        WTTE
            Weibull Time To Event model instance.
        """
        return WTTE(
            features=config['features'],
            min_tte=self.min_tte,
            max_sl=self.max_sl,
            seed=self.seed,
            verbose=self.verbose,
            path=f'{self.path}/wtte',
            **config['params']
        )
    
    def create_xgb(
        self,
        config: dict
    ) -> XGB:
        """
        Instantiate the XGBoost model.

        Parameters
        ----------
        config : dict
            Model configuration.

        Returns
        -------
        XGB
            XGBoost model instance.
        """
        return XGB(
            features=config['features'] + [self.wa_col, self.wb_col],
            min_tte=self.min_tte,
            seed=self.seed,
            verbose=self.verbose,
            path=f'{self.path}/xgb',
            **config['params']
        )
    
    def save_wtte(
        self
    ) -> Self:
        """
        Save the Weibull Time To Event model.
        """
        self.wtte.save()

        return self
    
    def save_xgb(
        self
    ) -> Self:
        """
        Save the XGBoost model.
        """
        self.xgb.save()

        return self
    
    def load_wtte(
        self
    ) -> WTTE:
        """
        Load the Weibull Time To Event model.

        Returns
        -------
        WTTE
            Weibull Time To Event model instance.
        """
        return WTTE(
            min_tte=self.min_tte,
            max_sl=self.max_sl,
            seed=self.seed,
            verbose=self.verbose,
            path=f'{self.path}/wtte',
        ).load()
    
    def load_xgb(
        self
    ) -> XGB:
        """
        Load the XGBoost model.

        Returns
        -------
        XGB
            XGBoost model instance.
        """
        return XGB(
            min_tte=self.min_tte,
            seed=self.seed,
            verbose=self.verbose,
            path=f'{self.path}/xgb',
        ).load()

    def fit_wtte(self) -> Self:
        """
        Fit the Weibull Time To Event model.
        """
        # Set train data
        d_wtte_train = self.dtrain[[self.id_col, self.tfs_col, self.tte_col] + self.wtte.features]
        # Scale/Normalize features
        self.wtte.scaler = StandardScaler()
        d_wtte_train[self.wtte.features] = self.wtte.scaler.fit_transform(d_wtte_train[self.wtte.features])
        # Build train tensor
        x_wtte_train, y_wtte_train = self.wtte.build_seq(d_wtte_train, deep=False)

        # Set test data
        d_wtte_test = self.dtest[[self.id_col, self.tfs_col, self.tte_col] + self.wtte.features]
        # Scale/Normalize features using the scaler from the training data
        d_wtte_test[self.wtte.features] = self.wtte.scaler.transform(d_wtte_test[self.wtte.features])
        # Build test tensor
        x_wtte_test, y_wtte_test = self.wtte.build_seq(d_wtte_test, deep=False)

        # Fit WTTE model
        self.wtte.fit(x_wtte_train, y_wtte_train, x_wtte_test, y_wtte_test)
        # Plot training history
        self.wtte.plot_history_eval(show=False, file='wtte-history.png')
        # Plot training weights
        self.wtte.plot_weights(show=False, file='wtte-weights.png')

        # Build deep validation tensor
        # In this case, we need the complete (deep) sequences in order to analize the model performance
        x_wtte_deep, y_wtte_deep = self.wtte.build_seq(d_wtte_test, deep=True)

        # Get sequence lengths
        self.wtte.sls = self.wtte.get_seq_lengths(y_wtte_deep)
        # Predict
        y_wtte_hat = self.wtte.predict(x_wtte_deep)
        # Set results
        self.wtte.set_results(y_wtte_hat)

        # Plot the distribution of Weibull alpha and beta parameters
        self.wtte.plot_params_dist(loc=-1, show=False, file='wtte-params-dist.png')
        # Plot each customer Weibull alpha and beta parameters over time
        self.wtte.plot_params_seq(show=False, file='wtte-params-seq.png')

        return self
    
    def fit_xgb(self) -> Self:
        """
        Fit the XGBoost model, using the Weibull Time To Event model predictions as input,
        along with the features defined for the model.
        """
        # Set prediction data
        d_wtte_pred = self.data[[self.id_col, self.tfs_col, self.tte_col] + self.wtte.features]
        # Scale/Normalize features using the scaler from the training data
        d_wtte_pred[self.wtte.features] = self.wtte.scaler.transform(d_wtte_pred[self.wtte.features])
        # Build prediction tensor
        x_wtte_pred, y_wtte_pred = self.wtte.build_seq(d_wtte_pred, deep=True)

        # Get sequence lengths
        self.wtte.sls = self.wtte.get_seq_lengths(y_wtte_pred)
        # Predict
        y_wtte_hat = self.wtte.predict(x_wtte_pred)
        # Set results
        self.wtte.set_results(y_wtte_hat)

        # Set train data
        # Include the Weibull Time To Event model predictions
        d_xgb_train = self.dtrain.merge(
            self.wtte.results[[self.id_col, self.tfs_col, self.wa_col, self.wb_col]],
            on=[self.id_col, self.tfs_col], how='left'
        )[[self.id_col, self.tfs_col, self.tte_col] + self.xgb.features]
        # Build train tensor
        x_xgb_train, y_xgb_train = self.xgb.build_seq(d_xgb_train)

        # Set test data
        # Include the Weibull Time To Event model predictions
        d_xgb_test = self.dtest.merge(
            self.wtte.results[[self.id_col, self.tfs_col, self.wa_col, self.wb_col]],
            on=[self.id_col, self.tfs_col], how='left'
        )[[self.id_col, self.tfs_col, self.tte_col] + self.xgb.features]
        # Build validation tensor
        x_xgb_test, y_xgb_test = self.xgb.build_seq(d_xgb_test)

        # Fit XGBoost model
        self.xgb.fit(x_xgb_train, y_xgb_train, x_xgb_test, y_xgb_test)
        # Plot training history
        self.xgb.plot_history_eval(show=False, file='xgb-history.png')

        # Predict
        y_xgb_hat = self.xgb.predict(x_xgb_test)
        # Set results
        self.xgb.set_results(y_xgb_hat, y_xgb_test)

        return self

    def fit(self) -> Self:
        """
        Fit the ensemble.
        """
        self.fit_wtte()
        self.fit_xgb()

        # Set model results from XGBoost predictions
        self.set_results()
        # Compute and set model scores
        self.set_scores()

        return self

    def predict(self) -> Self:
        """
        Predict the churn probability for the customers in the data.
        """
        # Set prediction data
        d_wtte_pred = self.data[[self.id_col, self.tfs_col, self.tte_col] + self.wtte.features]
        # Scale/Normalize features using the scaler from the training data
        d_wtte_pred[self.wtte.features] = self.wtte.scaler.transform(d_wtte_pred[self.wtte.features])
        # Build prediction tensor
        x_wtte_pred, y_wtte_pred = self.wtte.build_seq(d_wtte_pred, deep=True)

        # Get sequence lengths
        self.wtte.sls = self.wtte.get_seq_lengths(y_wtte_pred)
        # Predict
        y_wtte_hat = self.wtte.predict(x_wtte_pred)
        # Set results
        self.wtte.set_results(y_wtte_hat)

        # Set prediction data
        # Include the Weibull Time To Event model predictions
        d_xgb_pred = self.data.merge(
            self.wtte.results[[self.id_col, self.tfs_col, self.wa_col, self.wb_col]],
            on=[self.id_col, self.tfs_col], how='left'
        )[[self.id_col, self.tfs_col, self.tte_col] + self.xgb.features]
        # Build prediction tensor
        x_xgb_pred, _ = self.xgb.build_seq(d_xgb_pred)

        # Predict
        y_xgb_hat = self.xgb.predict(x_xgb_pred)
        # Set results
        self.xgb.set_results(y_xgb_hat)

        # Set model results from XGBoost predictions
        self.set_results()

        return self

    def set_results(
        self,
        results: Optional[pd.DataFrame] = None
    ) -> Self:
        """
        Set the results DataFrame with predicted results, along with other computed values.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Predicted values.
            If not provided, XGBoost model results will be used.
        """
        if results is not None:
            dr = results.copy()
        else:
            dr = self.xgb.results.copy()
        
        rcols = list(dr.columns)

        # Set target column to 1 if the prediction is greater than the threshold, 0 otherwise
        dr['tgt'] = (dr['pred'] > self.thr).astype(int)

        # Get cluster labels for each customer sequence
        if dr['tgt'].unique().shape[0] == 2:  # We have 2 classes (churn, no churn)
            # For customers that did not churn (0) -> 3 segments
            c_0 = self.get_clusters(dr[dr['tgt'] == 0]['pred'], 3)
            # For customers that churned (1) -> 2 segments
            c_1 = self.get_clusters(dr[dr['tgt'] == 1]['pred'], 2) + 3
            dr['segment'] = pd.concat([c_0, c_1], axis=0)
        else:  # Only 1 class available
            # Cluster all customers into 5 segments
            dr['segment'] = self.get_clusters(dr['pred'], 5)

        # Set WTTE results
        if self.wa_col not in dr.columns:
            if self.wtte is not None:
                dr = dr.merge(
                    self.wtte.results[[self.id_col, self.tfs_col, self.wa_col, self.wb_col]],
                    on=[self.id_col, self.tfs_col], how='left'
                ).rename(columns={
                    self.wa_col: 'wa',
                    self.wb_col: 'wb'
                })
            else:
                dr[self.wa_col] = np.NaN
                dr[self.wb_col] = np.NaN

        # Set Momentum and customer IDs (for identification)
        dr = dr.merge(
            self.data[[self.id_col, self.tfs_col, self.cid_col, 'momentum']],
            on=[self.id_col, self.tfs_col], how='left'
        ).rename(columns={
            self.cid_col: 'cid'
        })

        # Sort and select columns
        self.results = dr.sort_values([self.id_col, self.tfs_col])[rcols + [
            'tgt', 'segment', 'wa', 'wb', 'momentum', 'cid'
        ]]

        return self

    def set_scores(
        self,
        results: Optional[pd.DataFrame] = None,
        loc: Optional[int] = -1
    ) -> Self:
        """
        Set the results DataFrame with the predicted values.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Predicted values.
            If not provided, XGBoost model results will be used.
        loc : int, optional
            Location in each customer sequence.
            If provided, only the value at the given location will be used.
            If -1, it will use the last sequence prediction for each customer.
        """
        if results is not None:
            dr = results.copy()
        else:
            dr = self.xgb.results.copy()

        if self.tfs_col in dr.columns and loc is not None:
            dr = dr.sort_values([self.id_col, self.tfs_col]).groupby(self.id_col).nth(loc).reset_index()

        if 'true' in dr.columns:
            self.scores['auc'] = roc_auc_score(dr.true.values, dr.pred.values)

            self.thr, thrs, f1, precision, recall = self.precision_recall(dr.true.values, dr.pred.values)
            best = thrs.index(self.thr)

            self.scores['f1'] = f1[best]
            self.scores['precision'] = precision[best]
            self.scores['recall'] = recall[best]

            y_tgt = np.array([1 if i > self.thr else 0 for i in dr.pred.values])
            self.scores['acc'] = accuracy_score(dr.true.values, y_tgt)

        # Plot scores summary
        self.plot_scores(show=False, file='scores.png')
        # Plot histogram of predicted probabilities for each customer sequence
        self.plot_histogram(show=False, file='histogram.png')

        return self
    
    def get_clusters(
        self,
        y: pd.Series,
        n: int
    ) -> pd.Series:
        """
        Cluster the given series using KMeans, in order to split predictions
        into different segment groups or labels.

        Parameters
        ----------
        y : pd.Series
            Input series.
        n : int
            Number of clusters.
        
        Returns
        -------
        pd.Series
            Cluster labels.
        """
        cm = KMeans(
            n_clusters=n,
            random_state=self.seed
        ).fit(pd.DataFrame(y).values)

        clusters = np.arange(1, n + 1)
        labels = pd.Series([clusters[i] for i in cm.labels_], index=y.index)
        centers = pd.DataFrame(cm.cluster_centers_, index=clusters).loc[:, 0].sort_values()

        c = labels.apply(lambda x: centers.index[x - 1])

        return c
    
    def get_predictions(
        self,
        results: Optional[pd.DataFrame] = None,
        loc: Optional[int] = -1
    ) -> pd.DataFrame:
        """
        Get the predictions DataFrame.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Predicted values.
            If not provided, model prediction results will be used.
        loc : int, optional
            Location in each customer sequence.
            If provided, only the value at the given location will be used.
            If -1, it will use the last sequence prediction for each customer.
        """
        if results is not None:
            dr = results.copy()
        else:
            dr = self.results.copy()

        if self.tfs_col in dr.columns and loc is not None:
            dr = dr.sort_values([
                self.id_col, self.tfs_col
            ]).groupby(self.id_col).nth(loc).drop(columns=[self.tfs_col])

        return dr

    def precision_recall(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> tuple[float, list[float], list[float], list[float], list[float]]:
        """
        Calculate the precision, recall and F1 scores for the given true and predicted values.

        This scores are calculated for different threshold values, where the threshold is
        the probability limit to classify an observation as positive (1) or negative (0).
        The best threshold is chosen based on the best F1 score, choosing the threshold
        that maximizes the F1 score.

        This function returns a set of the following:

        - Threshold:
            The threshold is the probability limit to classify an observation
            as positive (1) or negative (0).
        - Thresholds:
            A list of threshold values used to calculate the precision, recall and F1 scores,
            which will serve as index for the precision, recall and F1 scores.
        - F1 score:
            The F1 score is the harmonic mean of the precision and recall.
        - Precision:
            Number of true positives (Tp) over the number of true positives plus
            the number of false positives (Fp).
            A precission of 0.6 means that when it predicts a customer will churn,
            it will be correct 60% of the time.
        - Recall:
            Number of true positives (Tp) over the number of true positives plus
            the number of false negatives (Fn).
            A recall of 0.4 means that it will correctly identify 40% of
            the customers that will churn.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.
        
        Returns
        -------
        tuple[float, list[float], list[float], list[float], list[float]]
            A tuple with the following values:
                - best classifier threshold
                - all index thresholds
                - F1 scores
                - precision scores
                - recall scores
        """
        f1 = []
        precision = []
        recall = []

        thrs = [x / 100. for x in range(0, 105, 5)]

        for x in thrs:
            t_pred = np.array([1. if i > x else 0. for i in y_pred])
            f1.append(f1_score(y_true, t_pred, pos_label=1))
            precision.append(precision_score(y_true, t_pred, pos_label=1))
            recall.append(recall_score(y_true, t_pred, pos_label=1))

        # Select the best threshold by finding the maximum F1 score
        # and its corresponding threshold index. This threshold will
        # be used as the probability limit to classify the target
        # variable as churned (1) or not churned (0).
        best = np.argmax(f1)
        thr = thrs[best]

        return thr, thrs, f1, precision, recall

    def plot_scores(
        self,
        results: Optional[pd.DataFrame] = None,
        loc: Optional[int] = -1,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the ROC curve, precision/recall, feature importances and confusion matrix,
        for the given true and predicted values, into a 2x2 grid of subplots.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Predicted values.
            If not provided, XGBoost model results will be used.
        loc : int, optional
            Location in each customer sequence.
            If provided, only the value at the given location will be used.
            If -1, it will use the last sequence prediction for each customer.
        show : bool, optional
            If True, the plot will be displayed.
        file : str, optional
            If provided, the plot will be saved to the given file.
        """
        if results is not None:
            dr = results.copy()
        else:
            dr = self.results.copy()

        if self.tfs_col in dr.columns and loc is not None:
            dr = dr.sort_values([self.id_col, self.tfs_col]).groupby(self.id_col).nth(loc).reset_index()

        summary = ['Thereshold: {}%'.format(format_number(self.thr * 100, 0))]
        for key, val in self.scores.items():
            summary.append('{}: {}%'.format(key.upper(), format_number(val * 100, 0)))

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 14), constrained_layout=True)
        axs = axs.flatten()

        self.plot_roc_curve(results=dr, loc=loc, ax=axs[0])
        self.plot_precision_recall(results=dr, loc=loc, ax=axs[1])
        self.plot_feature_importances(ax=axs[2])
        self.plot_confusion_matrix(results=dr, loc=loc, ax=axs[3])

        plt.suptitle(' | '.join(summary), y=1.03)

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_roc_curve(
        self,
        results: Optional[pd.DataFrame] = None,
        loc: Optional[int] = -1,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the Receiver Operating Characteristic (ROC) curve for the true and predicted values.

        The ROC curve is the plot of the true positive rate (TPR) against the false positive rate (FPR)
        at each threshold setting, and illustrates the performance of a binary classifier model.

        From the ROC curve, the Area Under the Curve (AUC) can be calculated to get a single scalar
        value to summarize the model's performance, which is a measure of sensitivity and specificity
        of the model and is useful for comparing the performance of different models.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Predicted values.
            If not provided, XGBoost model results will be used.
        loc : int, optional
            Location in each customer sequence.
            If provided, only the value at the given location will be used.
            If -1, it will use the last sequence prediction for each customer.
        ax : plt.Axes, optional
            Axes object to plot the ROC curve on.
        show : bool, optional
            If True, the plot will be displayed.
        file : str, optional
            If provided, the plot will be saved to the given file.
        """
        if results is not None:
            dr = results.copy()
        else:
            dr = self.results.copy()

        if self.tfs_col in dr.columns and loc is not None:
            dr = dr.sort_values([self.id_col, self.tfs_col]).groupby(self.id_col).nth(loc).reset_index()

        y_true = dr['true']
        y_pred = dr['pred']

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        data = (
            pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr
            }) * 100
        ).set_index('fpr')
        data['diag'] = data.index.values

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = None
        
        ax.plot(
            data.index, data['tpr'].values,
            color=get_color('blue'),
            label=f'AUC = {format_number(roc_auc * 100)}%'
        )
        ax.plot(
            data.index, data['diag'].values,
            color=get_color('grey'), ls='--', lw=2, alpha=0.2,
            label=None
        )

        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_xticklabels([f'{format_number(i)}%' for i in ax.get_xticks()])

        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 110, 10))
        ax.set_yticklabels([f'{format_number(i)}%' for i in ax.get_yticks()])

        ax.legend(loc='lower right')
        ax.set_title('ROC Curve')

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_precision_recall(
        self,
        results: Optional[pd.DataFrame] = None,
        loc: Optional[int] = -1,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the precision, recall and F1 scores for the true and predicted values.

        This plot shows the trade-off between precision and recall for different thresholds.
        See the ``precision_recall`` function for more details.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Predicted values.
            If not provided, XGBoost model results will be used.
        loc : int, optional
            Location in each customer sequence.
            If provided, only the value at the given location will be used.
            If -1, it will use the last sequence prediction for each customer.
        ax : plt.Axes, optional
            Axes object to plot the precision/recall curve on.
        show : bool, optional
            If True, the plot will be displayed.
        file : str, optional
            If provided, the plot will be saved to the given file.
        """
        if results is not None:
            dr = results.copy()
        else:
            dr = self.results.copy()

        if self.tfs_col in dr.columns and loc is not None:
            dr = dr.sort_values([self.id_col, self.tfs_col]).groupby(self.id_col).nth(loc).reset_index()

        y_true = dr['true']
        y_pred = dr['pred']

        thr, thrs, f1, precision, recall = self.precision_recall(y_true, y_pred)
        best = thrs.index(thr)

        data = (
            pd.DataFrame({
                'thrs': thrs,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }) * 100
        ).set_index('thrs')

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = None
        
        ax.plot(
            data.index, data['f1'].values,
            color=get_color('blue'), ls='-', lw=3,
            label=f'F1 ({format_number(f1[best] * 100)}%)'
        )
        ax.plot(
            data.index, data['precision'].values,
            color=get_color('yellow'), ls='-.', lw=2,
            label=f'Precision ({format_number(precision[best] * 100)}%)'
        )
        ax.plot(
            data.index, data['recall'].values,
            color=get_color('green'), ls='-.', lw=2,
            label=f'Recall ({format_number(recall[best] * 100)}%)'
        )

        ax.axvline([thr * 100], color=get_color('grey'), ls='--', lw=1, alpha=0.2)

        ax.set_xlim(0, 100)
        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_xticklabels([f'{format_number(i)}%' for i in ax.get_xticks()])

        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 110, 10))
        ax.set_yticklabels([f'{format_number(i)}%' for i in ax.get_yticks()])

        ax.legend(loc='upper right')
        ax.set_title('Precision / Recall')

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_feature_importances(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the feature importances for the given model, sorted in descending order.

        The feature importances are calculated as the mean impurity decrease for each feature
        across all trees. The higher the feature importance, the more important the feature is
        for the model's predictions.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes object to plot the feature importances on.
        show : bool, optional
            If True, the plot will be displayed.
        file : str, optional
            If provided, the plot will be saved to the given file.
        """
        if self.xgb is not None:
            fimp = pd.Series(
                self.xgb.model.feature_importances_, index=self.xgb.features
            ).sort_values(ascending=False)
        else:
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = None
        
        ax.barh(
            fimp.index, fimp.values,
            color=get_color('blue'), align='center', tick_label=fimp.index,
            label='Feature Importances'
        )
        ax.invert_yaxis()

        ax.set_title('Feature Importances')

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_confusion_matrix(
        self,
        results: Optional[pd.DataFrame] = None,
        loc: Optional[int] = -1,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the confusion matrix for the given true and predicted values.

        This plot shows the number of true positives, false positives,
        true negatives and false negatives. The diagonal elements represent the
        number of points for which the predicted label is equal to the true label,
        while the off-diagonal elements are those that are misclassified.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Predicted values.
            If not provided, XGBoost model results will be used.
        loc : int, optional
            Location in each customer sequence.
            If provided, only the value at the given location will be used.
            If -1, it will use the last sequence prediction for each customer.
        ax : plt.Axes, optional
            Axes object to plot the confusion matrix on.
        show : bool, optional
            If True, the plot will be displayed.
        file : str, optional
            If provided, the plot will be saved to the given file.
        """
        if results is not None:
            dr = results.copy()
        else:
            dr = self.results.copy()

        if self.tfs_col in dr.columns and loc is not None:
            dr = dr.sort_values([self.id_col, self.tfs_col]).groupby(self.id_col).nth(loc).reset_index()

        y_true = dr['true']
        y_pred = dr['pred']
        y_tgt = np.array([1 if i > self.thr else 0 for i in y_pred])

        classes = ['0', '1']
        cm = confusion_matrix(y_true, y_tgt)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 10))
        else:
            fig = None

        ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ticks = np.arange(len(classes))

        ax.set_xticks(ticks)
        ax.set_xticklabels(classes)
        ax.set_yticks(ticks)
        ax.set_yticklabels(classes)

        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i,
                format(cm[i, j], 'd'),
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black'
            )

        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')

        ax.set_title('Confusion Matrix')

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_histogram(
        self,
        results: Optional[pd.DataFrame] = None,
        loc: Optional[int] = -1,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the histogram of the predicted probabilities for each customer sequence.

        Parameters
        ----------
        results : pd.DataFrame, optional
            Predicted values.
            If not provided, model results will be used.
        loc : int, optional
            Location in each customer sequence.
            If provided, only the value at the given location will be used.
            If -1, it will use the last sequence prediction for each customer.
        ax : plt.Axes, optional
            Axes object to plot the histogram on.
        show : bool, optional
            If True, the plot will be displayed.
        file : str, optional
            If provided, the plot will be saved to the given file.
        """
        if results is not None:
            dr = results.copy()
        else:
            dr = self.results.copy()

        if self.tfs_col in dr.columns and loc is not None:
            dr = dr.sort_values([self.id_col, self.tfs_col]).groupby(self.id_col).nth(loc).reset_index()

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax.hist(
            dr['pred'] * 100, bins=50, color=get_color('blue'), rwidth=0.9
        )
        ax.axvline(self.thr * 100, color=get_color('grey'), ls='--', lw=1, alpha=0.5)

        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_xticklabels([f'{i}%' for i in ax.get_xticks()])

        ax.set_title('Prediction Probas | Thereshold: {}%'.format(format_number(self.thr * 100, 0)))

        self.plot_figure(fig=fig, show=show, file=file)
    
    def plot_figure(
        self,
        fig: plt.Figure = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot a figure.

        Parameters
        ----------
        fig : plt.Figure, optional
            Figure to plot.
            If not provided, a new figure will be created.
        show : bool, optional
            If True, the figure will be displayed.
        file : str, optional
            If provided, the figure will be saved to a file.
        """
        if fig is not None:
            if show:
                plt.show()

            if file:
                fig.savefig(self.get_path(file))

            fig.clear()

            plt.clf()
            plt.close('all')

    def set_path(self, path: str) -> str:
        """
        Set the path to save/load files.

        Parameters
        ----------
        path : str
            Path provided as base path for files.
            If not provided, the current working directory will be used.
            If the path does not exist, it will be created.

        Returns
        -------
        str
            Path to save/load files.
        """
        if not path:
            path = os.getcwd()
        elif not path.startswith('/'):
            path = f'{os.getcwd()}/{path}'

        if not os.path.exists(path):
            os.makedirs(path)
        
        return path

    def get_path(self, name: str) -> str:
        """
        Get the path to save/load a specific file.

        Parameters
        ----------
        name : str
            Name of the file.
        
        Returns
        -------
        str
            Path to save/load the file.
        """
        return f'{self.path}/{name}'

    def file_exists(self, name: str) -> bool:
        """
        Check if a file exists.
        """
        return os.path.isfile(self.get_path(name))

    def get_config(self) -> dict:
        """
        Get the model configuration.
        """
        return {
            k: v for k, v in self.__dict__.items()
            if k not in [
                'params', 'data', 'dtrain', 'dtest',
                'wtte', 'xgb', 'results',
                'seed', 'verbose', 'path'
            ]
        }

    def save(self) -> Self:
        """
        Save the ensemble models to files.
        """
        config = json.dumps(self.get_config())
        with open(self.get_path('config.json'), 'w') as fh:
            fh.write(config)

        params = json.dumps(self.params)
        with open(self.get_path('params.json'), 'w') as fh:
            fh.write(params)

        self.save_wtte()
        self.save_xgb()

        if not self.results.empty:
            self.results.to_json(self.get_path('results.json'))

        return self

    def load(self) -> Self:
        """
        Load the ensemble models from files.
        """
        if self.file_exists('config.json'):
            with open(self.get_path('config.json'), 'r') as fh:
                config = json.load(fh)
            for k, v in config.items():
                setattr(self, k, v)
        
        if self.file_exists('params.json'):
            with open(self.get_path('params.json'), 'r') as fh:
                params = json.load(fh)
            self.params = params

        self.wtte = self.load_wtte()
        self.xgb = self.load_xgb()

        if self.file_exists('results.json'):
            self.results = pd.read_json(self.get_path('results.json'))

        return self
