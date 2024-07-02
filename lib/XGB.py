import joblib
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.utils import shuffle
from xgboost.sklearn import XGBClassifier

from typing import Optional
from typing_extensions import Self

from .utils import get_color


class XGB(object):
    """
    XGBoost sliding window model for customer churn prediction.

    XGBoost (eXtreme Gradient Boosting) is an open-source software library
    which provides a gradient-boosted decision tree (GBDT) framework.
    It is widely used for both classification and regression problems.

    For the purpose of churn prediction, XGBoost is used to predict the probability
    of churn for each customer in a sliding window fashion, which is a common approach
    in time-to-event analysis. Instead of trying to predict the TTE directly we predict
    whether an event will happen within a preset timeframe, and use the probability of
    churn as a proxy for the probability of the event.
    For example: will the customer churn within the next 3 months?

    Parameters
    ----------
    features : list[str], optional
        List of feature names to be used in the model.
        If None, all columns except keys ('id' and 'tfs') will be used.
    min_tte : int, optional
        Minimum time to event for binary classification (positive if `tte` <= `min_tte`).
        Defaults to 1 (churn if customer churns in the current period or the next).
    seed : int, optional
        Base random seed for reproducibility.
    verbose : int, optional
        Verbosity level.
        If 0, no output will be printed.
        If 1, only important messages will be printed.
        If 2, all messages will be printed (including model fit and predict outputs).
        If 3, debug mode (all messages + model outputs + XGBoost logs).
    path : str, optional
        Path to store model files and images.
    kwargs : dict, optional
        Additional parameters to be used in the model.
    """

    # Column names
    id_col = 'id'  # Customer ID
    tfs_col = 'tfs'  # Periods from start date
    tte_col = 'tte'  # Periods until end date
    tgt_col = 'tgt'  # Target value (binary classification (positive if `tte` <= `min_tte`))

    def __init__(
        self,
        features: Optional[list[str]] = None,
        min_tte: int = 1,
        seed: int = 42,
        verbose: int = 0,
        path: Optional[str] = None,
        **kwargs
    ):
        # Execution params
        self.features = features or []
        self.min_tte = min_tte
        self.seed = seed
        self.verbose = verbose
        self.path = self.set_path(path)

        # Model params
        self.params = {
            'n': kwargs.get('n', 200),  # Number of trees
            'lr': kwargs.get('lr', 1e-2),  # Learning rate
            'max_depth': kwargs.get('max_depth', 16),  # Maximum depth of the tree
            'stop': kwargs.get('stop', 0),  # Early stopping rounds
            'metric': kwargs.get('metric', 'auc'),  # Metric to optimize
            'min_child_weight': kwargs.get('min_child_weight', 1),  # Minimum weight of a leaf
            'gamma': kwargs.get('gamma', 0),  # Minimum loss reduction required to split a leaf
            'weight_l1': kwargs.get('weight_l1', 0),  # L1 regularization coefficient
            'weight_l2': kwargs.get('weight_l2', 1),  # L2 regularization coefficient
            'dropout': kwargs.get('dropout', 0.2),  # Dropout rate
            'shuffle': kwargs.get('shuffle', False),  # Shuffle training data
            'reg_unb': kwargs.get('reg_unb', True)  # Regularize unbalanced classes
        }

        # Model instance
        self.model = None

        # Results
        self.history = pd.DataFrame()  # Training history
        self.results = pd.DataFrame()  # Predicted results

    def build_model(self) -> XGBClassifier:
        """
        Build the XGBoost model.

        Returns
        -------
        model : XGBClassifier
            XGBoost model.
        """
        subsample = None
        colsample_bytree = None
        if isinstance(self.params['dropout'], np.ndarray):
            subsample, colsample_bytree = 1.0 - self.params['dropout']
        elif self.params['dropout'] is not None and self.params['dropout'] > 0:
            subsample = colsample_bytree = 1.0 - self.params['dropout']

        model = XGBClassifier(
            n_estimators=self.params['n'],
            learning_rate=self.params['lr'],
            max_depth=self.params['max_depth'],
            booster='gbtree',
            objective='binary:logistic',
            min_child_weight=self.params['min_child_weight'],
            gamma=self.params['gamma'],
            reg_alpha=self.params['weight_l1'],
            reg_lambda=self.params['weight_l2'],
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            verbosity=self.verbose,
            n_jobs=-1,
            random_state=self.seed
        )

        return model

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Self:
        """
        Fit the model to the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training input sequences.
        y_train : np.ndarray
            Training target sequences.
        x_test : np.ndarray
            Validation input sequences.
        y_test : np.ndarray
            Validation target sequences.
        """
        # Build sequences
        x_train, y_train, _ = self.input_seq(x_train, y_train, shuffle_data=self.params['shuffle'])
        x_test, y_test, _ = self.input_seq(x_test, y_test)

        # Load model
        self.model = self.build_model()

        # In case of unbalanced classes (which is usually the case in churn prediction),
        # we can use class balancing by setting the scale_pos_weight parameter.
        # This value is calculated as the ratio of the number of negative to positive instances.
        unb_ratio = float(round(len(y_train[y_train == 0]) / len(y_train[y_train == 1]))) if self.params['reg_unb'] > 0 else None

        # Early stopping rounds
        estop = self.params['stop'] if self.params['stop'] > 0 else None

        # Fit
        self.model.set_params(
            scale_pos_weight=unb_ratio
        ).fit(
            x_train, y_train,
            eval_set=[(x_train, y_train), (x_test, y_test)],
            eval_metric=self.params['metric'],
            early_stopping_rounds=estop
        )
        h = self.model.evals_result()

        self.history = pd.DataFrame.from_dict({
            'metric': h['validation_0'][self.params['metric']],
            'val_metric': h['validation_1'][self.params['metric']]
        }, dtype='float')

        return self

    def predict(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Predict the probability of churn for each of the given data.

        Parameters
        ----------
        x : np.ndarray
            Input sequences.
        
        Returns
        -------
        np.ndarray
            Predicted values.
        """
        # Load model
        if self.model is None:
            self.load()

        # Build sequences
        x, _, z = self.input_seq(x)

        # Predict
        y_pred = self.model.predict_proba(x)[:, 1]

        # Build output
        y_pred = self.output_seq(y_pred, z)

        return y_pred

    def set_results(
        self,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None
    ) -> Self:
        """
        Set the results DataFrame with the predicted values.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values.
        y_true : np.ndarray, optional
            True values.
            If provided, it can be used as validation data.
        """
        y_pred = self.seq_to_df(y_pred)

        if y_true is not None:
            y_true = self.seq_to_df(y_true)

        self.results = y_pred.rename(columns={
            self.tgt_col: 'pred'
        }).sort_values([self.id_col, self.tfs_col])

        if y_true is not None:
            self.results = self.results.merge(
                y_true, on=[self.id_col, self.tfs_col], how='left'
            ).rename(columns={
                self.tgt_col: 'true'
            })

        self.results[[self.id_col, self.tfs_col]] = self.results[[self.id_col, self.tfs_col]].astype(int)

        return self

    def build_seq(
        self,
        data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the input and target sequences for the model.

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple with features and target sequences.
        """
        if self.features is None:
            self.features = list(data.columns[~data.columns.isin([
                self.id_col, self.tfs_col, self.tte_col, self.tgt_col
            ])])

        # If `tfs` column does not exist, create it with zeros (prediction data)
        if self.tfs_col not in list(data.columns):
            data[self.tfs_col] = 0

        # If `tgt` column does not exist, infer from `tte` or create it with zeros (prediction data)
        if self.tgt_col not in list(data.columns):
            if self.tte_col in list(data.columns):
                data[self.tgt_col] = data[self.tte_col].between(0, self.min_tte).astype(int)
            else:
                data[self.tgt_col] = 0

        # Split features and target sequence arrays
        x = data[[self.id_col, self.tfs_col, self.tgt_col] + self.features].values
        y = data[[self.id_col, self.tfs_col, self.tgt_col]].values

        return x, y

    def input_seq(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        shuffle_data: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the input sequences for the XGBoost model fit.

        Parameters
        ----------
        x : np.ndarray
            Input tensor.
        y : np.ndarray, optional
            Target tensor.
            If not provided, data will be used for prediction purposes.
        
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple with the input, target and key sequences.
        """
        x = np.array(x)
        if y is not None:
            y = np.array(y)

        z = x[:, :2]  # Key columns to be used for indexing results
        x = x[:, 3:]  # Input features
        if y is not None:
            y = y[:, 2]  # Target column

        if shuffle_data:
            if y is not None:
                x, y, z = shuffle(x, y, z, random_state=self.seed)
            else:
                x, z = shuffle(x, z, random_state=self.seed)

        return x, y, z

    def output_seq(
        self,
        y: np.ndarray,
        z: np.ndarray
    ) -> np.ndarray:
        """
        Prepare the output sequences of the XGBoost model predictions.

        Parameters
        ----------
        y : np.ndarray
            Predicted tensor.
        z : np.ndarray
            Key tensor.
        
        Returns
        -------
        np.ndarray
            Predicted sequences tensor.
        """
        y = np.array(y)
        z = np.array(z)

        n_seq = y.shape[0]

        y = y.reshape(n_seq, 1)
        seq = np.hstack([z, y])

        return seq

    def seq_to_df(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Convert the given tensor to a DataFrame.

        Parameters
        ----------
        x : np.ndarray
            Input tensor.
        y : np.ndarray, optional
            Target tensor.
        
        Returns
        -------
        pd.DataFrame
            Output DataFrame.
        """
        x = np.array(x)
        if y is not None:
            y = np.array(y)

        n_feat = len(self.features)

        if y is not None:
            x = x[:, -n_feat:]
            df = pd.DataFrame(np.hstack([y, x]), columns=[self.id_col, self.tfs_col, self.tgt_col] + self.features)
        else:
            df = pd.DataFrame(x, columns=[self.id_col, self.tfs_col, self.tgt_col])

        return df

    def plot_history_eval(
        self,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the training and validation loss over time,
        showing how the model's performance changes as the model learns.

        Parameters
        ----------
        ax : plt.Axes, optional
            Axes to plot the figure.
            If not provided, a new figure will be created.
        show : bool, optional
            If True, the figure will be displayed.
        file : str, optional
            If provided, the figure will be saved to a file.
        """
        if self.history.empty:
            return
        
        columns = ['metric', 'val_metric']

        df = self.history[columns]

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax.plot(
            df.index, df['metric'].values, color=get_color('blue'), label='training metric'
        )
        ax.plot(
            df.index, df['val_metric'].values, color=get_color('orange'), label='validation metric'
        )

        ax.set_xticks(np.arange(0, len(df) + 5, 5))

        ax.legend(loc='upper left')
        ax.set_title('epochs')

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

    def model_exists(self) -> bool:
        """
        Check if the model exists.
        """
        return self.file_exists('model.pkl')
    
    def model_save(self, model: XGBClassifier) -> str:
        """
        Save the model to a file.
        """
        path = self.get_path('model.pkl')
        joblib.dump(model, path)

        return path

    def model_load(self) -> XGBClassifier:
        """
        Load the model from a file.
        """
        path = self.get_path('model.pkl')

        return joblib.load(path)
    
    def get_config(self) -> dict:
        """
        Get the model configuration.
        """
        return {
            k: v for k, v in self.__dict__.items()
            if k not in [
                'model', 'params', 'history', 'results',
                'seed', 'verbose', 'path'
            ]
        }

    def save(self) -> Self:
        """
        Save the model, history, and other attributes to files.
        """
        config = json.dumps(self.get_config())
        with open(self.get_path('config.json'), 'w') as fh:
            fh.write(config)

        params = json.dumps(self.params)
        with open(self.get_path('params.json'), 'w') as fh:
            fh.write(params)

        if self.model is not None:
            self.model_save(self.model)

        if self.history is not None:
            self.history.to_json(self.get_path('history.json'), double_precision=10)

        if not self.results.empty:
            self.results.to_json(self.get_path('results.json'))

        return self

    def load(self) -> Self:
        """
        Load the model, history, and other attributes from files.
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

        if self.model_exists():
            self.model = self.model_load()

        if self.file_exists('history.json'):
            self.history = pd.read_json(self.get_path('history.json'))

        if self.file_exists('results.json'):
            self.results = pd.read_json(self.get_path('results.json'))

        return self
