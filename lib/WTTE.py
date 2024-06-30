import joblib
import json
import keras
from keras import ops as K
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from itertools import groupby
from operator import itemgetter
from typing import Literal, Optional
from typing_extensions import Self

from .utils import get_color, is_array


class WTTE(object):
    """
    Weibull Time To Event Recurrent Neural Network (WTTE-RNN) model for customer churn prediction.
    Based on the excellent work by Egil Martinsson [1]

    The model consists of a recurrent neural network (RNN) that takes a sequence of customer
    info and usage data over time as input and predicts the time to churn (TTE)
    from the two parameters that control the shape of the Weibull distribution,
    a commonly used distribution in survival analysis and time-to-event models.

    [1] WTTE-RNN: Weibull Time To Event Recurrent Neural Network (Egil Martinsson, 2016)
    http://publications.lib.chalmers.se/records/fulltext/253611/253611.pdf

    Parameters
    ----------
    features : list[str], optional
        List of feature names to be used in the model.
        If None, all columns except keys ('id', 'tfs' and 'tte') will be used.
    min_tte : int, optional
        Minimum time to event for binary classification (positive if `tte` <= `min_tte`).
        Defaults to 1 (churn if customer churns in the current period or the next).
    max_sl : int, optional
        Maximum sequence length.
        If 0, the maximum sequence length will be set to the maximum sequence length from data.
    kind : Literal['discrete', 'continuous'], optional
        Weibull distribution kind.
    wlevel : Literal['epoch', 'batch'], optional
        Level to save the weights and bias.
        If 'epoch', the weights and bias are saved after each epoch.
        If 'batch', the weights and bias are saved after each batch.
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
        features: Optional[list[str]] = None,
        min_tte: int = 1,
        max_sl: int = 0,
        kind: Literal['discrete', 'continuous'] = 'discrete',
        wlevel: Literal['epoch', 'batch'] = 'epoch',
        seed: int = 42,
        verbose: int = 0,
        path: Optional[str] = None,
        **kwargs
    ):
        # Column names
        self.id_col = 'id'  # User ID
        self.seq_col = 'seq'  # Current sequence
        self.tfs_col = 'tfs'  # Periods from start date
        self.tte_col = 'tte'  # Periods until end date

        # Weibull parameters to be predicted
        self.wa_col = 'wa'  # Alpha
        self.wb_col = 'wb'  # Beta

        # Execution params
        self.features = features or []
        self.min_tte = min_tte
        self.max_sl = max_sl
        self.kind = kind
        self.wlevel = wlevel
        self.seed = seed
        self.verbose = verbose
        self.path = self.set_path(path)

        # Model config
        self.params = {
            'nn': kwargs.get('nn', 0),  # Number of neurons in each LSTM layer (0 = max_sl)
            'hl': kwargs.get('hl', 2),  # Number of LSTM layers
            'lr': kwargs.get('lr', 1e-4),  # Learning rate
            'epochs': kwargs.get('epochs', 200),  # Number of epochs
            'batch': kwargs.get('batch', 512),  # Batch size
            'lr_decay': kwargs.get('lr_decay', 0),  # Learning rate decay (number of epochs)
            'stop': kwargs.get('stop', 0),  # Early stopping patience (number of epochs)
            'dropout': kwargs.get('dropout', 0.1),  # Dropout rate
            'weight_l1': kwargs.get('weight_l1', 0),  # L1 regularization penalty
            'weight_l2': kwargs.get('weight_l2', 1e-5),  # L2 regularization penalty
            'init_alpha': kwargs.get('init_alpha', None),  # Initialize alpha value (None = compute from data)
            'max_beta': kwargs.get('max_beta', 2.),  # Max beta value (regularize beta with value > 1)
            'shuffle': kwargs.get('shuffle', False),  # Shuffle data before training
            'epsilon': kwargs.get('epsilon', 1e-8)  # Epsilon value to avoid numerical instability
        }

        # Model instance
        self.model = None

        # Preprocessing objects
        self.encoder = None
        self.scaler = None
        self.imputer = None

        # Model params
        self.mask = np.NaN  # Mask value
        self.a_mean = np.NaN  # Mean of alpha values
        self.a_std = np.NaN  # Standard deviation of alpha values
        self.b_mean = np.NaN  # Mean of beta values
        self.b_std = np.NaN  # Standard deviation of beta values
        self.init_alpha = np.NaN  # Initial alpha value
        self.max_beta = np.NaN  # Maximum beta value

        # Results
        self.history = pd.DataFrame()  # Training history
        self.sls = pd.DataFrame()  # Customer sequence lengths
        self.results = pd.DataFrame()  # Predicted results
        self.watcher = None  # Weights watcher object

    def build_model(self) -> keras.Model:
        """
        Build the model architecture.

        Returns
        -------
        model : keras.Model
            Model architecture.
        """
        if np.isnan(self.mask):
            self.mask = -1.
        if self.params['nn'] == 0:
            self.params['nn'] = self.max_sl if self.max_sl > 0 else 1

        regularizer = None
        if self.params['weight_l1'] > 0 and self.params['weight_l2'] > 0:
            regularizer = keras.regularizers.l1_l2(l1=self.params['weight_l1'], l2=self.params['weight_l2'])
        elif self.params['weight_l1'] > 0:
            regularizer = keras.regularizers.l1(self.params['weight_l1'])
        elif self.params['weight_l2'] > 0:
            regularizer = keras.regularizers.l2(self.params['weight_l2'])

        model = keras.models.Sequential()

        model.add(
            keras.layers.Masking(
                mask_value=self.mask,
                input_shape=(None, len(self.features))
            )
        )

        for _ in np.arange(self.params['hl']):
            model.add(
                keras.layers.LSTM(
                    units=self.params['nn'],
                    activation='tanh',
                    dropout=self.params['dropout'],
                    kernel_regularizer=regularizer,
                    return_sequences=True
                )
            )

            model.add(
                keras.layers.LayerNormalization(
                    epsilon=keras.config.epsilon()
                )
            )

        model.add(
            keras.layers.Dense(2)
        )
        model.add(
            keras.layers.Activation(
                self.activation
            )
        )

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.params['lr'],
                clipnorm=1.
            ),
            loss=self.loss
        )

        return model

    def activation(
        self,
        ab: tf.Tensor
    ) -> tf.Tensor:
        """
        Activation function for the output layer.

        First we unstack the tensor into its two components (alpha and beta)
        in order to apply a different activation function to each component separately,
        regularize using given configuration and stack back into a single tensor.

        As the author of the original paper mentions,
        the choice of activation functions is motivated by the following reasons:

        - Alpha:

            (activation)
            Exponential units seems to give faster training than
            the original paper's softplus units. Makes sense due to logarithmic
            effect of change in alpha.
            (initialization)
            To get faster training and fewer exploding gradients,
            initialize alpha to be around its scale when beta is around 1.0,
            approx the expected value/mean of training tte.
            Because we're lazy, we want the correct scale of output built
            into the model so initialize implicitly,
            multiply assumed exp(0)=1 by scale factor `init_alpha`.

        - Beta:

            (activation)
            We want slow changes when beta -> 0 so Softplus made sense in the original
            paper but we get similar effect with sigmoid. It also has nice features.
            (regularization) Use `max_beta` to implicitly regularize the model.
            (initialization) Fixed to begin moving slowly around 1.0.

        Parameters
        ----------
        ab : tf.Tensor
            Output tensor.

        Returns
        -------
        tf.Tensor
            Output tensor with activation function applied.
        """
        epsilon = keras.config.epsilon()

        # Unstack tensor
        a = ab[..., 0]  # Alpha values
        b = ab[..., 1]  # Beta values

        # Exponential activation for alpha
        a = self.init_alpha * K.exp(a)

        # Sigmoid activation for beta
        if self.max_beta > 1:
            b = b - K.log(self.max_beta - 1.)
        b = self.max_beta * K.clip(K.sigmoid(b), epsilon, 1. - epsilon)

        # Restack tensor
        x = K.stack([a, b], axis=-1)

        return x

    def loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Loss function for the model.

        The loss function is defined as the negative log-likelihood of the predicted values
        given the true values, using the Weibull distribution.

        Having both Hazard Function (HF) and Cumulative Hazard Function (CHF) defined as:

        HF(t) = (t/alpha)^(beta-1) * (beta/alpha)
        CHF(t) = (t/alpha)^beta

        For the continuous Weibull distribution, the log-likelihood is given by:
        L = u * log(HF(t)) - CHF(t)
        See `loglik_continuous` for more details.

        For the discrete Weibull distribution, the log-likelihood is given by:
        L = u * log(CHF(t+1) - CHF(t)) - CHF(t+1)
        See `loglik_discrete` for more details.

        Parameters
        ----------
        y_true : tf.Tensor
            True values.
        y_pred : tf.Tensor
            Predicted values.

        Returns
        -------
        tf.Tensor
            Loss value.
        """
        epsilon = keras.config.epsilon()

        # Unstack tensors
        a_true = y_true[..., 0]
        b_true = y_true[..., 1]
        a_pred = y_pred[..., 0]
        b_pred = y_pred[..., 1]

        if self.kind == 'discrete':
            loglik = self.loglik_discrete(a_true, b_true, a_pred, b_pred)
        else:
            loglik = self.loglik_continuous(a_true, b_true, a_pred, b_pred)

        # Clip values to avoid numerical instability
        loglik = K.clip(loglik, K.log(epsilon), K.log(1. - epsilon))

        return -K.mean(loglik)

    def loglik_continuous(
        self,
        a_true: tf.Tensor,
        b_true: tf.Tensor,
        a_pred: tf.Tensor,
        b_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Log-likelihood function for the continuous Weibull distribution.

        Having both Hazard Function (HF) and Cumulative Hazard Function (CHF) defined as:

        HF(t) = (t/alpha)^(beta-1) * (beta/alpha)
        CHF(t) = (t/alpha)^beta

        The log-likelihood function for the Weibull distribution is given by:

        L = u * log(HF(t)) - CHF(t) = u * log((t/alpha)^(beta-1) * (beta/alpha)) - (t/alpha)^beta

        which, after a series of transformations, can be written as:

        L = u * (beta * log(t/alpha) + log(beta)) - (t/alpha)^beta

        where `t` and `u` are the true values of `alpha` and `beta`, respectively.

        See p.35 of WTTE-RNN: Weibull Time To Event Recurrent Neural Network (Egil Martinsson, 2016).

        Parameters
        ----------
        a_true : tf.Tensor
            True alpha values.
        b_true : tf.Tensor
            True beta values.
        a_pred : tf.Tensor
            Predicted alpha values.
        b_pred : tf.Tensor
            Predicted beta values.

        Returns
        -------
        tf.Tensor
            Log-likelihood values.
        """
        epsilon = keras.config.epsilon()

        # Compute (t/alpha)
        # Add a small number (epsilon) to avoid numerical instability
        ytp = (a_true + epsilon) / a_pred

        # Weibull log-likelihood
        loglik = b_true * (
            b_pred * K.log(ytp) + K.log(b_pred)
        ) - K.power(ytp, b_pred)

        return loglik

    def loglik_discrete(
        self,
        a_true: tf.Tensor,
        b_true: tf.Tensor,
        a_pred: tf.Tensor,
        b_pred: tf.Tensor
    ) -> tf.Tensor:
        """
        Log-likelihood function for the discrete Weibull distribution.

        Having both (discrete) Hazard Function (DHF) and Cumulative Hazard Function (DCHF) defined as:

        DHF(t) = CHF(t+1) - CHF(t) = ((t+1)/alpha)^beta - (t/alpha)^beta
        DCHF(t) = CHF(t+1) = ((t+1)/alpha)^beta

        The log-likelihood function for the Weibull distribution is given by:

        L = u * log(DHF(t)) - DCHF(t) = u * log(((t+1)/alpha)^beta - (t/alpha)^beta) - ((t+1)/alpha)^beta

        which, after a series of transformations, can be written as:

        L = u * log(exp(((t+1)/alpha)^beta - (t/alpha)^beta) - 1) - ((t+1)/alpha)^beta

        where `t` and `u` are the true values of `alpha` and `beta`, respectively.

        See p.35 of WTTE-RNN: Weibull Time To Event Recurrent Neural Network (Egil Martinsson, 2016).

        Parameters
        ----------
        a_true : tf.Tensor
            True alpha values.
        b_true : tf.Tensor
            True beta values.
        a_pred : tf.Tensor
            Predicted alpha values.
        b_pred : tf.Tensor
            Predicted beta values.

        Returns
        -------
        tf.Tensor
            Log-likelihood values.
        """
        epsilon = keras.config.epsilon()

        # Compute Weibull CHF for t and t+1
        # Add a small number (epsilon) to avoid numerical instability
        chf0 = K.power((a_true + epsilon) / a_pred, b_pred)
        chf1 = K.power((a_true + 1.) / a_pred, b_pred)

        # Weibull log-likelihood
        loglik = b_true * K.log(
            K.exp(chf1 - chf0) - 1.
        ) - chf1

        return loglik
    
    def init_model(
        self,
        x: np.ndarray,
        y: np.ndarray
    ):
        """
        Initialize the model with the given training data.

        Parameters
        ----------
        x : np.ndarray
            Input tensor.
        y : np.ndarray, optional
            Target tensor.
        """
        x = np.array(x)[..., 3:]  # Input features
        y = np.array(y)[..., 3:]  # Alpha and Beta (`wa`, `wb`)

        self.mask = np.nanmin(x) - 1
        self.a_mean = np.nanmean(y[..., 0])
        self.a_std = np.nanstd(y[..., 0])
        self.b_mean = np.nanmean(y[..., 1])
        self.b_std = np.nanstd(y[..., 1])

        # If `init_alpha` is not provided, use actual data to compute it
        if self.params['init_alpha'] is not None:
            self.init_alpha = self.params['init_alpha']
        else:
            # MLE of Geometric Distribution
            self.init_alpha = -1. / np.log(1. - 1. / (self.a_mean + 1.))
        
        # Set maximum beta value
        self.max_beta = self.params['max_beta']

        # Print model params
        if self.verbose > 0:
            print(f'{self.kind} -> Max Length: {self.max_sl} | Mask: {self.mask:.2f}')
            print(f'Alpha Mean: {self.a_mean:.2f} | Beta Mean: {self.b_mean:.2f}')
            print(f'Init Alpha: {self.init_alpha:.2f} | Max Beta: {self.max_beta:.2f}')
        
        return self

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray
    ) -> Self:
        """
        Fit the model to the training data sequences.

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
        keras.backend.clear_session()
        keras.config.set_epsilon(self.params['epsilon'])
        tf.random.set_seed(self.seed)
        
        # Initialize model
        self.init_model(x_train, y_train)

        # Build input sequences
        x_train, y_train, _ = self.input_seq(x_train, y_train)
        x_test, y_test, _ = self.input_seq(x_test, y_test)

        # Load model
        self.model = self.build_model()
        if self.verbose > 0:
            self.print_model_summary()

        nant = keras.callbacks.TerminateOnNaN()
        callbacks = [nant]

        # Set learning rate decay
        if self.params['lr_decay'] > 0:
            lr_decay = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=self.params['lr_decay'],
                factor=0.1,
                min_lr=self.params['lr'] * 0.1,
                verbose=self.verbose
            )
            callbacks.append(lr_decay)

        # Set early stopping
        if self.params['stop'] > 0:
            stop = keras.callbacks.EarlyStopping(patience=self.params['stop'])
            callbacks.append(stop)

        watcher = WeightWatcher(level=self.wlevel)
        callbacks.append(watcher)

        if self.verbose > 1:
            board = keras.callbacks.TensorBoard(
                log_dir=self.get_path('logs'),
                histogram_freq=1,
                write_images=True,
                update_freq=self.wlevel
            )
            callbacks.append(board)

        # Fit model
        h = self.model.fit(
            x_train, y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch'],
            validation_data=(x_test, y_test),
            shuffle=self.params['shuffle'],
            verbose=self.verbose,
            callbacks=callbacks
        )

        # Set training history
        self.history = pd.DataFrame.from_dict(h.history, dtype='float')
        self.watcher = watcher

        if self.history.dropna().empty:
            raise ValueError('Fit failed.')

        return self

    def predict(
        self,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Predict alpha and beta parameters for each of the given sequences.

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
            self.load_model()

        # Build sequences
        x, _, z = self.input_seq(x)

        # Predict
        y_pred = self.model.predict(x)

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
        if self.sls.empty and y_true is not None:
            self.sls = self.get_seq_lengths(y_true)

        y_pred = self.rebuild_seq(y_pred, use_sls=True)
        y_pred['pred'] = self.weibull_proba(
            y_pred[[self.wa_col, self.wb_col]].values,
            t=(self.min_tte + 1)
        )

        self.results = y_pred.sort_values([self.id_col, self.tfs_col])

        if y_true is not None:
            y_true = self.rebuild_seq(y_true, use_sls=True)
            y_true['true'] = (
                y_true[self.wa_col].isin(np.arange(self.min_tte + 1)) & (y_true[self.wb_col] > 0)
            ).astype(int)

            self.results = self.results.merge(
                y_true[[self.id_col, self.tfs_col, 'true']],
                on=[self.id_col, self.tfs_col], how='left'
            )

        self.results[[self.id_col, self.tfs_col]] = self.results[[self.id_col, self.tfs_col]].astype(int)

        return self

    def build_seq(
        self,
        data: pd.DataFrame,
        deep: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build sequences from the given data.

        The input DataFrame consists of customer sequences, where each row contains
        the ID, info and usage data of a single customer in a specific period.
        Each customer can have 1 to N rows, representing the evolution of the customer
        position and usage over time, with the following columns:

        - id: Customer ID.
        - tfs: Periods from start date.
        - tte: Periods until churn date.
        - features: Customer info and usage data at this point of time.

        In order to use this data to train the RNN model, it is necessary to build sequences
        in a way that the model can learn the evolution of the customer position and usage.
        To do so, we must build a tensor with the following dimensions: (n_seq, max_sl, n_feat + 5),
        where:

        - n_seq: Number of sequences (customers).
        - max_sl: Maximum sequence length.
        - n_feat: Number of features.

        The last dimension contains the following components:

        - id: Customer ID.
        - seq: Current sequence.
        - tfs: Periods from start date.
        - a: Weibull alpha parameter (scale).
        - b: Weibull beta parameter (shape).
        - features: Customer info and usage data.

        If `deep` is True, all possible sequences will be generated for each customer,
        so if a user has more than `max_sl` periods of data, multiple sequences will be created.
        Otherwise, only one sequence will be created with the last `max_sl` periods.

        Parameters `a` (alpha/scale) and `b` (beta/shape) are the Weibull parameters to be predicted,
        representing the time until the customer churns and the confidence in this prediction, respectively.
        This parameters are computed from the `tte` column, which represents the number of periods
        until the customer churns, using the following logic:

        - If customer has already churned (uncensored data):
            - `a` is set to the number of periods until churn.
            - `b` is set to 1 (absolute confidence).
        - If customer has not churned yet (censored data):
            - `a` is set to the number of periods until the last period of data.
            - `b` is set to 0 (no confidence).

        Parameters
        ----------
        data : pd.DataFrame
            Input data.
        deep : bool, optional
            If True, all possible sequences will be generated for each customer.
            Otherwise, only one sequence will be generated for each customer.

        All rows in the resulting tensor will have the same length, so the sequences will be padded
        with NaN values to match the maximum sequence length.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple with features and target sequences.
        """
        if len(self.features) == 0:
            self.features = list(data.columns[~data.columns.isin([self.id_col, self.tfs_col, self.tte_col])])

        # If `tte` column does not exist, create it with zeros (prediction data)
        if self.tte_col not in list(data.columns):
            data[self.tte_col] = 0
        
        data = data[[self.id_col, self.tfs_col, self.tte_col] + self.features]

        # Group data by customer ID in order to build the sequences
        sequences = [list(g) for _, g in groupby(data.values, key=itemgetter(0))]
        lengths = [len(s) for s in sequences]

        if self.max_sl == 0:
            self.max_sl = np.max(lengths)

        n_sl = self.max_sl + 1
        n_feat = len(self.features)
        seq = []

        for s in sequences:
            idx = s[0][0]  # Customer ID
            tfs = s[0][1]  # Periods from start date to first period recorded

            # Get the list of sequences to be used for each customer as an array of offsets
            # Each offset represents the starting point of a new sequence
            # By default, only one sequence is created with the last `max_sl` periods of each customer
            # But if `deep` is True, all possible sequences are created for each customer
            keys = [max([len(s) - n_sl, 0])]
            if deep:
                keys = list(np.arange(keys[0] + 1))

            for k in keys:
                cur_seq = np.empty([n_sl, n_feat + 5])

                # Get feature values for each row in the sequence using the offset `k`
                trunc = np.array(s)[k:(k + n_sl), 3:]  # Exclude 3 key columns in input data
                cur_seq[:len(trunc), 5:] = trunc  # After 5 key columns in output data

                # Compute Weibull `alpha` and `beta` parameters from `tte` column (s[r][2])
                # The logic is described in the method definition
                a = [
                    s[r][2] if r < len(s) and s[r][2] >= 0 else
                    len(s) - r - 1 if r < len(s) else
                    np.NaN
                    for r in np.arange(k, k + n_sl)
                ]
                b = [
                    1 if r < len(s) and s[r][2] >= 0 else
                    0 if r < len(s) else
                    np.NaN
                    for r in np.arange(k, k + n_sl)
                ]

                cur_seq[:, 0] = idx
                cur_seq[:, 1] = k + tfs
                cur_seq[:, 2] = np.arange(n_sl)
                cur_seq[:, 3] = a
                cur_seq[:, 4] = b
                cur_seq[len(trunc):, 3:] = np.NaN

                seq.append(cur_seq)

        seq = np.array(seq)

        # Split features and target sequence arrays
        x = np.array([np.delete(s, [3, 4], 1) for s in seq])
        y = seq[..., :5]

        return x, y

    def rebuild_seq(
        self,
        data: np.ndarray,
        use_sls: bool = False
    ) -> pd.DataFrame:
        """
        Rebuild the sequences DataFrame from the given tensor. It is the reverse process of `build_seq`.
        It deconstructs the tensor back into a DataFrame, where customer sequences are reunited into
        a single sequence per customer (if `deep` parameter was used while building the sequences).

        Parameters
        ----------
        data : np.ndarray
            Input tensor.
        use_sls : bool, optional
            If True, use actual sequence lengths DataFrame to crop each customer sequence.
            Otherwise, all sequences will be fully retreived, including out of sequence (null) values.

        Returns
        -------
        pd.DataFrame
            Sequences DataFrame.
        """
        data = self.seq_to_df(data, use_sls=use_sls)
        data[self.tfs_col] = data[self.seq_col] + data[self.tfs_col]

        data = (
            data
            .sort_values([self.id_col, self.tfs_col, self.seq_col])
            .groupby([self.id_col, self.tfs_col])
            .first()
            .reset_index()
            .drop([self.seq_col], axis=1)
        )

        return data

    def input_seq(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare the input sequences for the RRN model fit.

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

        z = x[..., :3]  # Key columns to be used for indexing results
        x = x[..., 3:]  # Input features

        x = np.nan_to_num(x, nan=self.mask)

        if y is not None:
            y = y[..., 3:]  # Target values

            y[..., 0] = np.nan_to_num(y[..., 0], nan=self.a_mean)
            y[..., 1] = np.nan_to_num(y[..., 1], nan=self.b_mean)

        return x, y, z

    def output_seq(
        self,
        y: np.ndarray,
        z: np.ndarray
    ) -> np.ndarray:
        """
        Prepare the output sequences of the RRN model predictions.

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
        max_sl = y.shape[1]

        z = z.reshape(n_seq * max_sl, 3)
        y = y.reshape(n_seq * max_sl, 2)

        seq = np.hstack([z, y]).reshape(n_seq, max_sl, 5)

        return seq

    def seq_to_df(
        self,
        x: np.ndarray,
        y: Optional[np.ndarray] = None,
        use_sls: bool = False,
        seq_last: bool = False
    ) -> pd.DataFrame:
        """
        Convert the given tensor to a DataFrame.

        Parameters
        ----------
        x : np.ndarray
            Input tensor.
        y : np.ndarray, optional
            Target tensor.
        use_sls : bool, optional
            If True, use actual sequence lengths DataFrame to crop each customer sequence.
            Otherwise, all sequences will be fully retreived, including out of sequence (null) values.
        seq_last : bool, optional
            If True, only the last sequence of each customer will be used.
            Otherwise, all sequences will be used.

        Returns
        -------
        pd.DataFrame
            Output DataFrame.
        """
        x = np.array(x)
        if y is not None:
            y = np.array(y)

        n_seq = x.shape[0]
        max_sl = x.shape[1]
        n_feat = len(self.features)

        if y is not None:
            x = np.array([s[:, -n_feat:] for s in x]).reshape(n_seq * max_sl, n_feat)
            y = y.reshape(n_seq * max_sl, 5)
            df = pd.DataFrame(
                np.hstack([y, x]),
                columns=[self.id_col, self.seq_col, self.tfs_col, self.wa_col, self.wb_col] + self.features
            )
        else:
            x = x.reshape(n_seq * max_sl, 5)
            df = pd.DataFrame(
                x,
                columns=[self.id_col, self.seq_col, self.tfs_col, self.wa_col, self.wb_col]
            )

        # Crop each sequence to its actual sequence length
        if use_sls:
            sls = self.sls.set_index('id')['length'].to_dict()
            df = df.loc[
                df.apply(lambda x: x[self.tfs_col] < sls[x[self.id_col]], axis=1)
            ]

        # Get only the last sequence of each customer
        if seq_last:
            df = df.groupby(self.id_col).last().reset_index()

        return df

    def get_seq_lengths(
        self,
        x: np.ndarray
    ) -> pd.DataFrame:
        """
        Get customer sequence lengths from the given tensor.
        For each customer, find the number of periods whit non-null values.

        Parameters
        ----------
        x : np.ndarray
            Input tensor.

        Returns
        -------
        pd.DataFrame
            Sequence lengths data.
        """
        return pd.DataFrame({
            'id': x[:, 0, 0].astype(int),
            'length': np.sum(np.all(~np.isnan(x), axis=-1), axis=1)
        })
    
    def pad_seq(
        self,
        data: pd.DataFrame,
        factorize: bool = True
    ) -> pd.DataFrame:
        """
        Pad the given sequences to the beggining of each customer's lifetime.
        Optionally, factorize customer IDs to get an increasing length shape (triangular).

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame.
        factorize : bool, default True
            If True, sort sequences by length and factorize customer IDs.

        Returns
        -------
        pd.DataFrame
            Padded DataFrame.
        """
        df = data[[
            self.id_col, self.tfs_col, self.wa_col, self.wb_col
        ]].sort_values([self.id_col, self.tfs_col], ignore_index=True)

        df[self.tfs_col] = df.groupby(self.id_col)[self.tfs_col].apply(lambda x: x - x.min()).explode().values

        if factorize:
            df['slen'] = df[self.id_col].map(df.groupby(self.id_col)[self.tfs_col].max())
            df = df.sort_values(['slen', self.id_col, self.tfs_col], ascending=[False, True, True], ignore_index=True)
            df[self.id_col] = pd.factorize(df[self.id_col])[0]

        df = df.sort_values([self.id_col, self.tfs_col])[[
            self.id_col, self.tfs_col, self.wa_col, self.wb_col
        ]]

        return df

    def weibull_percentile(
        self,
        ab: np.ndarray,
        p: float = 0.5
    ) -> float | np.ndarray:
        """
        Compute the Weibull distribution Quantile Percentile for a single customer.

        Q(p) = \alpha\left(-\log\left(1-p\right)\right)^{\frac{1}{\beta}}

        Parameters
        ----------
        ab : np.ndarray
            An array of shape (N, 2) with the Weibull Alpha and Beta parameters.
            If a 1D array is provided, it will be reshaped to (1, 2).
        p : float, default 0.5
            Percentile.

        Returns
        -------
        float | np.ndarray
            Quantile values.
            If `ab` is 1D, the result will be 1D.
        """
        ab = np.asfarray(ab)
        ndim = ab.ndim
        if ndim < 2:
            ab = ab.reshape(1, -1)

        a = ab[..., 0][:, np.newaxis]  # Alpha
        b = ab[..., 1][:, np.newaxis]  # Beta

        p = np.float64(p)

        percentile = a * np.power(-np.log(1. - p), 1. / b)

        if ndim < 2:
            percentile = percentile.reshape(-1)

        return percentile

    def weibull_pdf(
        self,
        ab: np.ndarray,
        t: np.ndarray | int = 30
    ) -> np.ndarray:
        """
        Compute the continuous Weibull distribution Probability Density Function (PDF) for a single customer.

        PDF(t) = (beta/alpha) * (t/alpha)^(beta-1) * exp(-(t/alpha)^beta)

        Parameters
        ----------
        ab : np.ndarray
            An array of shape (N, 2) with the Weibull Alpha and Beta parameters.
            If a 1D array is provided, it will be reshaped to (1, 2).
        t : np.ndarray | int
            Period steps to compute the PDF values.
            If an integer is provided, a range from 0 to `t` will be used.

        Returns
        -------
        np.ndarray
            PDF values.
            If `ab` is 1D, the result will be 1D.
        """
        ab = np.asfarray(ab)
        ndim = ab.ndim
        if ndim < 2:
            ab = ab.reshape(1, -1)

        a = ab[..., 0][:, np.newaxis]  # Alpha
        b = ab[..., 1][:, np.newaxis]  # Beta

        if is_array(t):
            t = np.asfarray(t)
        else:
            t = np.asfarray(np.arange(t + 1))

        # Weibull PDF
        pdf = (b / a) * np.power(t / a, b - 1) * np.exp(-np.power(t / a, b))

        if ndim < 2:
            pdf = pdf.reshape(-1)

        return pdf

    def weibull_cdf(
        self,
        ab: np.ndarray,
        t: Optional[np.ndarray | int] = 30
    ) -> np.ndarray:
        """
        Compute the continuous Weibull distribution Cumulative Density Function (CDF) for a single customer.

        CDF(t) = 1 - exp(-(t/alpha)^beta)

        Parameters
        ----------
        ab : np.ndarray
            An array of shape (N, 2) with the Weibull Alpha and Beta parameters.
            If a 1D array is provided, it will be reshaped to (1, 2).
        t : np.ndarray | int
            Period steps to compute the PDF values.
            If an integer is provided, a range from 0 to `t` will be used.

        Returns
        -------
        np.ndarray
            CDF values.
            If `ab` is 1D, the result will be 1D.
        """
        ab = np.asfarray(ab)
        ndim = ab.ndim
        if ndim < 2:
            ab = ab.reshape(1, -1)

        a = ab[..., 0][:, np.newaxis]  # Alpha
        b = ab[..., 1][:, np.newaxis]  # Beta

        if is_array(t):
            t = np.asfarray(t)
        else:
            t = np.asfarray(np.arange(t + 1))

        # Weibull CDF
        cdf = 1 - np.exp(-np.power(t / a, b))

        if ndim < 2:
            cdf = cdf.reshape(-1)
        
        return cdf

    def weibull_pmf(
        self,
        ab: np.ndarray,
        t: Optional[np.ndarray | int] = 30
    ) -> np.ndarray:
        """
        Compute the discrete Weibull distribution Probability Mass Function (PMF) for a single customer.

        PMF(t) = Pr[t <= T <= t+1] = CDF(t+1) - CDF(t) = exp(-(t/alpha)^beta) - exp(-((t+1)/alpha)^beta)

        Parameters
        ----------
        ab : np.ndarray
            An array of shape (N, 2) with the Weibull Alpha and Beta parameters.
            If a 1D array is provided, it will be reshaped to (1, 2).
        t : np.ndarray | int
            Period steps to compute the PDF values.
            If an integer is provided, a range from 0 to `t` will be used.

        Returns
        -------
        np.ndarray
            PMF values.
            If `ab` is 1D, the result will be 1D.
        """
        ab = np.asfarray(ab)
        ndim = ab.ndim
        if ndim < 2:
            ab = ab.reshape(1, -1)

        a = ab[..., 0][:, np.newaxis]  # Alpha
        b = ab[..., 1][:, np.newaxis]  # Beta

        if is_array(t):
            t = np.asfarray(t)
        else:
            t = np.asfarray(np.arange(t + 1))

        # Weibull PMF
        pmf = np.exp(-np.power(t / a, b)) - np.exp(-np.power((t + 1) / a, b))

        if ndim < 2:
            pmf = pmf.reshape(-1)
        
        return pmf

    def weibull_cmf(
        self,
        ab: np.ndarray,
        t: Optional[np.ndarray | int] = 30
    ) -> np.ndarray:
        """
        Compute the discrete Weibull distribution Cumulative Mass Function (CMF) for a single customer.

        CMF(t) = Pr[T <= t+1] = CDF(t+1) = 1 - exp(-((t+1)/alpha)^beta)

        Parameters
        ----------
        ab : np.ndarray
            An array of shape (N, 2) with the Weibull Alpha and Beta parameters.
            If a 1D array is provided, it will be reshaped to (1, 2).
        t : np.ndarray | int
            Period steps to compute the PDF values.
            If an integer is provided, a range from 0 to `t` will be used.

        Returns
        -------
        np.ndarray
            CMF values.
            If `ab` is 1D, the result will be 1D.
        """
        ab = np.asfarray(ab)
        ndim = ab.ndim
        if ndim < 2:
            ab = ab.reshape(1, -1)

        a = ab[..., 0][:, np.newaxis]  # Alpha
        b = ab[..., 1][:, np.newaxis]  # Beta

        if is_array(t):
            t = np.asfarray(t)
        else:
            t = np.asfarray(np.arange(t + 1))
        
        # Weibull CMF
        cmf = 1 - np.exp(-np.power((t + 1) / a, b))

        if ndim < 2:
            cmf = cmf.reshape(-1)
        
        return cmf
    
    def weibull_proba(
        self,
        ab: np.ndarray,
        kind: Literal['continuous', 'discrete'] = None,
        t: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute the probability of survival for a single customer after a given number of periods.
        The result is the probability predicted for the customer to survive until the given time.

        It uses the Weibull distribution cumulative mass function (CMF) to compute
        the probability of survival after a given number of periods.

        Parameters
        ----------
        ab : np.ndarray
            An array of shape (N, 2) with the Weibull Alpha and Beta parameters.
            If a 1D array is provided, it will be reshaped to (1, 2).
        kind : Literal['continuous', 'discrete'], optional
            Kind of plot.
            If 'continuous', the CDF will be used.
            If 'discrete', the CMF will be used.
            If None, the `kind` parameter value will be used.
        t : int, optional
            Number of periods from the starting point (0 = same period).
            If None, the `wsize` parameter value will be used.

        Returns
        -------
        np.ndarray
            Probability of survival after the given number of periods.
            If `ab` is 1D, the result will be 1D.
        """
        kind = kind or self.kind

        if t is None:
            t = int(self.min_tte)

        if kind == 'discrete':
            c = self.weibull_cmf(ab, t=t)
        elif kind == 'continuous':
            c = self.weibull_cdf(ab, t=t)

        return c[:, (t - 1)]

    def plot_params_dist(
        self,
        y: Optional[pd.DataFrame] = None,
        loc: Optional[int] = None,
        xmax: Optional[int] = None,
        ymax: Optional[int] = None,
        s: int = 10,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the distribution of Weibull alpha and beta parameters.

        Parameters
        ----------
        y : pd.DataFrame, optional
            Input data.
            If not provided, the `results` dataframe will be used.
        loc : int, optional
            Position in each customer's sequence.
            If -1, the last period will be used, representing the current params for each user.
            If greater than 0, the specified period from the beginning of each sequence will be used.
            If None, all periods will be considered, representing the params for each user at each period.
        xmax : int, optional
            Maximum value in the x-axis.
        ymax : int, optional
            Maximum value in the y-axis.
        s : int, optional
            Size of the scatter plot points.
        ax : plt.Axes, optional
            Axes to plot the figure.
            If not provided, a new figure will be created.
        show : bool, optional
            If True, the figure will be displayed.
        file : str, optional
            If provided, the figure will be saved to a file.
        """
        if y is None:
            y = self.results.copy()

        if loc:
            y = y.sort_values(self.tfs_col).groupby(self.id_col).nth(loc).reset_index()

        sx = y[self.wa_col].values
        sy = y[self.wb_col].values

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax.scatter(sx, sy, s=s, c=get_color('blue'), zorder=3)

        if not xmax:
            xmax = sx.max()
        ax.set_xlim((0, xmax))
        ax.set_xticks(np.arange(0, xmax + 10, 10))
        ax.set_xlabel('alpha')

        if not ymax:
            ymax = sy.max()
        ax.set_ylim((0, ymax))
        ax.set_yticks(np.arange(0, ymax + 0.5, 0.5))
        ax.set_ylabel('beta')

        ax.set_title('Weibull Params')

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_params_seq(
        self,
        y: Optional[pd.DataFrame] = None,
        factorize: bool = True,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot each customer Weibull alpha and beta parameters over time, showing how the parameters
        change from one period to the next as the customer info and usage change.

        In this plot, each line represents a customer and each column represents a period,
        so the plot is a sequence of customers over time.

        - Alpha: the risk of a customer churning, hotter colors represent lower TTEs,
        which means the customer is more likely to churn soon.
        In a well fitted model, colors shoud be hotter near the end of each sequence.

        - Beta: the level of certainty of predictions, greener colors represent higher certainty,
        which means the model is more certain about the customer's predicted TTE.
        In a well fitted model, colors shoud be greener for customers with many periods recorded.

        Parameters
        ----------
        y : pd.DataFrame, optional
            Input data.
            If not provided, the `results` dataframe will be used.
        factorize : bool, optional
            If True, sort sequences by length and factorize customer IDs,
            so that we get an increasing length shape (triangular).
        show : bool, optional
            If True, the figure will be displayed.
        file : str, optional
            If provided, the figure will be saved to a file.
        """
        if y is None:
            y = self.results.copy()

        df = self.pad_seq(y, factorize=factorize)

        x = np.arange(df[self.tfs_col].max() + 1)
        y = np.arange(df[self.id_col].max() + 1)
        wa = pd.pivot_table(df, index=self.id_col, columns=self.tfs_col, values=self.wa_col)
        wb = pd.pivot_table(df, index=self.id_col, columns=self.tfs_col, values=self.wb_col)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8), constrained_layout=True)
        axs = axs.flatten()

        a_norm = mcolors.TwoSlopeNorm(vcenter=self.a_mean, vmin=0, vmax=self.max_sl)
        axs[0].pcolormesh(x, y, wa, norm=a_norm, cmap='hot')
        axs[0].set_title('alpha')

        b_norm = mcolors.TwoSlopeNorm(vcenter=self.b_mean, vmin=0, vmax=self.max_beta)
        axs[1].pcolormesh(x, y, wb, norm=b_norm, cmap='Spectral')
        axs[1].set_title('beta')

        for ax in axs:
            ax.set_xlim((0, np.max(x)))
            ax.set_ylim((0, np.max(y)))

            ax.set_xlabel('time')
            ax.set_ylabel('sequence')

        plt.suptitle('Weibull Sequences', y=1.03)

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_single_params(
        self,
        y: pd.DataFrame,
        id: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the distribution of the Weibull alpha and beta parameters
        for a single customer over time, showing how the parameters change
        from one period to the next as the customer info and usage change.

        The alpha parameter represents the scale of the Weibull distribution,
        which denotes the time it takes for the customer to churn,
        while the beta parameter represents the shape of the Weibull distribution,
        which is a measure of dispersion, meaning how sure we are about the result.

        Parameters
        ----------
        y : pd.DataFrame
            Input data.
        id : int, optional
            Customer ID.
            If not provided, it will assume data is for a single customer.
        """
        if id:
            y = y[y[self.id_col] == id]

        x = np.array(np.arange(len(y[self.wa_col])))

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax2 = ax.twinx()

        ax.plot(x, y[self.wa_col], color=get_color('blue'))
        ax.set_xlabel('time')
        ax.set_ylabel('alpha')

        ax2.plot(x, y[self.wb_col], color=get_color('red'))
        ax2.set_ylabel('beta')

        for t in ax.get_yticklabels():
            t.set_color(get_color('blue'))
        for t in ax2.get_yticklabels():
            t.set_color(get_color('red'))

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_weibull(
        self,
        y: pd.DataFrame,
        kind: Literal['continuous', 'discrete'] = None,
        id: Optional[int] = None,
        loc: int = -1,
        t: int = 30,
        ax: Optional[plt.Axes] = None,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot both the probability and cumulative functions for a single customer at a specific period,
        showing how the survival of the customer is modeled by the Weibull distribution.

        Basically, the probability function shows the probability of the customer
        to churn at a given time (the peak of the distribution is the most probable churn period),
        while the cumulative function shows the probability of the customer to churn before any given time.

        Depending on the kind of plot (discrete or continuous), the PMF or the PDF will be plotted.

        Parameters
        ----------
        y : pd.DataFrame
            Input data.
        kind : Literal['continuous', 'discrete'], optional
            Kind of plot.
            If 'continuous', the PDF will be used.
            If 'discrete', the PMF will be used.
            If None, the `kind` parameter value will be used.
        id : int, optional
            Customer ID.
            If not provided, it will assume data is for a single customer.
        loc : int, optional
            Position in the customer's sequence.
            If -1, the last period will be used, representing the current params for each user.
            If greater than 0, the specified period from the beginning of each sequence will be used.
            If None, all periods will be considered, representing the params for each user at each period.
        t : int, optional
            Number of periods from the starting point (0 = same period).
            By default, 30 periods are used.
        """
        kind = kind or self.kind

        y = y.sort_values(self.tfs_col).groupby(self.id_col).nth(loc).reset_index()
        if id is not None:
            y = y[y[self.id_col] == id].iloc[0]

        ab = y[[self.wa_col, self.wb_col]].values

        if kind == 'discrete':
            p = self.weibull_pmf(ab, t=t)
            c = self.weibull_cmf(ab, t=t)
        else:
            p = self.weibull_pdf(ab, t=t)
            c = self.weibull_cdf(ab, t=t)

        x = np.arange(len(p))

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax2 = ax.twinx()

        ax.plot(x, p, color=get_color('blue'))
        ax.set_xlabel('time')
        ax.set_ylabel('P{}F'.format(kind[0].upper()))

        ax2.plot(x, c, color=get_color('red'))
        ax2.set_ylabel('C{}F'.format(kind[0].upper()))

        for t in ax.get_yticklabels():
            t.set_color(get_color('blue'))
        for t in ax2.get_yticklabels():
            t.set_color(get_color('red'))

        self.plot_figure(fig=fig, show=show, file=file)

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

        columns = ['loss', 'val_loss']
        if 'learning_rate' in self.history.columns:
            columns.append('learning_rate')

        df = self.history[columns]

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax.plot(
            df.index.values, df['loss'].values, color=get_color('blue'), label='loss'
        )
        ax.plot(
            df.index.values, df['val_loss'].values, color=get_color('orange'), label='val_loss'
        )

        if 'learning_rate' in self.history.columns:
            lr = self.params['lr']

            ax2 = ax.twinx()
            ax2.plot(
                df.index.values, df['learning_rate'].values,
                color=get_color('grey'), ls='--', lw=2, alpha=0.2,
                label='learning rate'
            )

            ax2.set_ylim((lr * 0.001, lr * 10))
            ax2.set_yscale('log')
            ax2.yaxis.set_tick_params(which='minor', length=0)
            ax2.legend(loc='upper right')

        ax.legend(loc='upper left')
        ax.set_title('loss')

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_weights(
        self,
        show: bool = True,
        file: Optional[str] = None
    ):
        """
        Plot the bias and weights of the model over time,
        showing how the model's bias and weights change as the model learns.

        Parameters
        ----------
        show : bool, optional
            If True, the figure will be displayed.
        file : str, optional
            If provided, the figure will be saved to a file.
        """
        b = self.watcher.bias
        b_mean = b.mean(axis=1)

        epochs = b.shape[0]
        x = np.arange(epochs)

        w = self.watcher.weights.reshape(epochs, -1, 2)
        w_mean = w.mean(axis=1)
        w_min = w.min(axis=1)
        w_max = w.max(axis=1)

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5), constrained_layout=True)
        axs = axs.flatten()

        bax = axs[0]
        bax2 = bax.twinx()

        bax.plot(x, b_mean[:, 0], color=get_color('blue'))
        bax.set_ylabel('alpha')

        bax2.plot(x, b_mean[:, 1], color=get_color('red'))
        bax2.set_ylabel('beta')

        for t in bax.get_yticklabels():
            t.set_color(get_color('blue'))
        for t in bax2.get_yticklabels():
            t.set_color(get_color('red'))

        bax.set_title('bias')

        wax = axs[1]
        wax2 = wax.twinx()

        wax.plot(x, w_mean[:, 0], color=get_color('blue'))
        wax.plot(x, w_min[:, 0], color=get_color('blue'), linestyle='--')
        wax.plot(x, w_max[:, 0], color=get_color('blue'), linestyle='--')
        wax.set_ylabel('alpha')

        wax2.plot(x, w_mean[:, 1], color=get_color('red'))
        wax2.plot(x, w_min[:, 1], color=get_color('red'), linestyle='--')
        wax2.plot(x, w_max[:, 1], color=get_color('red'), linestyle='--')
        wax2.set_ylabel('beta')

        for t in wax.get_yticklabels():
            t.set_color(get_color('blue'))
        for t in wax2.get_yticklabels():
            t.set_color(get_color('red'))

        wax.set_title('weights')

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

    def print_model_summary(self, **kwargs):
        """
        Print the model summary.
        """
        model = self.model or self.build_model()
        model.summary(**kwargs)

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
        return self.file_exists('wtte.weights.h5')

    def save_model(self) -> str:
        """
        Save the model to a file.
        """
        path = self.get_path('wtte.weights.h5')
        self.model.save_weights(path)

        return path

    def load_model(self) -> Self:
        """
        Load the model from a file.
        """
        path = self.get_path('wtte.weights.h5')
        self.model = self.build_model()
        self.model.load_weights(path)

        return self

    def watcher_exists(self) -> bool:
        """
        Check if the watcher exists.
        """
        return self.file_exists('watcher-weights.npy')
    
    def save_watcher(self) -> Self:
        np.save(self.get_path('watcher-weights.npy'), self.watcher.weights)
        np.save(self.get_path('watcher-bias.npy'), self.watcher.bias)

        return self

    def load_watcher(self) -> Self:
        weights = np.load(self.get_path('watcher-weights.npy'))
        bias = np.load(self.get_path('watcher-bias.npy'))

        self.watcher = WeightWatcher(level=self.wlevel).set_weights(weights, bias)

        return self
    
    def get_config(self) -> dict:
        """
        Get the model configuration.
        """
        return {
            k: v for k, v in self.__dict__.items()
            if k not in [
                'model', 'params', 'history',
                'sls', 'results', 'watcher',
                'encoder', 'scaler', 'imputer',
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
            self.save_model()
        
        if self.watcher is not None:
            self.save_watcher()

        if self.history is not None:
            self.history.to_json(self.get_path('history.json'), double_precision=10)

        if not self.sls.empty:
            self.sls.to_json(self.get_path('sls.json'))

        if not self.results.empty:
            self.results.to_json(self.get_path('results.json'))

        if self.encoder is not None:
            joblib.dump(self.encoder, self.get_path('encoder.pkl'))

        if self.scaler is not None:
            joblib.dump(self.scaler, self.get_path('scaler.pkl'))

        if self.imputer is not None:
            joblib.dump(self.imputer, self.get_path('imputer.pkl'))

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
            self.load_model()
        
        if self.watcher_exists():
            self.load_watcher()

        if self.file_exists('history.json'):
            self.history = pd.read_json(self.get_path('history.json'))

        if self.file_exists('sls.json'):
            self.sls = pd.read_json(self.get_path('sls.json'))

        if self.file_exists('results.json'):
            self.results = pd.read_json(self.get_path('results.json'))

        if self.file_exists('encoder.pkl'):
            self.encoder = joblib.load(self.get_path('encoder.pkl'))

        if self.file_exists('scaler.pkl'):
            self.scaler = joblib.load(self.get_path('scaler.pkl'))

        return self


class WeightWatcher(keras.callbacks.Callback):
    """
    Keras Callback to keep an eye on output layer weights.

    Parameters
    ----------
    level : Literal['epoch', 'batch'], optional
        Level to save the weights and bias.
        If 'epoch', the weights and bias are saved after each epoch.
        If 'batch', the weights and bias are saved after each batch.
    """

    def __init__(
        self,
        level: Literal['epoch', 'batch'] = 'epoch'
    ):
        self.level = level

        self.epoch = 0
        self.batch = 0

        self.weights = None
        self.bias = None
    
    def on_train_begin(self, logs=None):
        epochs = self.params['epochs']
        steps = self.params['steps'] if self.level == 'batch' else 1
        seqlen = self.model.layers[-3].output.shape[-1]
        output = self.model.layers[-1].output.shape[-1]
        
        self.weights = np.full((epochs, steps, seqlen, output), np.NaN)
        self.bias = np.full((epochs, steps, output), np.NaN)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
    
    def on_batch_begin(self, batch, logs=None):
        if self.level == 'batch':
            self.batch = batch

    def on_batch_end(self, batch, logs=None):
        if self.level == 'batch':
            self.add_weights()

    def on_epoch_end(self, epoch, logs=None):
        if self.level == 'epoch':
            self.add_weights()

    def on_train_end(self, logs=None):
        self.weights = self.weights[:self.epoch + 1]
        self.bias = self.bias[:self.epoch + 1]
    
    def add_weights(self):
        weights, bias = self.model.get_weights()[-2:]

        self.weights[self.epoch, self.batch] = weights
        self.bias[self.epoch, self.batch] = bias
    
    def set_weights(self, weights: np.ndarray, bias: np.ndarray) -> Self:
        self.weights = weights
        self.bias = bias

        # Infer watcher params from saved arrays
        self.level = 'batch' if self.weights.shape[1] > 1 else 'epoch'
        self.epoch = self.weights.shape[0] - 1
        self.batch = self.weights.shape[1] - 1

        return self
