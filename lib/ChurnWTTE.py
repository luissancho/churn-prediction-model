import joblib
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from itertools import groupby
from operator import itemgetter
from typing import Optional

from .utils import get_color


class ChurnWTTE(object):

    def __init__(
        self,
        features: Optional[list[str]] = None,
        max_sl: int = 0,
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
        self.y_col = 'y'  # Alpha
        self.u_col = 'u'  # Beta

        # Execution params
        self.seed = seed  # Base random seed
        self.verbose = verbose  # Print progress
        self.path = path or os.getcwd()  # Path to store model files

        # Model params
        self.params = {
            'nn': kwargs.get('nn', 0),
            'hl': kwargs.get('hl', 2),
            'lr': kwargs.get('lr', 1e-3),
            'epochs': kwargs.get('epochs', 100),
            'batch': kwargs.get('batch', 32),
            'lr_decay': kwargs.get('lr_decay', 10),
            'stop': kwargs.get('stop', 30),
            'kind': kwargs.get('kind', 'discrete'),
            'dropout': kwargs.get('dropout', 0.2),
            'weight_l1': kwargs.get('weight_l1', 0),
            'weight_l2': kwargs.get('weight_l2', 1e-5),
            'init_alpha': kwargs.get('init_alpha', None),
            'reg_alpha': kwargs.get('reg_alpha', True),
            'max_beta': kwargs.get('max_beta', 2.0),
            'clip_prob': kwargs.get('clip_prob', 1e-6),
            'epsilon': kwargs.get('epsilon', 1e-8),
            'shuffle': kwargs.get('shuffle', False)
        }

        # Model instance
        self.model = None

        # Norm instances
        self.encoder = None
        self.scaler = None
        self.imputer = None

        # Rules
        self.features = features or []  # Features
        self.max_sl = max_sl  # Max number of sequence lengths

        # Model info
        self.mask = np.NaN
        self.y_mean = np.NaN
        self.u_mean = np.NaN

        # Results
        self.history = pd.DataFrame()
        self.sls = pd.DataFrame()
        self.results = pd.DataFrame()

    def build_model(self):
        if np.isnan(self.mask):
            self.mask = -1.0
        if self.params['nn'] == 0:
            self.params['nn'] = self.max_sl if self.max_sl > 0 else 1

        regularizer = None
        if self.params['weight_l1'] > 0 and self.params['weight_l2'] > 0:
            regularizer = tf.keras.regularizers.l1_l2(l1=self.params['weight_l1'], l2=self.params['weight_l2'])
        elif self.params['weight_l1'] > 0:
            regularizer = tf.keras.regularizers.l1(self.params['weight_l1'])
        elif self.params['weight_l2'] > 0:
            regularizer = tf.keras.regularizers.l2(self.params['weight_l2'])

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Masking(mask_value=self.mask, input_shape=(None, len(self.features))))

        for _ in np.arange(self.params['hl']):
            model.add(tf.keras.layers.LSTM(
                self.params['nn'],
                activation='tanh',
                dropout=self.params['dropout'],
                recurrent_dropout=self.params['dropout'],
                kernel_regularizer=regularizer,
                recurrent_regularizer=regularizer,
                return_sequences=True
            ))
            model.add(tf.keras.layers.BatchNormalization(
                gamma_regularizer=regularizer,
                beta_regularizer=regularizer
            ))

        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2)))
        model.add(tf.keras.layers.Activation(self.activation))

        model.compile(
            loss=self.loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.params['lr']),
            sample_weight_mode='temporal'
        )

        return model

    def activation(self, ab):
        # Unstack tensor
        a = ab[..., 0]
        b = ab[..., 1]

        init_alpha = self.params['init_alpha']
        max_beta = self.params['max_beta']

        # Initialize alpha
        a = init_alpha * tf.keras.backend.exp(a)

        # Fix beta to begin moving slowly around 1.0
        if max_beta > 1.05:
            b = b - np.log(max_beta - 1.0)
        b = max_beta * tf.keras.backend.sigmoid(b)

        x = tf.keras.backend.stack([a, b], axis=-1)

        return x

    def loss(self, y_true, y_pred):
        # Unstack tensors
        y = y_true[..., 0]
        u = y_true[..., 1]
        a = y_pred[..., 0]
        b = y_pred[..., 1]

        kind = self.params['kind']
        clip_prob = self.params['clip_prob']

        if kind == 'discrete':
            loglik = self.loglik_discrete(y, u, a, b)
        else:
            loglik = self.loglik_continuous(y, u, a, b)

        if clip_prob is not None:
            loglik = tf.keras.backend.clip(loglik, math.log(clip_prob), math.log(1 - clip_prob))

        loss = -1.0 * loglik

        return loss

    def fit(self, x_train, y_train, x_test, y_test):
        tf.keras.backend.clear_session()
        tf.keras.backend.set_epsilon(self.params['epsilon'])
        tf.random.set_seed(self.seed)

        nant = tf.keras.callbacks.TerminateOnNaN()
        callbacks = [nant]

        if self.params['lr_decay'] > 0:
            red_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                patience=self.params['lr_decay'],
                factor=0.1,
                min_lr=self.params['lr'] * 0.01,
                verbose=self.verbose
            )
            callbacks.append(red_lr)

        if self.params['stop'] > 0:
            estop = tf.keras.callbacks.EarlyStopping(patience=self.params['stop'])
            callbacks.append(estop)

        # Build sequences
        x_train, y_train, _ = self.input_seq(x_train, y_train)
        x_test, y_test, _ = self.input_seq(x_test, y_test)

        # Load model
        self.model = self.build_model()
        if self.verbose > 0:
            self.print_model_summary()

        # Fit
        h = self.model.fit(
            x_train, y_train,
            epochs=self.params['epochs'],
            batch_size=self.params['batch'],
            validation_data=(x_test, y_test),
            shuffle=self.params['shuffle'],
            verbose=self.verbose,
            callbacks=callbacks
        )

        self.history = pd.DataFrame.from_dict(h.history, dtype='float')

        return self

    def predict(self, x):
        # Load model
        if self.model is None:
            self.load()

        # Build sequences
        x, _, z = self.input_seq(x)

        # Predict
        y_pred = self.model.predict(x)

        # Build output
        y_pred = self.output_seq(y_pred, z)

        return y_pred

    def set_results(self, y_pred, y_true=None):
        if self.sls.empty and y_true is not None:
            self.sls = self.get_seq_lengths(y_true)

        y_pred = self.rebuild_seq(y_pred, use_sls=True)
        if y_true is not None:
            y_true = self.rebuild_seq(y_true, use_sls=True)

        self.results = y_pred.sort_values([self.id_col, self.tfs_col])
        self.results[[self.id_col, self.tfs_col]] = self.results[[self.id_col, self.tfs_col]].astype(int)

        return self

    def build_seq(self, data, deep=False):
        if len(self.features) == 0:
            self.features = list(data.columns[~data.columns.isin([self.id_col, self.tfs_col, self.tte_col])])

        if self.tte_col not in list(data.columns):
            data[self.tte_col] = 0
        
        data = data[[self.id_col, self.tfs_col, self.tte_col] + self.features]

        sequences = [list(g) for _, g in groupby(data.values, key=itemgetter(0))]
        lengths = [len(s) for s in sequences]

        if self.max_sl == 0:
            self.max_sl = np.max(lengths)

        n_sl = self.max_sl + 1
        n_feat = len(self.features)
        seq = []

        for s in sequences:
            idx = s[0][0]

            keys = [max([len(s) - n_sl, 0])]
            if deep:
                keys = list(np.arange(keys[0] + 1))

            for i in keys:
                cur_seq = np.empty([n_sl, n_feat + 5])

                trunc = np.array(s)[i:(n_sl + i), 3:]
                cur_seq[:len(trunc), 5:] = trunc

                cur_seq[:, 0] = idx
                cur_seq[:, 1] = i
                cur_seq[:, 2] = np.arange(n_sl)
                cur_seq[:, 3] = [s[r][2] if r < len(s) and s[r][2] >= 0 else len(s) - r - 1 if r < len(s) else np.NaN for r in np.arange(i, n_sl + i)]
                cur_seq[:, 4] = [1 if r < len(s) and s[r][2] >= 0 else 0 if r < len(s) else np.NaN for r in np.arange(i, n_sl + i)]
                cur_seq[len(trunc):, 3:] = np.NaN

                seq.append(cur_seq)

        seq = np.array(seq)

        x = np.array([np.delete(s, [3, 4], 1) for s in seq])
        y = seq[..., :5]

        return x, y

    def rebuild_seq(self, data, use_sls=False):
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

    def input_seq(self, x, y=None):
        z = x[..., :3]

        x = x[..., 3:]

        if np.isnan(self.mask):
            self.mask = np.nanmin(x) - 1

        x = np.nan_to_num(x, nan=self.mask)

        if y is not None:
            y = y[..., 3:]

            if np.isnan(self.y_mean):
                self.y_mean = np.nanmean(y[:, :, 0])
            if np.isnan(self.u_mean):
                self.u_mean = np.nanmean(y[:, :, 1])

            y[..., 0] = np.nan_to_num(y[..., 0], nan=self.y_mean)
            y[..., 1] = np.nan_to_num(y[..., 1], nan=self.u_mean)

            if self.params['init_alpha'] is None:
                self.params['init_alpha'] = -1.0 / np.log(1.0 - 1.0 / (self.y_mean + 1.0))
                if self.params['reg_alpha']:
                    self.params['init_alpha'] = self.params['init_alpha'] / self.u_mean

        return x, y, z

    def output_seq(self, y, z):
        n_seq = y.shape[0]
        max_sl = y.shape[1]

        z = z.reshape(n_seq * max_sl, 3)
        y = y.reshape(n_seq * max_sl, 2)

        seq = np.hstack([z, y]).reshape(n_seq, max_sl, 5)

        return seq

    def seq_to_df(self, x, y=None, use_sls=False, seq_last=False):
        n_seq = x.shape[0]
        max_sl = x.shape[1]
        n_feat = len(self.features)

        if y is not None:
            x = np.array([s[:, -n_feat:] for s in x]).reshape(n_seq * max_sl, n_feat)
            y = y.reshape(n_seq * max_sl, 5)
            df = pd.DataFrame(np.hstack([y, x]), columns=[self.id_col, self.seq_col, self.tfs_col, self.y_col, self.u_col] + self.features)
        else:
            x = x.reshape(n_seq * max_sl, 5)
            df = pd.DataFrame(x, columns=[self.id_col, self.seq_col, self.tfs_col, self.y_col, self.u_col])

        if use_sls:
            sls = self.sls.set_index('id')['length'].to_dict()
            df = df.loc[
                df.apply(lambda x: x[self.tfs_col] < sls[x[self.id_col]], axis=1)
            ]

        if seq_last:
            df = df.groupby(self.id_col).last().reset_index()

        return df

    def get_seq_lengths(self, x):
        return pd.DataFrame({
            'id': x[:, 0, 0].astype(int),
            'length': np.sum(np.all(~np.isnan(x), axis=-1), axis=1)
        })

    def loglik_discrete(self, y, u, a, b):
        epsilon = tf.keras.backend.epsilon()

        hazard0 = tf.keras.backend.pow((y + epsilon) / a, b)
        hazard1 = tf.keras.backend.pow((y + 1.0) / a, b)

        loglik = u * tf.keras.backend.log(tf.keras.backend.exp(hazard1 - hazard0) - (1.0 - epsilon)) - hazard1

        return loglik

    def loglik_continuous(self, y, u, a, b):
        epsilon = tf.keras.backend.epsilon()

        ya = (y + epsilon) / a

        loglik = u * (tf.keras.backend.log(b) + b * tf.keras.backend.log(ya)) - tf.keras.backend.pow(ya, b)

        return loglik

    def weibull_percentile(self, y, id=None, p=0.5):
        if id:
            y = y[y[self.id_col] == id]

        a = y[self.y_col]
        b = y[self.u_col]

        return a * np.power(-np.log(1.0 - p), 1.0 / b)

    def weibull_pdf(self, y, id=None, t=None):
        if id:
            y = y[y[self.id_col] == id]
        
        if t is None:
            t = np.arange(30)
        elif not isinstance(t, (np.ndarray, list, tuple)):
            t = np.arange(t)
        t = np.asfarray(t)
        
        a = y[self.y_col]
        b = y[self.u_col]

        return (b / a) * np.power(t / a, b - 1) * np.exp(-np.power(t / a, b))

    def weibull_cdf(self, y, id=None, t=None):
        if id:
            y = y[y[self.id_col] == id]
        
        if t is None:
            t = np.arange(30)
        elif not isinstance(t, (np.ndarray, list, tuple)):
            t = np.arange(t)
        t = np.asfarray(t)

        a = y[self.y_col]
        b = y[self.u_col]

        return 1 - np.exp(-np.power(t / a, b))

    def weibull_pmf(self, y, id=None, t=None):
        if t is None:
            t = np.arange(30)
        elif not isinstance(t, (np.ndarray, list, tuple)):
            t = np.arange(t)
        t = np.asfarray(t)

        return self.weibull_cdf(y, id=id, t=(t + 1)) - self.weibull_cdf(y, id=id, t=t)

    def weibull_cmf(self, y, id=None, t=None):
        if t is None:
            t = np.arange(30)
        elif not isinstance(t, (np.ndarray, list, tuple)):
            t = np.arange(t)
        t = np.asfarray(t)

        return self.weibull_cdf(y, id=id, t=(t + 1))

    def weibull_cmf_bulk(self, y, t=None):
        if t is None:
            t = np.arange(30)
        elif not isinstance(t, (np.ndarray, list, tuple)):
            t = np.arange(t)
        t = np.asfarray(t)
        
        y = y.sort_values(self.tfs_col).groupby(self.id_col).last().reset_index()

        n_seq = y.shape[0]
        cmfs = np.empty([n_seq, t.shape[0]], dtype=float)

        for i in np.arange(n_seq):
            cmfs[i] = self.weibull_cmf(y.loc[i], t=t)

        cmfs = pd.DataFrame(cmfs, index=y[self.id_col], columns=list(t))

        return cmfs

    def prob_surv(self, y, id=None, loc=-1, t=1):
        if id:
            y = y[y[self.id_col] == id].iloc[loc]

        cmf = self.weibull_cmf(y, t=t)

        return cmf[t - 1]

    def bulk_prob_surv(self, y, t=1):
        cmfs = self.weibull_cmf_bulk(y, t=t)

        y = cmfs.loc[:, t - 1]
        y.name = 'prob'

        return y

    def bulk_true_surv(self, y):
        y = y.sort_values(self.tfs_col).groupby(self.id_col).last().reset_index()
        y[self.tte_col] = [1 if i in [0., 1.] else 0 for i in y[self.tte_col]]
        y = y.set_index(self.id_col).rename_axis(None)[self.tte_col]
        y.name = 'true'

        return y

    def plot_params_dist(self, y, loc=None, xmax=None, s=20, ax=None, show=True, file=None):
        if loc:
            y = y.sort_values(self.tfs_col).groupby(self.id_col).nth(loc).reset_index()

        sx = y[self.y_col].rename('Alpha')
        sy = y[self.u_col].rename('Beta')

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax.scatter(sx, sy, s=s, c=get_color('blue'), zorder=3)

        if xmax:
            ax.set_xlim((0, xmax))
        
        ax.set_title('Params Distribution')

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_single_params(self, y, id=None, ax=None, show=True, file=None):
        if id:
            y = y[y[self.id_col] == id]

        x = np.array(range(len(y[self.y_col])))

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax2 = ax.twinx()

        ax.plot(x, y[self.y_col], color=get_color('blue'))
        ax.set_xlabel('Time')
        ax.set_ylabel('Alpha')

        ax2.plot(x, y[self.u_col], color=get_color('red'))
        ax2.set_ylabel('Beta')

        for t in ax.get_yticklabels():
            t.set_color(get_color('blue'))
        for t in ax2.get_yticklabels():
            t.set_color(get_color('red'))

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_weibull(self, y, kind=None, id=None, loc=-1, t=30, ax=None, show=True, file=None):
        kind = kind or self.params['kind']

        y = y.sort_values(self.tfs_col).groupby(self.id_col).nth(loc).reset_index()
        if id is not None:
            y = y[y[self.id_col] == id].iloc[0]

        if kind == 'discrete':
            p = self.weibull_pmf(y, t=t)
            c = self.weibull_cmf(y, t=t)
        else:
            p = self.weibull_pdf(y, t=t)
            c = self.weibull_cdf(y, t=t)

        x = np.array(range(len(p)))

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax2 = ax.twinx()

        ax.plot(x, p, color=get_color('blue'))
        ax.set_xlabel('Time')
        ax.set_ylabel('P{}F'.format(kind[0].upper()))

        ax2.plot(x, c, color=get_color('red'))
        ax2.set_ylabel('C{}F'.format(kind[0].upper()))

        for t in ax.get_yticklabels():
            t.set_color(get_color('blue'))
        for t in ax2.get_yticklabels():
            t.set_color(get_color('red'))

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_history_eval(self, ax=None, show=True, file=None):
        if self.history.empty:
            return
        
        columns = ['loss', 'val_loss']
        if 'lr' in self.history.columns:
            columns.append('lr')

        df = self.history[columns]

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax.plot(
            df.index, df['loss'].values, color=get_color('blue'), label='training loss'
        )
        ax.plot(
            df.index, df['val_loss'].values, color=get_color('orange'), label='validation loss'
        )

        if 'lr' in self.history.columns:
            model_lr = self.params['lr']
            ax2 = ax.twinx()
            ax2.plot(
                df.index, df['lr'].values, color=get_color('grey'), ls='--', lw=2, alpha=0.2, label=f'learning rate ({model_lr:.0e})'
            )
            ax2.set_ylim((model_lr * 0.001, model_lr * 10))
            ax2.set_yscale('log')
            ax2.yaxis.set_tick_params(which='minor', length=0)

        ax.set_xticks(np.arange(0, len(df) + 5, 5))

        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')

        ax.set_title('epochs')

        self.plot_figure(fig=fig, show=show, file=file)

    def print_model_summary(self):
        if self.model is not None:
            self.model.summary()
        else:
            self.build_model().summary()
    
    def plot_figure(self, fig=None, show=True, file=None):
        if fig is not None:
            if show:
                plt.show()

            if file:
                fig.savefig(self.get_path(file))

            fig.clear()

            plt.clf()
            plt.close('all')

    def get_path(self, name: str) -> str:
        if not os.path.exists('{}/{}'.format(os.getcwd(), self.path)):
            os.makedirs('{}/{}'.format(os.getcwd(), self.path))

        return '{}/{}/{}'.format(os.getcwd(), self.path, name)

    def file_exists(self, name: str) -> bool:
        return os.path.isfile(self.get_path(name))

    def model_exists(self) -> bool:
        return self.file_exists('weights.h5')

    def model_save(self, model):
        path = self.get_path('weights.h5')
        model.save_weights(path)

        return path

    def model_load(self):
        path = self.get_path('weights.h5')
        model = self.build_model()
        model.load_weights(path)

        return model

    def save(self):
        params = json.dumps({
            k: v for k, v in self.__dict__.items()
            if k not in [
                'model', 'history', 'sls', 'results', 'encoder', 'scaler', 'imputer', 'seed', 'verbose', 'path'
            ]
        })
        with open(self.get_path('params.json'), 'w') as fh:
            fh.write(params)

        if self.model is not None:
            self.model_save(self.model)

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

    def load(self):
        if self.file_exists('params.json'):
            with open(self.get_path('params.json'), 'r') as fh:
                params = json.load(fh)
            for k, v in params.items():
                setattr(self, k, v)

        if self.model_exists():
            self.model = self.model_load()

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
