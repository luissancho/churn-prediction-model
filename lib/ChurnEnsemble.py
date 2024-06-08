import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from typing import Optional
from typing_extensions import Self

from .ChurnWTTE import ChurnWTTE
from .ChurnXGB import ChurnXGB


class ChurnEnsemble(object):

    def __init__(
        self,
        seed: int = 42,
        verbose: int = 0,
        path: Optional[str] = None
    ):
        # Column names
        self.id_col = 'id'  # Subscription numeric ID
        self.tp_col = 'tp'  # Period (month)
        self.ts_col = 'ts'  # Start date
        self.te_col = 'te'  # End date
        self.tfs_col = 'tfs'  # Periods from start date
        self.tte_col = 'tte'  # Periods until end date
        self.tgt_col = 'tgt'  # Target value
        self.sid_col = 'sid'  # Customer hash ID

        # Weibull parameters to be predicted
        self.y_col = 'y'  # Alpha
        self.u_col = 'u'  # Beta

        # Execution params
        self.seed = seed  # Base random seed
        self.verbose = verbose  # Print progress
        self.path = path or os.getcwd()  # Path to store model files

        self._data = pd.DataFrame.from_records({})

        # Split dates
        self.dt_start = None  # Date to start the train data
        self.dt_split = None  # Date to split train and test data
        self.dt_end = None  # Date to end the test data

        self.wtte = None
        self.xgb = None
    
    @property
    def data(self):
        return self._data.copy()

    @data.setter
    def data(self, data):
        self._data = data
    
    @property
    def rows(self):
        return self._data.shape[0]

    def set_params(
        self, 
        wtte_params: Optional[dict] = None, 
        xgb_params: Optional[dict] = None,
        dt_end: Optional[pd.Timestamp | str | int] = None,
        dt_split: Optional[pd.Timestamp | str | int] = None,
        dt_start: Optional[pd.Timestamp | str | int] = None
    ) -> Self:
        if xgb_params is not None:
            self.xgb_params = xgb_params
        if wtte_params is not None:
            self.wtte_params = wtte_params
        
        if dt_end is not None:
            if isinstance(dt_end, (int, float)):
                self.dt_end = self.data[self.tp_col].max() - pd.DateOffset(months=int(dt_end))
            else:
                self.dt_end = pd.Timestamp(dt_end)
        
        if dt_split is not None:
            if isinstance(dt_split, (int, float)):
                self.dt_split = self.dt_end - pd.DateOffset(months=int(dt_split))
            else:
                self.dt_split = pd.Timestamp(dt_split)
        
        if dt_start is not None:
            if isinstance(dt_start, (int, float)):
                if dt_start > 0:
                    self.dt_start = self.dt_split - pd.DateOffset(months=int(dt_start))
                else:
                    self.dt_start = self.data[self.tp_col].min()
            else:
                self.dt_start = pd.Timestamp(dt_start)

        return self
    
    def fit_wtte(
        self
    ) -> Self:
        df = self.data

        self.wtte = ChurnWTTE(
            features=self.wtte_params['features'],
            max_sl=self.wtte_params['max_sl'],
            seed=self.seed,
            verbose=self.verbose,
            path=f'{self.path}/wtte',
            **self.wtte_params['params']
        )

        d_wtte_train = df[
            df[self.tp_col].between(self.dt_start, self.dt_split)
        ][[self.id_col, self.tfs_col, self.tte_col] + self.wtte.features]
        self.wtte.scaler = StandardScaler()
        d_wtte_train[self.wtte.features] = self.wtte.scaler.fit_transform(d_wtte_train[self.wtte.features])
        x_wtte_train, y_wtte_train = self.wtte.build_seq(d_wtte_train, deep=False)

        d_wtte_test = df[
            (df[self.tp_col].between(self.dt_start, self.dt_end)) & ((df[self.te_col].isnull()) | (df[self.te_col] > self.dt_split)
            )
        ][[self.id_col, self.tfs_col, self.tte_col] + self.wtte.features]
        d_wtte_test[self.wtte.features] = self.wtte.scaler.transform(d_wtte_test[self.wtte.features])
        x_wtte_test, y_wtte_test = self.wtte.build_seq(d_wtte_test, deep=False)

        self.wtte.fit(x_wtte_train, y_wtte_train, x_wtte_test, y_wtte_test)
        self.wtte.plot_history_eval(show=False, file='wtte-history.png')

        self.wtte.sls = self.wtte.get_seq_lengths(y_wtte_test)
        y_wtte_hat = self.wtte.predict(x_wtte_test)
        self.wtte.set_results(y_wtte_hat)

        self.wtte.plot_params_dist(self.wtte.results, loc=-1, show=False, file='wtte-params.png')

        return self
    
    def fit_xgb(
        self
    ) -> Self:
        df = self.data

        self.xgb = ChurnXGB(
            features=self.xgb_params['features'] + [self.y_col, self.u_col],
            seed=self.seed,
            verbose=self.verbose,
            path=f'{self.path}/xgb',
            **self.xgb_params['params']
        )

        d_wtte_pred = df[
            df[self.tp_col].between(self.dt_start, self.dt_end)
        ][[self.id_col, self.tfs_col, self.tte_col] + self.wtte.features]
        d_wtte_pred[self.wtte.features] = self.wtte.scaler.transform(d_wtte_pred[self.wtte.features])
        x_wtte_pred, y_wtte_pred = self.wtte.build_seq(d_wtte_pred, deep=True)

        self.wtte.sls = self.wtte.get_seq_lengths(y_wtte_pred)
        y_wtte_hat = self.wtte.predict(x_wtte_pred)
        self.wtte.set_results(y_wtte_hat)

        d_xgb_train = df[
            df[self.tp_col].between(self.dt_start, self.dt_split)
        ].merge(
            self.wtte.results[[self.id_col, self.tfs_col, self.y_col, self.u_col]],
            on=[self.id_col, self.tfs_col], how='left'
        )[[self.id_col, self.tfs_col, self.tgt_col] + self.xgb.features]
        x_xgb_train, y_xgb_train = self.xgb.build_seq(d_xgb_train)

        d_xgb_test = df[
            df[self.tp_col].between(self.dt_split, self.dt_end)
        ].merge(
            self.wtte.results[[self.id_col, self.tfs_col, self.y_col, self.u_col]],
            on=[self.id_col, self.tfs_col], how='left'
        )[[self.id_col, self.tfs_col, self.tgt_col] + self.xgb.features]
        x_xgb_test, y_xgb_test = self.xgb.build_seq(d_xgb_test)

        self.xgb.fit(x_xgb_train, y_xgb_train, x_xgb_test, y_xgb_test)
        self.xgb.plot_history_eval(show=False, file='xgb-history.png')

        y_xgb_hat = self.xgb.predict(x_xgb_test)
        self.xgb.set_results(y_xgb_hat, y_xgb_test)

        self.xgb.plot_scores(show=False, file='xgb-scores.png')
        self.xgb.plot_histogram(self.xgb.results, loc=-1, show=False, file='xgb-histogram.png')

        return self

    def fit(
        self
    ) -> Self:
        self.fit_wtte()
        self.fit_xgb()

        return self

    def predict(
        self
    ) -> Self:
        df = self.data

        d_wtte_pred = df[[self.id_col, self.tfs_col, self.tte_col] + self.wtte.features]
        d_wtte_pred[self.wtte.features] = self.wtte.scaler.transform(d_wtte_pred[self.wtte.features])
        x_wtte_pred, y_wtte_pred = self.wtte.build_seq(d_wtte_pred, deep=True)

        self.wtte.sls = self.wtte.get_seq_lengths(y_wtte_pred)
        y_wtte_hat = self.wtte.predict(x_wtte_pred)
        self.wtte.set_results(y_wtte_hat)

        d_xgb_pred = df.merge(
            self.wtte.results[[self.id_col, self.tfs_col, self.y_col, self.u_col]],
            on=[self.id_col, self.tfs_col], how='left'
        )[[self.id_col, self.tfs_col, self.tgt_col] + self.xgb.features]
        x_xgb_pred, _ = self.xgb.build_seq(d_xgb_pred)

        y_xgb_hat = self.xgb.predict(x_xgb_pred)
        self.xgb.set_results(y_xgb_hat)

        self.xgb.plot_histogram(self.xgb.results, loc=-1, show=False, file='xgb-histogram-pred.png')

        return self

    def results(
        self
    ) -> Self:
        df = self.xgb.results.copy()
        df[self.tgt_col] = (df['pred'] > self.xgb.thr).astype(int)

        c_0 = self.xgb.get_clusters(df[df[self.tgt_col] == 0]['pred'], 3)
        c_1 = self.xgb.get_clusters(df[df[self.tgt_col] == 1]['pred'], 2) + 3
        df['seg'] = pd.concat([c_0, c_1], axis=0)

        df = df.merge(
            self.wtte.results[[self.id_col, self.tfs_col, self.y_col, self.u_col]],
            on=[self.id_col, self.tfs_col], how='left'
        ).rename(columns={
            self.y_col: 'wa',
            self.u_col: 'wb'
        }).merge(
            self.data[[self.id_col, self.tfs_col, 'sid', 'momentum']],
            on=[self.id_col, self.tfs_col], how='left'
        ).rename(columns={
            'momentum': 'mom'
        }).sort_values([self.id_col, self.tfs_col]).groupby(self.id_col).last().reset_index()
    
        df[self.id_col] = df['sid']
        df = df[[
            self.id_col, 'pred', self.tgt_col, 'seg', 'wa', 'wb', 'mom'
        ]]

        return df
    
    def save(
        self
    ) -> Self:
        self.wtte.save()
        self.xgb.save()

        return self
    
    def load(
        self
    ) -> Self:
        self.wtte = ChurnWTTE(
            seed=self.seed,
            verbose=self.verbose,
            path=f'{self.path}/wtte'
        ).load()

        self.xgb = ChurnXGB(
            seed=self.seed,
            verbose=self.verbose,
            path=f'{self.path}/xgb'
        ).load()

        return self
