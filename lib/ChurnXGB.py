import joblib
import json
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from xgboost.sklearn import XGBClassifier

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

from .utils import format_number, get_color


class ChurnXGB(object):

    def __init__(self, features=None, seed=42, verbose=0, path=None, **kwargs):
        # Column names
        self.id_col = 'id'  # User ID
        self.tfs_col = 'tfs'  # Current cycle
        self.tgt_col = 'tgt'  # Target value

        # Execution params
        self.seed = seed  # Base random seed
        self.verbose = verbose  # Print progress
        self.path = path or os.getcwd()  # Path to store model files

        # Model params
        self.params = {
            'n': kwargs.get('n', 1000),
            'lr': kwargs.get('lr', 1e-2),
            'max_depth': kwargs.get('max_depth', 3),
            'stop': kwargs.get('stop', 100),
            'metric': kwargs.get('metric', 'auc'),
            'min_child_weight': kwargs.get('min_child_weight', 1),
            'gamma': kwargs.get('gamma', 0),
            'weight_l1': kwargs.get('weight_l1', 0),
            'weight_l2': kwargs.get('weight_l2', 1),
            'dropout': kwargs.get('dropout', 0.2),
            'shuffle': kwargs.get('shuffle', False),
            'reg_unb': kwargs.get('reg_unb', False)
        }

        # Model instance
        self.model = None

        # Rules
        self.features = features or []  # Features

        # Results
        self.history = pd.DataFrame()  # Training history
        self.results = pd.DataFrame()  # Prediction results
        self.thr = .5  # Minimum probability thereshold
        self.scores = {
            'acc': .0,
            'auc': .0,
            'f1': .0,
            'precision': .0,
            'recall': .0
        }

    def build_model(self):
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
            silent=False if self.verbose > 0 else True,
            n_jobs=-1,
            random_state=self.seed
        )

        return model

    def fit(self, x_train, y_train, x_test, y_test):
        # Build sequences
        x_train, y_train, _ = self.input_seq(x_train, y_train, shuffle_data=self.params['shuffle'])
        x_test, y_test, _ = self.input_seq(x_test, y_test)

        # Load model
        self.model = self.build_model()

        unb_ratio = float(round(len(y_train[y_train == 0]) / len(y_train[y_train == 1]))) if self.params['reg_unb'] > 0 else None
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

    def predict(self, x):
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

    def set_results(self, y_pred, y_true=None):
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

        if y_true is not None:
            self.scores['auc'] = roc_auc_score(self.results.true.values, self.results.pred.values)

            self.thr, thrs, f1, precision, recall = self.precision_recall(self.results.true.values, self.results.pred.values)
            best = thrs.index(self.thr)

            self.scores['f1'] = f1[best]
            self.scores['precision'] = precision[best]
            self.scores['recall'] = recall[best]

            y_tgt = np.array([1 if i > self.thr else 0 for i in self.results.pred.values])
            self.scores['acc'] = accuracy_score(self.results.true.values, y_tgt)

        return self

    def build_seq(self, data):
        if self.features is None:
            self.features = list(data.columns[~data.columns.isin([self.id_col, self.tfs_col, self.tgt_col])])

        if self.tfs_col not in list(data.columns):
            data[self.tfs_col] = 0

        if self.tgt_col not in list(data.columns):
            data[self.tgt_col] = 0

        x = data[[self.id_col, self.tfs_col, self.tgt_col] + self.features].values
        y = data[[self.id_col, self.tfs_col, self.tgt_col]].values

        return x, y

    def input_seq(self, x, y=None, shuffle_data=False):
        z = x[:, :2]
        x = x[:, 3:]
        if y is not None:
            y = y[:, 2]

        if shuffle_data:
            if y is not None:
                x, y, z = shuffle(x, y, z, random_state=self.seed)
            else:
                x, z = shuffle(x, z, random_state=self.seed)

        return x, y, z

    def output_seq(self, y, z):
        n_seq = y.shape[0]

        y = y.reshape(n_seq, 1)
        seq = np.hstack([z, y])

        return seq

    def seq_to_df(self, x, y=None):
        n_feat = len(self.features)

        if y is not None:
            x = x[:, -n_feat:]
            df = pd.DataFrame(np.hstack([y, x]), columns=[self.id_col, self.tfs_col, self.tgt_col] + self.features)
        else:
            df = pd.DataFrame(x, columns=[self.id_col, self.tfs_col, self.tgt_col])

        return df

    def get_columns(self, x, exclude=None, dtypes=None):
        exclude = exclude or []
        dtypes = dtypes or []

        if len(dtypes) > 0:
            x = x.select_dtypes(include=set(dtypes))

        c = x.columns

        if len(exclude) > 0:
            c = c[~c.isin(exclude)]

        return list(c)
    
    def get_clusters(self, y, n):
        cm = KMeans(
            n_clusters=n,
            random_state=self.seed
        ).fit(pd.DataFrame(y).values)

        clusters = np.arange(1, n + 1)
        labels = pd.Series([clusters[i] for i in cm.labels_], index=y.index)
        centers = pd.DataFrame(cm.cluster_centers_, index=clusters).loc[:, 0].sort_values()

        c = labels.apply(lambda x: centers.index[x - 1])

        return c

    def precision_recall(self, y_true, y_pred):
        f1 = []
        precision = []
        recall = []

        thrs = [x / 100. for x in range(0, 105, 5)]

        for x in thrs:
            t_pred = np.array([1. if i > x else 0. for i in y_pred])
            f1.append(f1_score(y_true, t_pred, pos_label=1))
            precision.append(precision_score(y_true, t_pred, pos_label=1))
            recall.append(recall_score(y_true, t_pred, pos_label=1))

        best = np.argmax(f1)
        thr = thrs[best]

        return thr, thrs, f1, precision, recall

    def plot_scores(self, show=True, file=None):
        summary = ['Thereshold: {}%'.format(format_number(self.thr * 100, 0))]
        for key, val in self.scores.items():
            summary.append('{}: {}%'.format(key.upper(), format_number(val * 100, 0)))

        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 14), constrained_layout=True)
        axs = axs.flatten()

        self.plot_roc_curve(axs[0])
        self.plot_precision_recall(axs[1])
        self.plot_feature_importances(axs[2])
        self.plot_confusion_matrix(axs[3])

        plt.suptitle(' | '.join(summary), y=1.03)

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_roc_curve(self, ax=None, show=True, file=None):
        y_true = self.results['true']
        y_pred = self.results['pred']

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

    def plot_precision_recall(self, ax=None, show=True, file=None):
        y_true = self.results['true']
        y_pred = self.results['pred']

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

    def plot_feature_importances(self, ax=None, show=True, file=None):
        fimp = pd.Series(self.model.feature_importances_, index=self.features).sort_values(ascending=False)

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

    def plot_confusion_matrix(self, ax=None, show=True, file=None):
        y_true = self.results['true']
        y_pred = self.results['pred']
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

    def plot_histogram(self, y, loc=-1, ax=None, show=True, file=None):
        if loc:
            y = y.sort_values(self.tfs_col).groupby(self.id_col).nth(loc).reset_index()

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 8))
        else:
            fig = None

        ax.hist(
            y['pred'] * 100, bins=50, color=get_color('blue'), rwidth=0.9
        )

        ax.set_xticks(np.arange(0, 110, 10))
        ax.set_xticklabels([f'{i}%' for i in ax.get_xticks()])

        ax.set_title('Prediction Probas')

        self.plot_figure(fig=fig, show=show, file=file)

    def plot_history_eval(self, ax=None, show=True, file=None):
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

    def model_exists(self):
        return self.file_exists('model.pkl')
    
    def model_save(self, model):
        path = self.get_path('model.pkl')
        joblib.dump(model, path)

        return path

    def model_load(self):
        path = self.get_path('model.pkl')

        return joblib.load(path)

    def save(self):
        params = json.dumps({
            k: v for k, v in self.__dict__.items()
            if k not in ['model', 'history', 'results', 'seed', 'verbose', 'path']
        })
        with open(self.get_path('params.json'), 'w') as fh:
            fh.write(params)

        if self.model is not None:
            self.model_save(self.model)

        if self.history is not None:
            self.history.to_json(self.get_path('history.json'), double_precision=10)

        if not self.results.empty:
            self.results.to_json(self.get_path('results.json'))

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

        if self.file_exists('results.json'):
            self.results = pd.read_json(self.get_path('results.json'))

        return self
