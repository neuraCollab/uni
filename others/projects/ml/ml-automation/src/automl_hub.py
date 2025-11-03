import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import get_scorer
from .utils import get_metric_for_backend
from .utils import simple_text_preprocess, embed_texts
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

class AutoMLHub:
    def __init__(self, config_path="config/config.yaml", **kwargs):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Переопределение конфига из kwargs
        for k, v in kwargs.items():
            if k in self.config:
                self.config[k] = v

        self.backend = self.config['backend']
        self.task_type = self.config['task_type']
        self.metric = self.config['metric']
        self.time_budget = self.config.get('time_budget', 300)
        self.random_state = self.config.get('random_state', 42)
        self.n_jobs = self.config.get('n_jobs', -1)

        self.model = None
        self.is_fitted = False
    # хранит обработчики текста: dict col -> {'mode': 'embed'|'tfidf', ...}
    self._text_processors = {}
    # список колонок, используемых при обучении (последовательность важна для FLAML)
    self.train_columns = None

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        print(f"🚀 Запуск AutoML через {self.backend}...")

        # Если есть текстовые колонки, попробуем их автоматически обработать и ЗАПОМНИТЬ процессоры
        self._text_columns = []
        if isinstance(X, pd.DataFrame):
            for col in X.select_dtypes(include=['object', 'string']).columns:
                non_null_frac = X[col].notnull().mean()
                if non_null_frac > 0.0:
                    self._text_columns.append(col)
        if self._text_columns:
            print(f"ℹ️ Обнаружены текстовые столбцы: {self._text_columns}. Попытка автоматической предобработки/эмбеддингов...")
            X = X.copy()
            # очистим старые процессоры
            self._text_processors = {}
            for col in self._text_columns:
                proc = simple_text_preprocess(X[col])
                embs = embed_texts(proc)
                if embs is not None:
                    # сохраним информацию об эмбеддингах
                    dim = embs.shape[1]
                    self._text_processors[col] = {'mode': 'embed', 'dim': dim}
                    # вставим эмбеддинги в X, удалив исходную колонку
                    emb_df = pd.DataFrame(embs, index=X.index).add_prefix(f"{col}_emb_")
                    X = pd.concat([X.drop(columns=[col]), emb_df], axis=1)
                else:
                    # fallback: TF-IDF vectorization и сохранение векторайзера
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    tf = TfidfVectorizer(max_features=200)
                    tfm = tf.fit_transform(proc)
                    tf_df = pd.DataFrame(tfm.toarray(), index=X.index).add_prefix(f"{col}_tf_")
                    self._text_processors[col] = {'mode': 'tfidf', 'vectorizer': tf, 'n_features': tf_df.shape[1]}
                    X = pd.concat([X.drop(columns=[col]), tf_df], axis=1)

        # После всех преобразований сохраним порядок колонок, чтобы использовать его при predict
        if isinstance(X, pd.DataFrame):
            self.train_columns = X.columns.tolist()

        if self.backend == "flaml":
            self._fit_flaml(X, y)
        elif self.backend == "h2o":
            self._fit_h2o(X, y)
        elif self.backend == "tpot":
            self._fit_tpot(X, y)
        elif self.backend == "autosklearn":
            self._fit_autosklearn(X, y)
        elif self.backend == "pycaret":
            self._fit_pycaret(X, y)
        else:
            raise ValueError(f"Неизвестный backend: {self.backend}")

        self.is_fitted = True
        print("✅ Обучение завершено.")

    def _fit_flaml(self, X, y):
        from flaml import AutoML
        automl = AutoML()
        settings = {
            "time_budget": self.time_budget,
            "metric": get_metric_for_backend(self.metric, "flaml"),
            "task": self.task_type,
            "seed": self.random_state,
            "n_jobs": self.n_jobs,
            **self.config.get('flaml', {})
        }
        automl.fit(X, y, **settings)
        self.model = automl
        self.best_model_name = automl.best_estimator

    def _fit_h2o(self, X, y):
        import h2o
        from h2o.automl import H2OAutoML
        h2o.init()
        train = h2o.H2OFrame(pd.concat([X, y.to_frame(name='target')], axis=1))
        x_cols = X.columns.tolist()
        y_col = 'target'

        aml = H2OAutoML(
            max_runtime_secs=self.time_budget,
            sort_metric=get_metric_for_backend(self.metric, "h2o"),
            seed=self.random_state,
            **self.config.get('h2o', {})
        )
        aml.train(x=x_cols, y=y_col, training_frame=train)
        self.model = aml
        self.best_model_name = aml.leaderboard[0, "model_id"]

    def _fit_tpot(self, X, y):
        from tpot import TPOTClassifier, TPOTRegressor
        cls = TPOTClassifier if self.task_type == "classification" else TPOTRegressor
        tpot = cls(
            generations=self.config['tpot'].get('generations', 5),
            population_size=self.config['tpot'].get('population_size', 20),
            scoring=self.metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=2
        )
        tpot.fit(X, y)
        self.model = tpot
        self.best_model_name = str(tpot.fitted_pipeline_)

    def _fit_autosklearn(self, X, y):
        from autosklearn.classification import AutoSklearnClassifier
        from autosklearn.regression import AutoSklearnRegressor
        cls = AutoSklearnClassifier if self.task_type == "classification" else AutoSklearnRegressor
        automl = cls(
            time_left_for_this_task=self.time_budget,
            metric=get_metric_for_backend(self.metric, "autosklearn"),
            seed=self.random_state,
            n_jobs=self.n_jobs
        )
        automl.fit(X, y)
        self.model = automl
        self.best_model_name = "AutoSklearn ensemble"

    def _fit_pycaret(self, X, y):
        from pycaret.classification import setup, compare_models, pull
        from pycaret.regression import setup as setup_reg, compare_models as compare_models_reg, pull as pull_reg

        data = X.copy()
        data['target'] = y

        if self.task_type == "classification":
            setup(data, target='target', silent=True, session_id=self.random_state)
            best = compare_models(sort=self.metric, n_select=1)
            self.model = best
            leaderboard = pull()
        else:
            setup_reg(data, target='target', silent=True, session_id=self.random_state)
            best = compare_models_reg(sort=self.metric, n_select=1)
            self.model = best
            leaderboard = pull_reg()

        self.best_model_name = type(best).__name__

    def predict(self, X):
        if not self.is_fitted:
            raise Exception("Модель не обучена!")
        if isinstance(X, np.ndarray):
            # Если у нас есть train_columns, используем их как имена колонок; иначе создаём DataFrame без имён
            if self.train_columns is not None:
                X = pd.DataFrame(X, columns=self.train_columns)
            else:
                X = pd.DataFrame(X)

        if self.backend == "h2o":
            import h2o
            X_h2o = h2o.H2OFrame(X)
            preds = self.model.predict(X_h2o)
            return preds.as_data_frame().iloc[:, 0].values
        elif self.backend == "autosklearn" or self.backend == "flaml" or self.backend == "tpot":
            # Ensure columns align with train
            if self.train_columns is not None and isinstance(X, pd.DataFrame):
                for c in self.train_columns:
                    if c not in X.columns:
                        X[c] = 0
                # drop any extra columns not seen in training
                X = X[self.train_columns]
            return self.model.predict(X)
        elif self.backend == "pycaret":
            from pycaret.classification import predict_model
            from pycaret.regression import predict_model as predict_model_reg
            fn = predict_model if self.task_type == "classification" else predict_model_reg
            pred_df = fn(self.model, data=X)
            return pred_df['prediction_label'].values

    def predict_proba(self, X):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("Модель не поддерживает predict_proba")
        if isinstance(X, np.ndarray):
            if self.train_columns is not None:
                X = pd.DataFrame(X, columns=self.train_columns)
            else:
                X = pd.DataFrame(X)

        # Обработаем текстовые колонки так же, как при обучении, используя сохранённые процессоры
        if hasattr(self, '_text_processors') and self._text_processors:
            X = X.copy()
            for col, proc_info in self._text_processors.items():
                # Если исходной текстовой колонки в X нет — пропускаем (вместо неё будут нулевые эмбеддинги/фичи)
                if col in X.columns:
                    proc = simple_text_preprocess(X[col])
                    if proc_info['mode'] == 'embed':
                        # Попытаемся получить эмбеддинги с тем же размером
                        embs = embed_texts(proc)
                        if embs is None:
                            # если не удалось получить, создаём нулевые эмбеддинги
                            embs = np.zeros((len(X), proc_info['dim']))
                        emb_df = pd.DataFrame(embs, index=X.index).add_prefix(f"{col}_emb_")
                        X = pd.concat([X.drop(columns=[col]), emb_df], axis=1)
                    elif proc_info['mode'] == 'tfidf':
                        vec = proc_info.get('vectorizer')
                        if vec is not None:
                            tfm = vec.transform(proc)
                            tf_df = pd.DataFrame(tfm.toarray(), index=X.index).add_prefix(f"{col}_tf_")
                        else:
                            # если по какой-то причине векторайзер потерян — создаём нулевые колонки
                            n = proc_info.get('n_features', 0)
                            tf_df = pd.DataFrame(np.zeros((len(X), n)), index=X.index).add_prefix(f"{col}_tf_")
                        X = pd.concat([X.drop(columns=[col]), tf_df], axis=1)

        # Синхронизируем колонки: добавим недостающие и упорядочим
        if self.train_columns is not None and isinstance(X, pd.DataFrame):
            for c in self.train_columns:
                if c not in X.columns:
                    X[c] = 0
            X = X[self.train_columns]

        return self.model.predict_proba(X)

    def plot_feature_importance(self, top_n=20, figsize=(8,6), savepath=None):
        """Построить важности признаков для моделей, где это поддерживается.

        Для flaml/auto-sklearn/ensemble sklearn models используем доступные атрибуты.
        """
        if not self.is_fitted:
            raise Exception("Модель не обучена!")

        fi = None
        # flaml: .feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            fi = self.model.feature_importances_
            cols = None
        elif getattr(self.model, 'best_estimator_', None) is not None and hasattr(self.model.best_estimator_, 'feature_importances_'):
            fi = self.model.best_estimator_.feature_importances_
            cols = None
        elif hasattr(self.model, 'feature_importances'):
            try:
                fi = np.array(self.model.feature_importances())
            except Exception:
                fi = None

        if fi is None:
            raise Exception("Feature importance недоступна для этого типа модели")

        # Попытаемся получить имена колонок из обучающих данных, если они были сохранены
        try:
            cols = getattr(self, 'train_columns', None)
        except Exception:
            cols = None

        if cols is None:
            cols = [f"f_{i}" for i in range(len(fi))]

        idx = np.argsort(fi)[::-1][:top_n]
        plot_df = pd.DataFrame({'feature': np.array(cols)[idx], 'importance': fi[idx]})

        plt.figure(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=plot_df)
        plt.title('Feature importance')
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath)
        plt.show()

    def plot_roc_auc(self, X, y, figsize=(8,6), savepath=None):
        """Построить ROC-кривую и вывести AUC для задач классификации.
        """
        if not self.is_fitted:
            raise Exception("Модель не обучена!")
        if self.task_type != 'classification':
            raise Exception('ROC применимо только для классификации')

        # Ensure preprocess text columns
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if hasattr(self, '_text_columns') and self._text_columns:
            for col in list(self._text_columns):
                if col in X.columns:
                    proc = simple_text_preprocess(X[col])
                    embs = embed_texts(proc)
                    if embs is not None:
                        emb_df = pd.DataFrame(embs, index=X.index).add_prefix(f"{col}_emb_")
                        X = pd.concat([X.drop(columns=[col]), emb_df], axis=1)
                    else:
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        tf = TfidfVectorizer(max_features=200)
                        tfm = tf.fit_transform(proc)
                        tf_df = pd.DataFrame(tfm.toarray(), index=X.index).add_prefix(f"{col}_tf_")
                        X = pd.concat([X.drop(columns=[col]), tf_df], axis=1)

        # predict probabilities (binary/multiclass handling)
        try:
            y_score = self.predict_proba(X)
        except Exception as e:
            raise

        # If multiclass, compute micro-average
        if y_score.ndim == 2 and y_score.shape[1] > 2:
            # binarize y
            from sklearn.preprocessing import label_binarize
            classes = np.unique(y)
            y_bin = label_binarize(y, classes=classes)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(y_score.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # micro-average
            fpr['micro'], tpr['micro'], _ = roc_curve(y_bin.ravel(), y_score.ravel())
            roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

            plt.figure(figsize=figsize)
            plt.plot(fpr['micro'], tpr['micro'], label=f'micro-average ROC (AUC = {roc_auc["micro"]:.3f})')
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC - micro-average')
            plt.legend(loc='lower right')
            if savepath:
                plt.savefig(savepath)
            plt.show()
            return roc_auc
        else:
            # binary
            if y_score.ndim == 2:
                y_score_pos = y_score[:,1]
            else:
                y_score_pos = y_score
            fpr, tpr, _ = roc_curve(y, y_score_pos)
            roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=figsize)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            if savepath:
                plt.savefig(savepath)
            plt.show()
            return roc_auc_val

    def score(self, X, y):
        scorer = get_scorer(self.metric)
        y_pred = self.predict(X)
        return scorer._score_func(y, y_pred)

    def get_best_model_name(self):
        return self.best_model_name

    def save_model(self, filepath):
        if self.backend == "h2o":
            import h2o
            model_id = self.model.leader.key
            h2o.save_model(model=h2o.get_model(model_id), path=filepath, force=True)
        else:
            joblib.dump(self.model, filepath)
        print(f"💾 Модель сохранена: {filepath}")

    @staticmethod
    def load_model(filepath, backend, task_type):
        automl = AutoMLHub(backend=backend, task_type=task_type)
        if backend == "h2o":
            import h2o
            h2o.init()
            automl.model = h2o.load_model(filepath)
        else:
            automl.model = joblib.load(filepath)
        automl.is_fitted = True
        return automl