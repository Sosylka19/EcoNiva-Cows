import pandas as pd
import os
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import time
import joblib

warnings.filterwarnings('ignore')

class CatBoostFattyAcidsPredictor:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.target_columns = ['лауриновая', 'стеариновая', 'пальмитиновая', 
                              'олеиновая', 'линолевая', 'линоленовая']
        self.best_params = {}
        
    def load_and_preprocess_data(self, file_path, test_size=0.2, feature_engineering=True):
        """
        Загрузка и предобработка данных с разделением на train/test
        """
        self.df = pd.read_excel(file_path)
    
        X_full = self.df.drop(columns=['Рецепты', 'Unnamed: 0', 
                                      'лауриновая', 'стеариновая', 'пальмитиновая', 
                                      'олеиновая', 'линолевая', 'линоленовая'])
        
        y_full = self.df[self.target_columns]
        
        X_full = X_full.fillna(0)
        

        if feature_engineering:
            X_full = self._apply_feature_engineering(X_full)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_full, y_full, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Данные загружены и разделены: {len(self.X_train)} train, {len(self.X_test)} test")
        if feature_engineering:
            print(f"Применено feature engineering: {X_full.shape[1]} признаков")
        
        return self.X_train, self.y_train, self.X_test, self.y_test
    
    def _apply_feature_engineering(self, X):
        """
        Применение feature engineering для улучшения качества модели
        """
        X_eng = X.copy()
        

        important_features = []
        for col in X.columns:
            if X[col].std() > 0 and X[col].nunique() > 5: 
                important_features.append(col)
        
        important_features = important_features[:20]
        
        for feature in important_features:
            X_eng[f'{feature}_squared'] = X[feature] ** 2
        
        top_features = important_features[:10]
        for i, feat1 in enumerate(top_features):
            for feat2 in top_features[i+1:]:
                X_eng[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_eng['mean_features'] = X[numeric_cols].mean(axis=1)
            X_eng['std_features'] = X[numeric_cols].std(axis=1)
            X_eng['sum_features'] = X[numeric_cols].sum(axis=1)
        
        return X_eng
    
    def get_hyperparams_grid(self, target_name):
        base_params = {
            'random_seed': [self.random_state],
            'verbose': [False],
            'loss_function': ['RMSE'],
            'od_type': ['Iter'],
        }

        if target_name in ['лауриновая', 'стеариновая', 'олеиновая']:
            param_grid = {
                **base_params,
                'iterations': [800, 1200],           
                'learning_rate': [0.03, 0.05],       
                'depth': [5, 7],                    
                'l2_leaf_reg': [1, 3, 5],             
                'bagging_temperature': [0.5, 1],      
                'random_strength': [0.5, 1]           
            }
        elif target_name in ['пальмитиновая']:
            param_grid = {
                **base_params,
                'iterations': [1200, 1600],
                'learning_rate': [0.06, 0.1],
                'depth': [6, 9],
                'l2_leaf_reg': [3, 7],
                'bagging_temperature': [0.5],
                'random_strength': [1],
            }
        else:
            param_grid = {
                **base_params,
                'iterations': [1000, 1400],
                'learning_rate': [0.05, 0.08],
                'depth': [6, 8],
                'l2_leaf_reg': [2, 5],
                'bagging_temperature': [0.5],
                'random_strength': [1],
            }

        return param_grid
    
    def search_best_hyperparams(self, target, n_splits=3):
        """
        Поиск лучших гиперпараметров с помощью GridSearchCV
        """
        print(f"\n Поиск лучших гиперпараметров для: {target}")
        
        param_grid = self.get_hyperparams_grid(target)
        
        sample_size = min(2000, len(self.X_train))
        if len(self.X_train) > sample_size:
            sample_idx = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_sample = self.X_train.iloc[sample_idx]
            y_sample = self.y_train[target].iloc[sample_idx]
        else:
            X_sample = self.X_train
            y_sample = self.y_train[target]
        
        catboost_model = CatBoostRegressor()
        
        cv_folds = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=catboost_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        total_combinations = (len(param_grid['iterations']) * len(param_grid['learning_rate']) * 
                            len(param_grid['depth']) * len(param_grid['l2_leaf_reg']) * 
                            len(param_grid['bagging_temperature']) * len(param_grid['random_strength']))
        print(f"  Поиск по {total_combinations} комбинациям...")
        start_time = time.time()
        
        grid_search.fit(X_sample, y_sample)
        
        search_time = time.time() - start_time
        print(f"  Поиск завершен за {search_time:.2f} сек")
        print(f"  Лучший RMSE: {np.sqrt(-grid_search.best_score_):.4f}")
        
        self.best_params[target] = grid_search.best_params_
        
        return grid_search.best_params_
    
    def cross_validate_target(self, target, n_splits=5):
        """
        Кросс-валидация для одной целевой переменной (только на обучающей выборке)
        """
        print(f"\n Кросс-валидация для: {target}")
        
        X_array = self.X_train.values
        y_array = self.y_train[target].values
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        fold_scores = {
            'r2': [], 'rmse': [], 'mae': [], 
            'train_time': [], 'models': []
        }
        
        if target in self.best_params:
            params = self.best_params[target]
        else:
            params = {
                'random_seed': self.random_state,
                'verbose': False,
                'early_stopping_rounds': 50,
                'loss_function': 'RMSE',
                'od_type': 'Iter',
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'bagging_temperature': 1,
                'random_strength': 1
            }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_array)):
            print(f"  Fold {fold+1}/{n_splits}...", end=" ")
            start_time = time.time()
            
            X_train_fold, X_val_fold = X_array[train_idx], X_array[val_idx]
            y_train_fold, y_val_fold = y_array[train_idx], y_array[val_idx]
            
            model = CatBoostRegressor(**params)
            
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                verbose=False
            )
            
            y_pred_fold = model.predict(X_val_fold)
            

            r2_fold = r2_score(y_val_fold, y_pred_fold)
            rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
            mae_fold = mean_absolute_error(y_val_fold, y_pred_fold)
            
            train_time = time.time() - start_time
            
            fold_scores['r2'].append(r2_fold)
            fold_scores['rmse'].append(rmse_fold)
            fold_scores['mae'].append(mae_fold)
            fold_scores['train_time'].append(train_time)
            fold_scores['models'].append(model)
            
            print(f"R2: {r2_fold:.4f}, RMSE: {rmse_fold:.4f}")
        
        return fold_scores
    
    def search_all_hyperparams(self, n_splits=3):
        """
        Поиск лучших гиперпараметров для всех целевых переменных
        """
        print("Начало поиска лучших гиперпараметров для всех целевых переменных")
        
        for target in self.target_columns:
            self.search_best_hyperparams(target, n_splits)
        
        print("\nПоиск гиперпараметров завершен!")
        
        for target, params in self.best_params.items():
            print(f"\n{target}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
    
    def train_with_cross_validation(self, n_splits=5, search_hyperparams=False, hyperparams_splits=3):
        """
        Основной метод обучения с кросс-валидацией (только на обучающей выборке)
        
        Args:
            n_splits: количество фолдов для кросс-валидации
            search_hyperparams: выполнять ли поиск гиперпараметров
            hyperparams_splits: количество фолдов для поиска гиперпараметров
        """
        if search_hyperparams:
            self.search_all_hyperparams(hyperparams_splits)
        
        print("Начало обучения с кросс-валидацией")
        
        for target in self.target_columns:
            cv_results = self.cross_validate_target(target, n_splits)
            
            best_model_idx = np.argmax(cv_results['r2'])
            best_model = cv_results['models'][best_model_idx]
            
            if target in self.best_params:
                final_params = self.best_params[target]
            else:
                final_params = {
                    'random_seed': self.random_state,
                    'verbose': False,
                    'early_stopping_rounds': 50,
                    'loss_function': 'RMSE',
                    'od_type': 'Iter',
                    'iterations': 1000,
                    'learning_rate': 0.05,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'bagging_temperature': 1,
                    'random_strength': 1
                }
            
            final_model = CatBoostRegressor(**final_params)
            final_model.fit(self.X_train, self.y_train[target], verbose=False)
            
            self.models[target] = final_model
            self.cv_results[target] = {
                'mean_r2': np.mean(cv_results['r2']),
                'std_r2': np.std(cv_results['r2']),
                'mean_rmse': np.mean(cv_results['rmse']),
                'std_rmse': np.std(cv_results['rmse']),
                'mean_mae': np.mean(cv_results['mae']),
                'std_mae': np.std(cv_results['mae']),
                'mean_train_time': np.mean(cv_results['train_time']),
                'fold_scores': cv_results
            }
            
            self.feature_importance[target] = self._get_feature_importance(final_model)
        
        self.is_fitted = True
        print("\n обучение завершено")
    
    def _get_feature_importance(self, model):
        """Получение важности признаков"""
        importance = model.get_feature_importance()
        feature_names = self.X_train.columns
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def evaluate_on_test_set(self):
        """
        Оценка на тестовой выборке (которая была отделена при загрузке данных)
        """
        if not self.is_fitted:
            raise ValueError("Модели не обучены! Сначала вызовите train_with_cross_validation()")
        
        print("\nоценка на тесте")
        
        test_results = {}
        
        for target in self.target_columns:
            model = self.models[target]
            y_pred = model.predict(self.X_test)
            
            r2 = r2_score(self.y_test[target], y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test[target], y_pred))
            mae = mean_absolute_error(self.y_test[target], y_pred)
            
            test_results[target] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae
            }
            
            print(f"{target:15} | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
        
        return test_results
    
    def print_cv_summary(self):
        """
        Вывод сводки по кросс-валидации
        """
        print("сводка по кв")

        
        summary_data = []
        for target in self.target_columns:
            results = self.cv_results[target]
            summary_data.append({
                'Target': target,
                'Mean R2': f"{results['mean_r2']:.4f} ± {results['std_r2']:.4f}",
                'Mean RMSE': f"{results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}",
                'Mean MAE': f"{results['mean_mae']:.4f} ± {results['std_mae']:.4f}",
                'Time (s)': f"{results['mean_train_time']:.2f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        avg_r2 = np.mean([self.cv_results[target]['mean_r2'] for target in self.target_columns])
        avg_rmse = np.mean([self.cv_results[target]['mean_rmse'] for target in self.target_columns])
        
        print(f"\n средние метрики по всем целевым переменным:")
        print(f"   Средний R2:  {avg_r2:.4f}")
        print(f"   Средний RMSE: {avg_rmse:.4f}")
    
    def plot_feature_importance(self, top_n=15):
        """
        Визуализация важности признаков
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for idx, target in enumerate(self.target_columns):
            importance_df = self.feature_importance[target].head(top_n)
            
            axes[idx].barh(range(len(importance_df)), importance_df['importance'])
            axes[idx].set_yticks(range(len(importance_df)))
            axes[idx].set_yticklabels(importance_df['feature'])
            axes[idx].set_title(f'Важность признаков: {target}')
            axes[idx].set_xlabel('Важность')
        
        plt.tight_layout()
        os.makedirs('./data', exist_ok=True)
        plt.savefig('./data/feature_importance.png', )
        plt.show()
    
    def plot_cv_results(self):
        """
        Визуализация результатов кросс-валидации
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes = axes.ravel()
        
        for idx, target in enumerate(self.target_columns):
            fold_scores = self.cv_results[target]['fold_scores']['r2']
            axes[idx].plot(range(1, len(fold_scores) + 1), fold_scores, 'o-', linewidth=2, markersize=8)
            axes[idx].axhline(y=self.cv_results[target]['mean_r2'], color='r', linestyle='--', 
                            label=f'Mean: {self.cv_results[target]["mean_r2"]:.4f}')
            axes[idx].set_xlabel('Fold')
            axes[idx].set_ylabel('R2 Score')
            axes[idx].set_title(f'R2 по фолдам: {target}')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs('./data', exist_ok=True)
        plt.savefig('./data/cv_results.png')
        plt.show()
    
    def predict_new_data(self, new_data):
        """
        Предсказание на новых данных
        """
        if not self.is_fitted:
            raise ValueError("Модели не обучены!")
        
        feature_names = None
        try:
            any_target = self.target_columns[0]
            if isinstance(self.feature_importance.get(any_target), pd.DataFrame):
                feature_names = self.feature_importance[any_target]['feature'].tolist()
        except Exception:
            feature_names = None

        if feature_names is None and hasattr(self, 'X_train') and isinstance(getattr(self, 'X_train'), pd.DataFrame):
            feature_names = list(self.X_train.columns)

        if feature_names is None:
            raise ValueError("Не удалось определить набор признаков модели для выравнивания входных данных")

        new_data = new_data.reindex(columns=feature_names, fill_value=0)

        predictions = {}
        for target in self.target_columns:
            predictions[target] = self.models[target].predict(new_data)
        
        return pd.DataFrame(predictions)
    
    def save_models(self, filepath_prefix='catboost_model'):
        """
        Сохранение обученных моделей
        """
        if not self.is_fitted:
            raise ValueError("Модели не обучены!")
        
        for target in self.target_columns:
            filename = f"data/{filepath_prefix}_{target}.cbm"
            self.models[target].save_model(filename)
            print(f" модель для {target} сохранена как: {filename}")
        
        results_filename = f"data/{filepath_prefix}_results.joblib"
        joblib.dump({
            'cv_results': self.cv_results,
            'feature_importance': self.feature_importance,
            'target_columns': self.target_columns,
            'best_params': self.best_params
        }, results_filename)
        
        print(f" результаты сохранены как: {results_filename}")
    
    def load_models(self, filepath_prefix='catboost_model'):
        """
        Загрузка обученных моделей
        """
        self.models = {}
        for target in self.target_columns:
            filename = f"data/{filepath_prefix}_{target}.cbm"
            self.models[target] = CatBoostRegressor()
            self.models[target].load_model(filename)
        
        results_filename = f"data/{filepath_prefix}_results.joblib"
        results_data = joblib.load(results_filename)
        self.cv_results = results_data['cv_results']
        self.feature_importance = results_data['feature_importance']
        self.best_params = results_data.get('best_params', {})
        
        self.is_fitted = True

def main():
    """
    Основной скрипт выполнения
    """
    print("CatBoost Predictor for Fatty Acids Analysis")
    
    predictor = CatBoostFattyAcidsPredictor(random_state=42)
    
    try:
        X_train, y_train, X_test, y_test = predictor.load_and_preprocess_data('./src/kis.xlsx', feature_engineering=True)
        
        predictor.train_with_cross_validation(n_splits=5, search_hyperparams=True, hyperparams_splits=3)

        predictor.print_cv_summary()
        
        test_results = predictor.evaluate_on_test_set()
        
        predictor.plot_cv_results()
        predictor.plot_feature_importance(top_n=15)
        
        predictor.save_models('fatty_acids_model')
        
        print("\n Пример предсказания на первых 5 samples из тестовой выборки:")
        sample_data = X_test.head(5)
        predictions = predictor.predict_new_data(sample_data)
        actual_values = y_test.head(5)
        
        comparison = pd.concat([actual_values, predictions.add_prefix('pred_')], axis=1)
        print(comparison.round(4))
        
    except Exception as e:
        print(f" Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()