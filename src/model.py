import pandas as pd
import json
import joblib
from typing import Dict, List, Tuple

from src.baseline import CatBoostFattyAcidsPredictor

def preproc_data(data: pd.DataFrame) -> pd.DataFrame:
        columns = pd.read_pickle('./src/columns')
        columns = columns.to_list()
        columns = ['Рецепты'] + columns


        # преобразуем в одну строку и подгоняем под pivot talbe
        stacked = data.set_index('Ингредиенты').stack()
        wide_row = stacked.to_frame().T
        wide_row.columns = [f"{ingredient} {metric}" for ingredient, metric in wide_row.columns]

        wide_row = wide_row.reindex(columns=columns, fill_value=0)
        for col in wide_row.columns:
                if wide_row[col].dtype == object:
                        wide_row[col] = (
                                wide_row[col]
                                .astype(str)
                                .str.replace('\u00A0', '', regex=False) 
                                .str.replace(' ', '', regex=False)      
                                .str.replace(',', '.', regex=False)    
                        )
                        wide_row[col] = pd.to_numeric(wide_row[col], errors='coerce')
        wide_row = wide_row.fillna(0)
        return wide_row

def predictor(data: pd.DataFrame) -> pd.DataFrame:
        
        data = preproc_data(data)
        
        predictor = CatBoostFattyAcidsPredictor(random_state=42)
    
        try:
            predictor.load_models(filepath_prefix='fatty_acids_model')
            feature_importances = {}
            for i in predictor.target_columns:
                feature_importances[i]= predictor.feature_importance[i].feature.to_list()
            with open('data/feature_importances.json', 'w') as f:
                json.dump(feature_importances, f)

            data = predictor.predict_new_data(data)

        except Exception as e:
            print(f" Ошибка: {e}")
            import traceback
            traceback.print_exc()
        return data

def get_main_features(top_n: int = 3) -> Dict[str, List[Tuple[str, float]]]:
        """
        Возвращает топ-3 важных признаков для каждой кислоты.
        """
        json_path = 'data/feature_importances.json'
        try:
                with open(json_path, 'r', encoding='utf-8') as f:
                        data_json = json.load(f)
                result: Dict[str, List[Tuple[str, float]]] = {}
                for target, items in data_json.items():
                        sorted_items = sorted(
                                ((it.get('feature'), float(it.get('importance', 0))) for it in items),
                                key=lambda x: x[1], reverse=True
                        )
                        result[target] = sorted_items[:top_n]
                return result
        except Exception:
                pass

        try:
                jb = joblib.load('data/fatty_acids_model_results.joblib')
                fi = jb.get('feature_importance')
                result: Dict[str, List[Tuple[str, float]]] = {}
                if isinstance(fi, dict):
                        for target, df in fi.items():
                                try:
                                        top_df = df.sort_values('importance', ascending=False).head(top_n)
                                        result[target] = list(zip(top_df['feature'].tolist(), top_df['importance'].astype(float).tolist()))
                                except Exception:
                                        continue
                        if result:
                                return result
        except Exception:
                pass

        return {}

        
        