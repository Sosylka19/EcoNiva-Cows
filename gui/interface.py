import io
import json
from typing import Optional, Tuple, List
import pathlib
import os
import shutil

import gradio as gr
import pandas as pd

from src.preprocessor import extract_table_from_pdf
from src.model import predictor, get_main_features

FA_COLUMNS = [
    "Лауриновая", "Стеариновая", "Пальмитиновая",
    "Олеиновая", "Линолевая", "Линоленовая"
]
APP_TITLE = "Анализ жирных кислот"
APP_DESC = (
    "1) Загрузите PDF или Excel с таблицей рациона.\n"
    "2) Отредактируйте извлечённую таблицу (при необходимости).\n"
    "3) Нажмите «Начать анализ», чтобы получить предсказания по жирным кислотам.\n"
    "Можно править таблицу и запускать анализ многократно."
)



def extract_table_from_excel(path: str) -> Optional[pd.DataFrame]:
    """
    Читает первую таблицу из Excel (первый лист) по пути к файлу.
    """
    try:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    except Exception:
        # fallback без явного указания engine
        df = pd.read_excel(path, sheet_name=0)
    if not df.empty:
        df.columns = [str(c).strip() for c in df.columns]
        return df
    return None


def preprocess_uploaded_file(file: gr.File) -> Tuple[pd.DataFrame, str]:
    """
    Унифицированный препроцессор. Возвращает (таблица, сообщение-статус).
    """
    if file is None:
        return pd.DataFrame(), "Файл не загружен."

    name = file.name or ""
    path = "data/" + os.path.basename(file)
    shutil.copyfile(name, path)


    if name.lower().endswith((".xlsx", ".xls")):
        table = extract_table_from_excel(path)
        if table is None or table.empty:
            return pd.DataFrame(), "Не удалось извлечь таблицу из Excel."
        return table, f"Извлечена таблица из Excel: {table.shape[0]}×{table.shape[1]}."
    elif name.lower().endswith(".pdf"):
        table = extract_table_from_pdf(path)
        if table is None or table.empty:
            return pd.DataFrame(), (
                "Не удалось найти таблицу в PDF. "
                "Проверьте качество PDF или попробуйте Excel."
            )
        return table, f"Найдена таблица в PDF: {table.shape[0]}×{table.shape[1]}."
    else:
        return pd.DataFrame(), "Поддерживаются PDF, XLSX, XLS."



def on_upload(file: gr.File) -> Tuple[pd.DataFrame, str]:
    table, msg = preprocess_uploaded_file(file)
    return table, msg


def on_analyze(current_table: pd.DataFrame) -> pd.DataFrame:
    try:
        return predictor(current_table)
    except Exception as e:
        return pd.DataFrame([{col: f"Ошибка: {e}" if i == 0 else "" for i, col in enumerate(FA_COLUMNS)}])


with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown(APP_DESC)

    with gr.Row():
        file_in = gr.File(label="Загрузите PDF / Excel", file_types=[".pdf", ".xlsx", ".xls"])

    status = gr.Markdown()

    with gr.Row():
        data_frame = gr.Dataframe(
            headers=['СВ %','ГП кг','СВ кг', '% ГП','% СВ'],
            value=pd.DataFrame(),
            interactive=True,
            wrap=True,
            label="Извлечённая таблица (редактируема)"
        )

    analyze_btn = gr.Button("Начать анализ", variant="primary")
    n_state = gr.State(3)
    gr.Markdown("## Предсказанные жирные кислоты")
    fa_table = gr.Dataframe(
        headers=FA_COLUMNS,
        value=pd.DataFrame(columns=FA_COLUMNS),
        interactive=False,
        wrap=True
    )
    features_box = gr.Markdown()

    file_in.change(fn=on_upload, inputs=file_in, outputs=[data_frame, status])
    def on_analyze_and_features(current_table: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, str]:
        # предсказания
        try:
            preds = predictor(current_table)
        except Exception as e:
            preds = pd.DataFrame([{col: f"Ошибка: {e}" if i == 0 else "" for i, col in enumerate(FA_COLUMNS)}])
        # фичи
        try:
            mapping = get_main_features(top_n=int(n))
            if not mapping:
                features_md = "Главные фичи недоступны. Обучите/сохраните результаты сначала."
            else:
                lines = ["## Корректировка рецепта"]
                lines.append("### Главные фичи по кислотам(формат данных: ингредиент + содержание ГП или СВ: вклад в кислоту):")
                
                for acid, items in mapping.items():
                    lines.append(f"\nЧтобы изменить **{acid}** кислоту, нужно изменять следующие параметры:")
                    for feat, imp in items:
                        lines.append(f"- {feat}: {imp:.4f}")
                features_md = "\n".join(lines)
        except Exception as e:
            features_md = f"Ошибка при получении фич: {e}"
        return preds, features_md

    analyze_btn.click(fn=on_analyze_and_features, inputs=[data_frame, n_state], outputs=[fa_table, features_box])

if __name__ == "__main__":
    demo.launch()