import pathlib
import pandas as pd
import camelot
import re
import sys

RE_DATE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
RE_LONG_CODE = re.compile(r"\b(?:\d{1,4}\.){2,}\d{1,4}\b") 
RE_YEAR = re.compile(r"\b(19|20)\d{2}\b")
RE_TRAIL_DOTNUM = re.compile(r"(?<=\s)[\.\-]\d{1,3}\b")    
RE_PERCENT_TAIL = re.compile(r"\d+\s?%.*$")                
RE_MULTI_SPACE = re.compile(r"\s+")
RE_SP_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")
RE_PUNCT_DUP = re.compile(r"[,.]{2,}")

TAILKEYS = r"(суха[яй]|мелк|помол|сечк|ток\d*|гранул|экструд|смесь|мешан|энпкх|энанпкх|энапкх|эн|энк)"
RE_TAIL_AFTER_COMMA = re.compile(rf",\s*.*?(?=($|\b))", re.IGNORECASE)
RE_TAILKEYS_AFTER_COMMA = re.compile(rf",\s*(?={TAILKEYS})[^\n]*", re.IGNORECASE)

def clean_ingredient(s: str) -> str:
    """
    Clear ingredients with regulat expressions
    """
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)

    s = s.replace("\n", " ").strip()

    s = re.sub(r"\bкомбиком\b", "комбикорм", s, flags=re.IGNORECASE)

    masks = {}
    # оставляем комбикормы номера
    def _mask(match):
        key = f"__MASKNO_{len(masks)}__"
        masks[key] = match.group(0)
        return key
    s = re.sub(r"№\s*\d+", _mask, s) 

    s = RE_DATE.sub("", s)
    s = RE_LONG_CODE.sub("", s)
    s = RE_YEAR.sub("", s)
    s = RE_TRAIL_DOTNUM.sub("", s)

    s = RE_PERCENT_TAIL.sub("", s)

    s = RE_TAILKEYS_AFTER_COMMA.sub("", s)

    s = s.replace("/", " ")

    s = RE_SP_BEFORE_PUNCT.sub(r"\1", s)
    s = RE_PUNCT_DUP.sub(lambda m: m.group(0)[0], s)
    s = RE_MULTI_SPACE.sub(" ", s).strip(" ,.;:")

    for key, val in masks.items():
        s = s.replace(key, val)

    s = s.rstrip(".,;: ").strip()

    return s


def preprocessor(table: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess table
    """
    table_columns = table.iloc[1]
    table.columns = table_columns
    table.columns.name = None
    table = table.drop(index=[0, 1])
    table = table.reset_index(drop=True)
    return (pd.DataFrame(table).iloc[:, :5])

def extract_table_from_pdf(path: pathlib.Path) -> pd.DataFrame:
    """
    Parsing tables
    """
    f = str(path)
        
    try:
        tables = camelot.read_pdf(
            f,
            flavor='lattice'
        )

        data = tables[0]
        data_preproc = preprocessor(data.df)
        data_preproc['Ингредиенты'] = data_preproc['Ингредиенты'].map(clean_ingredient)
        data_preproc = data_preproc.drop_duplicates(subset=['Ингредиенты'], keep='first')
        

        
    except Exception as e:
        sys.exit(f'{e}')

    return data_preproc

def extract_table_from_excel(path: str) -> pd.DataFrame | None:
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
