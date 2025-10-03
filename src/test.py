import pandas as pd
import json

def merge_if_same_columns(df1, df2):
    if set(df1.columns) == set(df2.columns):
        return pd.concat([df1, df2], ignore_index=True)
    else:
        return None

with open("data.json", "r", encoding="utf-8") as file:
    data = json.load(file)

new_data = {}

for row in data:
    merged_group = []
    current_df = None
    
    for json_str in data[row]:
        df = pd.read_json(json_str)
        
        if current_df is None:
            current_df = df
        else:
            merged = merge_if_same_columns(current_df, df)
            if merged is not None:
                current_df = merged
            else:
                merged_group.append(current_df.to_json())
                current_df = df
    
    if current_df is not None:
        merged_group.append(current_df.to_json())
        
    new_data[row] = merged_group
    print(f"Обработано: {row}")

with open("data_new.json", "w", encoding='utf-8') as file:
    json.dump(new_data, file, ensure_ascii=False, indent=2)