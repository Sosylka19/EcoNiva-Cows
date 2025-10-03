import os
import json
import camelot
import gc
import time
import shutil
import pathlib


def main(list_ingredients):
    path = pathlib.Path('data')
    dirs = [x for x in path.iterdir() if x.is_dir()]
    out = {}
        
    for d in dirs:
        files =  [x for x in d.iterdir() if x.suffix == '.pdf']
        for f in files:
            try:
                # print(f"Обрабатывается файл: {f}")
                
                tables = camelot.read_pdf(
                    f,
                    pages='all',
                    flavor='lattice'
                )

                list_ingredients[f] = tables[0]
                
                # jslist = []
                # for t in tables:
                #     df = t.df
                #     jslist.append(df.to_json(orient='records', force_ascii=False))
                
                # out[str(f)[:-4]] = jslist
                
                # for table in tables:
                #     if hasattr(table, '_tempdir') and table._tempdir:
                #         try:
                #             time.sleep(0.1)  
                #             shutil.rmtree(table._tempdir, ignore_errors=True)
                #         except:
                #             pass
                
                # del tables
                # gc.collect()
                # time.sleep(0.5)  
                
            except Exception as e:
                print(f"Ошибка при обработке файла {f}: {e}")
                time.sleep(0.5)  
                continue
    
    # time.sleep(120)
    # with open("data.json", "w", encoding='utf-8') as file:
    #     json.dump(out, file, ensure_ascii=False, indent=2)
    
    print("Обработка завершена успешно!")

if __name__ == '__main__':
    list_ingredients = {}
    main(list_ingredients)
    print(list_ingredients)