"""
Здесь будет связываться парсинг 
"""
import camelot

tables = camelot.read_pdf('data/hach/КН/Отчет_Д1 Аристово 11.07.25_КНВ.pdf')
tables.export('foo.csv', f='csv', compress=True)