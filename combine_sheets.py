import os
import pandas as pd

writer = pd.ExcelWriter('out.xlsx', engine='xlsxwriter')
all_files = os.listdir('Results/compiled/')
for f in all_files:
    df = pd.read_csv('Results/compiled/'+f)
    df.to_excel(writer, sheet_name=os.path.basename(f))

writer.save()