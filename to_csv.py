import pandas as pd
import os, csv

rows = []
for file in os.listdir(r'C:\Users\Usuario\Desktop\crowd results\1.good'):
    if file.endswith('.png'):
        # print(file)
        rows.append({'file': file, 'category': 'good'})

for file in os.listdir(r'C:\Users\Usuario\Desktop\crowd results\3.unknown'):
    if file.endswith('.png'):
        # print(file)
        rows.append({'file': file, 'category': 'unknown'})

for file in os.listdir(r'C:\Users\Usuario\Desktop\crowd results\2.error'):
    if file.endswith('.png'):
        # print(file)
        rows.append({'file': file, 'category': 'bad'})

df = pd.DataFrame(rows)

df.to_csv('airways_classified.csv', index=False, quoting=csv.QUOTE_ALL)
print(df)


