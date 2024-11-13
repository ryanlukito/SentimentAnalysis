from datapreprocessing import DataPreprocessing
import pandas as pd

print(pd.read_csv("label.csv")['label'].value_counts())
#teguh_barata_aji = DataPreprocessing(filename='data_sean.csv')


#labeling = teguh_barata_aji.labellingData()
#labeling.to_csv('label.csv', index=False)

#data_cleaned = teguh_barata_aji.clean_text('label.csv')

#print(data_cleaned)