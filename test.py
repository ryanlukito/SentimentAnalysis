from datapreprocessing import DataPreprocessing

teguh_barata_aji = DataPreprocessing(filename='data_ryan.csv')


labeling = teguh_barata_aji.labellingData()
labeling.to_csv('label.csv', index=False)

# data_cleaned = teguh_barata_aji.clean_text('label.csv')

# print(data_cleaned)