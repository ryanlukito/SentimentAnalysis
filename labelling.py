import pandas as pd

class Labelling:
    def __init__(self, filename):
        self.filename = filename

    def labellingData(self):
        filename = 'REPLACE WITH DATASET PATH'

        data_hooligans = pd.read_csv(filename)
        data_hooligans_fulltext = data_hooligans['full_text']

        def label_data(text_series):
            labels = []

            for i, text in enumerate(text_series):
                print(f'Text {i+1}/{len(text_series)}:')
                print(text)
                label = input('Enter Label: ')
                labels.append(label)
                print('\n')

            labeled_data = pd.DataFrame({'text': text_series, 'label':labels})
            return labeled_data

        labeled_data = label_data(data_hooligans_fulltext)

        return labeled_data