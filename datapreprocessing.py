import pandas as pd
import re

class DataPreprocessing:
    def __init__(self, filename):
        self.filename = filename

    def labellingData(self):
        data_hooligans = pd.read_csv(self.filename)
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
    
    def clean_text(self, col_name = 'full_text'):
        data_hooligans = pd.read_csv(self.filename)
        data_hooligans_fulltext = data_hooligans[col_name]
        def cleanUp(text):
            replacements = {
                r'\blg\b': 'lagi',
                r'\bbgt\b': 'sangat',
                r'\bsm\b': 'sama',
                r'\bntu\b': 'itu',
                r'\+\b': '',                  # Hapus tanda plus
                r'\bkyk\b': 'kayak',
                r'\bak\b': 'aku',
                r'\bato\b': 'atau',
                r'\bd\b': 'di',
                r'\bIndo\b': 'indonesia',
                r'\bjd\b': 'jadi',
                r'\bskrg\b': 'sekarang',
                r'\b\\b': 'atau',
                r'\bgak\b': 'tidak',
                r'\bga\b': 'tidak'
            }

            # Terapkan setiap penggantian dalam teks
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                
            # Hapus mention seperti @username
            text = re.sub(r'@\w+', '', text)
            
            # Hapus mention seperti #hastag
            text = re.sub(r'#\w+', '', text)
            
            # Hapus titik-titik berlebihan menjadi satu titik saja
            text = re.sub(r'\.{2,}', '.', text)
            text = re.sub(r'\.{2,}', '. ', text)
            
            # Ganti pola huruf-angka-huruf seperti 'ngalah2in' dengan 'ngalah ngalahin'
            text = re.sub(r'(\w+)(\d)(\w+)', lambda m: f"{m.group(1)} {m.group(1)}{m.group(3)}", text)
            
            # Ganti pola kata yang memiliki angka di tengah (misalnya "macam2" menjadi "macam macam")
            text = re.sub(r'(\w+)(\d)(\w*)', lambda m: f"{m.group(1)} {m.group(1)}{m.group(3)}", text)
            
            # misahin angka kata (contoh 20biji 20 biji)
            text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)
            
            # ngehapus link
            text = re.sub(r'https://t\.co/\S+', '', text)
            
            # Hapus spasi ekstra di awal/akhir teks
            return text.strip()
        
        return data_hooligans_fulltext.apply(cleanUp)
        