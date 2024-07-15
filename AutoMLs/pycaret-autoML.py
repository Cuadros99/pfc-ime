from pycaret.classification import *
import pandas as pd
import os
import numpy as np

# Importação do dataset
file_path = os.path.join(os.getcwd(), '..', 'Datasets', 'CSE-CIC-IDS2018', 'raw','0_01-partial-dataset.csv')
data = pd.read_csv(file_path)

# Altera o tipo da columa Timestamp
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

# Troca os valores inf por nan, que por sua vez serão tratados pelo pycaret
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Setup the PyCaret environment
my_setup = setup(
                    data, 
                    target = 'Label',
                    normalize = True,
                    transformation= True,
                    create_date_columns = ['hour', 'minute', 'second', 'day', 'month'],
                    date_features = ['Timestamp']
                    )

# Extrai o dataset pré-processado
df = get_config('dataset_transformed')

# Treina os modelos de diferentes algoritmos
best_model = compare_models()

