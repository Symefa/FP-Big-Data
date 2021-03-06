from time import sleep
from json import dumps
from kafka import KafkaProducer
import os
import pandas as pd

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x:
                         dumps(x).encode('utf-8'))

# Load Dataset
dataset_folder_path = os.path.join(os.getcwd(), 'datasets')
dataset_file_path = os.path.join(dataset_folder_path, 'cbg_preprocessed.csv')
model = 3
limit = 5000
counter = 0
with open(dataset_file_path, "r", encoding="utf-8") as f:
    for row in f:
        if counter > model*limit:
            break
        producer.send('cbg-visitors', value=row)
        counter += 1
        print(row)
        sleep(0.0000001)
