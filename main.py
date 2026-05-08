import json
from pathlib import Path

from PIL import Image

from classification import WaterClassifier
from learning import learn
import pika
import sys
import os
import base64
import io
from db import changeTaskStatus

def main():

    requiredFiles = ['water_classifier.pkl', 'water_labels.pkl']

    requiredFilesExistCheck = True

    for file in requiredFiles:
        if not Path(file).is_file():
            requiredFilesExistCheck = False

    if not requiredFilesExistCheck:
        learn()
    else:
        print('Модель уже обучена')

    water_classifier = WaterClassifier()

    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    channel.queue_declare(queue='water_classification')

    def callback(ch, method, properties, body):

        imgJSONString = body.decode('utf-8')
        imgJSON = json.loads(imgJSONString)
        imgAnotherByteArray = bytearray(imgJSON['data']['file']['buffer']['data'])
        anotherImg = Image.open(io.BytesIO(imgAnotherByteArray))

        res = water_classifier.predict(anotherImg)
        changeTaskStatus(imgJSON['data']['taskId'], res['position'])

    channel.basic_consume(queue='water_classification', on_message_callback=callback, auto_ack=True)

    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)