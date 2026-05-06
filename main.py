from pathlib import Path

from classification import WaterClassifier
from learning import learn

if __name__ == "__main__":

    # requiredFiles = ['water_classifier.pkl', 'water_labels.pkl']
    #
    # requiredFilesExistCheck = True
    #
    # for file in requiredFiles:
    #     if not Path(file).is_file():
    #         requiredFilesExistCheck = False
    #
    # if not requiredFilesExistCheck:
    #     learn()
    # else:
    #     print('Модель уже обучена')

    learn()

    water_classifier = WaterClassifier()
    res = water_classifier.predict('IMG_0877.JPG')
    print(res['position'])