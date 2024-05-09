import os
import shutil
from config import config


def make_experiment():
    experiment_path = config.output_path + config.experiment_name
    if os.path.exists(experiment_path) == False:
        os.mkdir(experiment_path)

    model_path = experiment_path + '/' + config.model_name
    if os.path.exists(model_path) == False:
        os.mkdir(model_path)
        weight = model_path + '/weights'
        os.mkdir(weight)
    print('[INFO] experiment directory has been created.')

    prediction_path = model_path + '/predictions'
    if os.path.exists(prediction_path) == False:
        os.mkdir(prediction_path)


