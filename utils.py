import os
import json
from scoring_program.scoring_utils import execute_all

def read_json(path):
    with open(path) as f:
        file = json.load(f)
    return file

def write_json(path, dictionary):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w+') as f:
        json.dump(dictionary, f, indent=4)

def gather_and_save(config, trainer):
    # gather all predictions and save
    RESULT_DIR = f"./RESULTS/{config.logging.run_name}"
    DB_PATH = os.path.join('data', config.data.db_id, f'{config.data.db_id}.sqlite')
    
    predictions = {}
    for i in range(trainer.world_size):
        predictions.update(read_json(os.path.join(RESULT_DIR, f"predictions_{i}.json")))
    predictions = execute_all(predictions, db_path=DB_PATH, tag='pred')

    # iterate in predictions to find "error_pred" value, replace with "null"
    for k, v in predictions.items():
        if 'error_pred' in v:
            predictions[k] = "null"
    
    write_json(os.path.join(RESULT_DIR, "predictions.json"), predictions)
    os.system(f"cd {RESULT_DIR} && zip -r predictions.zip predictions.json")
