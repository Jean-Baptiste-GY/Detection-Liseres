import sys
import numpy as np
import os
import pathlib
import json
import tensorflow as tf


def json_and_model(model_name, generator, parameters, structure_parameter = 'filters_sequence', notes = 'Aucunes notes', folder_name=None):
    generator_name = generator.__name__

    
    if not folder_name:
        if structure_parameter:
            structure = '_'.join(['x'.join(str(i) for i in e) for e in parameters[structure_parameter]])
        else:
            structure=''
            adaptateur2h_min_bias
        folder_name = f'{datetime.now().strftime("%Y-%m-%d")}_{model_name}_{structure}'
    
    JSON = {
        'notes': notes,
        'name': model_name,
        'generator': generator_name,
        'generator_parameters': parameters,
        'structure': parameters[structure_parameter] if structure_parameter else None,
        'folder_name' : folder_name,
        'training' : []
    }


    return JSON, generator(**parameters)

def get_dataset_dictionary(dataset):
    return {
        'images' : list(dataset.labels.index),
        'labels' : list(dataset.labels)
    }

def compare_dataset(JSON, key, possible_dataset):

    if len(JSON['training']) == 0:
        return possible_dataset
    
    i = -1
    while JSON['training'][i]['datasets'][key] == 'SAME_DATASET':
        i -= 1
        if -i > len(JSON['training']): # Ne devrait jamais arriver !!!
            print("ATTENTION ! Le dataset n'est pas défini lors de la première phase d'entrainement !")
            return possible_dataset
            
    if possible_dataset == JSON['training'][i]['datasets'][key]:
        return 'SAME_DATASET'
    else:
        return possible_dataset
    weights = decoder.get_weights(n=decoder.layers[1].weights[0].shape[0])

def add_training(JSON, training_dataset, history, validation_dataset=None, augmentations = [], training_weights={0:1, 1:1}):
    training_info = {
        'epochs'  : len(history.epoch),
        'augmentations' : augmentations,
        'optimizer' : history.model.optimizer.get_config(),
        'weights' : training_weights,
        'datasets': {'training' : compare_dataset(JSON, 'training', get_dataset_dictionary(training_dataset))},
        'results' : {k : history.history[k][-1] for k in history.history.keys()},
        'history' : history.history
    }

    if validation_dataset is not None:
        training_info['datasets']['validation'] = compare_dataset(JSON, 'validation', get_dataset_dictionary(validation_dataset))
       

    training_info['optimizer']['learning_rate'] = float(training_info['optimizer']['learning_rate'])
    JSON['training'].append(training_info)

def train_model(JSON, model, training_dataset, validation_dataset, compile_params, fit_params, augmentations=[], save_path='./trained_models/'):
    compile_params['weighted_metrics'] = compile_params['metrics']
    model.compile(**compile_params)

    erreur = False
    
    try:
        model.fit(**fit_params)
    except KeyboardInterrupt:
        erreur = True
        print('Interruption !')
    finally:
        add_training(JSON=JSON,
                 training_dataset=training_dataset,
                 history=model.history,
                 validation_dataset=validation_dataset,
                 augmentations=augmentations,
                 training_weights= fit_params['class_weight'] if 'class_weight' in fit_params.keys() else fit_params['sample_weight'] if 'sample_weight' in fit_params.keys() else [],
                )

        save_json_and_model(
            JSON=JSON,
            model=model,
            path=save_path,
            generate_weights=True
        )

        if erreur:
            raise KeyboardInterrupt

def save_json_and_model(JSON, model, path, folder_name=None, generate_weights=False, model_suffix=None):


    folder_name = JSON["folder_name"]

    path = pathlib.Path(path)
    if not os.path.isdir(path/folder_name):
        os.mkdir(path/folder_name)

    if generate_weights:
        weight_path = path/folder_name/'weights'
        if os.path.isdir(weight_path):
            for e in os.listdir(weight_path):
                os.unlink(weight_path.joinpath(e))
        else:
            os.mkdir(weight_path)

        for i, w in enumerate(model.weights):
            np.save(weight_path.joinpath(f"weights_{i}"), w)

    if model_suffix:
        name = f'model_{model_suffix}.keras'
    else:
        name = 'model.keras'

    if os.path.isfile(path/folder_name/name):
        os.rename(path/folder_name/name, path/folder_name/'model_backup.keras')
    model.save(path/folder_name/name)
    
    save_json(JSON, path)

def save_json(JSON, path):
    path = pathlib.Path(path)
    with open(path/JSON['folder_name']/'model_info.json', 'w', encoding='utf-8') as f:
        json.dump(JSON, f, ensure_ascii=False, indent=4)  
    

def load_json_and_model(path, custom_objects):
    path = pathlib.Path(path)
    model, JSON = None, None
    if os.path.isdir(path):
        if os.path.isfile(path.joinpath('model.keras')):
            model_path = path.joinpath('model.keras')
            json_path = path.joinpath('model_info.json')
        elif os.path.isdir(path/'weights'):
           pass
    elif os.path.isfile(path):
        model_path = path
        json_path = path.parent.joinpath('model_info.json')

    with open(json_path, 'r') as file:
        JSON = json.load(file)
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    return JSON, model