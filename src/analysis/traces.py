import os
import pandas as pd


def load_traces(model_paths, model_names, params):
    training_traces ,val_traces = [], []

    for path in model_paths:
        for file in os.listdir(path):
            if file.endswith((".csv")):
                trace = pd.read_csv(
                    os.path.join(path, file) , header=None, 
                    names=['epoch', 'value'],
                    index_col = [0]
                )
            
                if file.startswith('val'):
                    val_traces.append(trace)
                else:
                    training_traces.append(trace)
                    
                    
    training_traces = pd.concat(
        training_traces, keys=model_names, names=params)
    val_traces = pd.concat(val_traces, keys=model_names, names=params)
    
    traces = pd.concat(
        [training_traces, val_traces], 
        keys=['training', 'validation'], names=['loss']
    )

    return traces
