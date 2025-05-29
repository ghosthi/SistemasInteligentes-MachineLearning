import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time
from RandomForest import RandomForest, taxa_acertos, segmenta_dataset
from RandomForestParams import FIXED_PARAMS

def avalia_hiperparams(x_train, y_train, x_test, y_test, param_name, param_values, fixed_params):
    accuracies = []
    training_times = []
    
    for value in param_values:
        params = fixed_params.copy()
        params[param_name] = value
        
        if param_name == 'n_feature' and value is not None:
            if isinstance(value, str):
                if value == 'sqrt':
                    n_features = int(np.sqrt(x_train.shape[1]))
                elif value == 'log2':
                    n_features = int(np.log2(x_train.shape[1]))
            else:
                n_features = int(x_train.shape[1] * float(value))

            params['n_feature'] = n_features
        
        start_time = time.time()
        model = RandomForest(**params)
        model.fit(x_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(x_test)
        acc = taxa_acertos(y_test, y_pred)
        
        accuracies.append(acc)
        training_times.append(training_time)
    
    return accuracies, training_times

def plot_sensitivity(results, param_name, param_values, label, fig_num):
    plt.figure(num=fig_num, figsize=(12, 10))
    
    plt.subplot(1, 2, 1)
    param_values = [1 if v is None else v for v in param_values]

    plt.plot(param_values, results['accuracies'], 'o-', color='k')
    plt.title(f'Sensibilidade de {label}')
    plt.xlabel(label)
    plt.ylabel('Taxa de acertos')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(param_values, results['times'], 'o-', color='k')
    plt.title(f'Tempo de Treino vs {label}')
    plt.xlabel(label)
    plt.ylabel('Tempo (segundos)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'sensibilidade_{param_name}.png')

if __name__ == "__main__":
    data = np.genfromtxt('treino_sinais_vitais_com_label.txt', delimiter=',', skip_header=1)
    x = data[:, :-1]
    y = data[:, -1].astype(int)

    x_train, x_test, y_train, y_test = segmenta_dataset(x, y, test_size=0.2)

    param_studies = {
        'n_trees': {
            'values': [1, 3, 5, 10, 15, 20, 30, 60],
            'results': None,
            'nome_bonito': "Num. Árvores"
        },
        'max_depth': {
            'values': [3, 5, 8, 12, 15, 20, 25, 30],
            'results': None,
            'nome_bonito': "Profundidade Máxima"
        },
        'min_samples_split': {
            'values': [2, 3, 5, 8, 10, 15, 20, 30],
            'results': None,
            'nome_bonito': "Min. Amostras na segmentação"
        },
        'n_feature': {
            'values': ['sqrt', 'log2', 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, None],
            'results': None,
            'nome_bonito': "Num. Features"
        }
    }

    i = 0
    for param_name, study in param_studies.items():
        accuracies, times = avalia_hiperparams(
            x_train, y_train, x_test, y_test,
            param_name, study['values'], FIXED_PARAMS
        )
        
        study['results'] = {
            'accuracies': accuracies,
            'times': times
        }
        
        plot_sensitivity(study['results'], param_name, study['values'], study["nome_bonito"], i)
        i+=1

    plt.show()