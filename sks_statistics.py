import numpy as np
import pickle


if __name__ == '__main__':
    path = './evaluation/sks/{type}_results.pkl'
    types = ['qr', 'dwt_qr', 'dft_qr']

    # Load the data
    for type in types:
        print('Loading {}...'.format(type))
        with open(path.format(type=type), 'rb') as f:
            results = pickle.load(f)
            for field in results:
                mean = np.mean(results[field])
                std = np.std(results[field])
                median = np.median(results[field])
                print('{}: mean = {}, std = {}, median = {}'.format(field, mean, std, median))
        print('='*80)
