import numpy as np
from pathlib import Path


def process_to_txt(modes_BASE, target_pollute, dims_BASE):

    ndim_str = "x".join([str(i) for i in dims_BASE])

    # dict_name = '../for_matlab/beijing_' + target_pollute + '/' + ndim_str + '/'
    dict_name = '../for_matlab/agg_beijing_' + target_pollute + '/' + ndim_str + '/'


    Path(dict_name).mkdir(parents=True, exist_ok=True)

    # load_name = '../beijing/' + '_'.join(
    #     modes_BASE
    # ) + '_' + target_pollute + '/' + 'DISCT_' + ndim_str + '_no_agg' + '.npy'

    load_name = '../beijing/' + '_'.join(
        modes_BASE
    ) + '_' + target_pollute + '/' + 'DISCT_' + ndim_str + '.npy'

    dataset_name = 'beijing_' + target_pollute

    full_data = np.load(load_name, allow_pickle=True).item()

    for fold in range(5):

        data_dict = full_data['data'][fold]
        data_dict['ndims'] = full_data['ndims']

        train_data = np.concatenate(
            [data_dict['tr_ind'], data_dict['tr_y'].reshape(-1, 1)], 1)
        test_data = np.concatenate(
            [data_dict['te_ind'], data_dict['te_y'].reshape(-1, 1)], 1)

        fmt = ['%d' for i in range(len(data_dict['ndims']))] + ['%.3f']

        file_train = dict_name + dataset_name + "_train_" + str(fold) + '.txt'
        file_test = dict_name + dataset_name + "_test_" + str(fold) + '.txt'

        np.savetxt(file_train, train_data, fmt=fmt, delimiter=' ')
        np.savetxt(file_test, test_data, fmt=fmt, delimiter=' ')

    ndims_info = 'dataset: %s, ndims: %s' % (dataset_name,
                                             str(data_dict['ndims']))

    f = open(dict_name + 'ndims.txt', "w+")
    f.write(ndims_info)
    f.write(
        "\n  base-0 indexing, space-separated, last-column is entry values, last-mode is DDT-mode (use for static baseline, drop for streaming baselines), all entries ordered by DDT-mode (last mode)"
    )


modes_BASE = ['TEMP', 'PRES', 'time']
target_pollute_list = ['PM2.5', 'PM10', 'SO2']
dims_BASE_list = [[50, 50, 150], [100, 100, 300], [300, 300, 1000],
                  [428, 501, 1461]]

for target_pollute in target_pollute_list:
    for dims_BASE in dims_BASE_list:
        process_to_txt(modes_BASE, target_pollute, dims_BASE)
