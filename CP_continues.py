import numpy as np
import torch
from model_continues_mode_tensor import Continues_Mode_Tensor_CP
import utils_continues
import tqdm
import yaml
import time

args = utils_continues.parse_args_continues_tensor()

torch.random.manual_seed(args.seed)

# assert args.dataset in {
#     'beijing_PM25', 'beijing_PM10', 'beijing_NO2', 'others'
# }

args.method = "CP"
print('dataset: ', args.dataset, ' rank: ', args.R_U)

config_path = "./config/config_" + args.dataset + "_" + args.method + ".yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

data_path = config["data_path"]

# prepare hyper_dict and data_dict

hyper_dict = utils_continues.make_hyper_dict(config, args)

EPOCH = hyper_dict["epoch"]
test_rmse = []
test_MAE = []

result_dict = {}
start_time = time.time()

for fold_id in range(args.num_fold):

    data_dict = utils_continues.make_data_dict(hyper_dict, data_path, fold_id)

    model = Continues_Mode_Tensor_CP(hyper_dict, data_dict)

    for epoch in tqdm.tqdm(range(EPOCH)):

        # reset LDS
        model.reset()

        # approx the msg from data-llk by standard CEP
        # LDS update:filter + smooth

        if hyper_dict['CEP_UPDATE_INNNER_MODE'] == True:
            for mode in range(model.nmods):
                model.msg_approx_U(mode)

                # LDS update:filter + smooth
                model.LDS_update(mode)
                # posterior update based on LDS-result
                model.post_update_U(mode)

        else:
            for mode in range(model.nmods):
                model.msg_approx_U(mode)

                # LDS update:filter + smooth
            for mode in range(model.nmods):
                # LDS update:filter + smooth
                model.LDS_update(mode)
                # posterior update based on LDS-result
                model.post_update_U(mode)

        model.msg_approx_tau()
        model.post_update_tau()
        # print('tau:',model.E_tau)

        if hyper_dict["EVALU_EPOCH"] > 0:

            if epoch % hyper_dict["EVALU_EPOCH"] == 0:

                pred, train_result = model.model_test(model.tr_ind_DISCT,
                                                      model.tr_y)

                model.post_merge_U()

                pred, test_result = model.model_test(model.te_ind_DISCT,
                                                     model.te_y)

                print("epoch:", epoch,
                      "train-rmse:%.3f" % train_result['rmse'],
                      "test-rmse::%.3f" % test_result['rmse'])

    # get the post.U on never-seen idx for test
    model.post_merge_U()

    pred, test_result = model.model_test(model.te_ind_DISCT, model.te_y)
    test_MAE.append(test_result['MAE'].cpu().numpy().squeeze())
    test_rmse.append(test_result['rmse'].cpu().numpy().squeeze())

    print('fold:', fold_id, ",test-error:", test_result)

rmse_array = np.array(test_rmse)
MAE_array = np.array(test_MAE)
result_dict['time'] = time.time() - start_time
result_dict['rmse_avg'] = rmse_array.mean()
result_dict['rmse_std'] = rmse_array.std()
result_dict['MAE_avg'] = MAE_array.mean()
result_dict['MAE_std'] = MAE_array.std()

utils_continues.make_log(args, hyper_dict, result_dict)
