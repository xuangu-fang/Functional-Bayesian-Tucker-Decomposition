"""
Implementation of Bayesian continues-mode Tensor, current is CP version, to be extended to Tucker 

Author: Shikai Fang
SLC, Utah, 2022.11
"""

import numpy as np
from numpy.lib import utils
import torch
import matplotlib.pyplot as plt
from model_LDS import LDS_GP_continues_mode
import os
import tqdm
import utils_continues
import bisect
import tensorly as tl

tl.set_backend("pytorch")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
JITTER = 1e-4

torch.manual_seed(332)


class Continues_Mode_Tensor_CP:

    def __init__(self, hyper_dict, data_dict):
        """-----------------hyper-paras---------------------"""
        self.device = hyper_dict["device"]
        self.R_U = hyper_dict["R_U"]  # rank of latent factor of embedding

        self.v = hyper_dict["v"]  # prior varience of embedding (scaler)
        self.a0 = hyper_dict["a0"]
        self.b0 = hyper_dict["b0"]
        self.DAMPING = hyper_dict["DAMPING"]
        self.DAMPING_tau = hyper_dict["DAMPING_tau"]
        """----------------data-dependent paras------------------"""

        self.ndims = data_dict["ndims"]
        self.nmods = len(self.ndims)

        self.tr_ind_DISCT = data_dict["tr_ind_DISCT"]
        self.tr_ind_CONTI = data_dict["tr_ind_CONTI"]

        self.tr_y = torch.tensor(data_dict["tr_y"]).to(self.device)
        # N*1
        self.N = len(self.tr_y)

        self.te_ind_DISCT = data_dict["te_ind_DISCT"]
        self.te_ind_CONTI = data_dict["te_ind_CONTI"]
        self.te_y = torch.tensor(data_dict["te_y"]).to(self.device)

        # utils attributes to compute the embeding of never-seen idx in test data
        self.never_seen_test_idx = data_dict["never_seen_test_idx"]
        self.CONTI_2_DISCT_dicts = data_dict["CONTI_2_DISCT_dicts"]
        self.DISCT_2_CONTI_dicts = data_dict["DISCT_2_CONTI_dicts"]

        LDS_continues_mode_paras = data_dict["LDS_continues_mode_paras"]
        """build dynamics (LDS-GP-CONTI class) for each modes (store using list)"""
        self.traj_class = []
        for mode in range(self.nmods):
            traj_class_mode = LDS_GP_continues_mode(
                LDS_continues_mode_paras[mode])
            self.traj_class.append(traj_class_mode)

        # posterior: store the most recently posterior from LDS
        self.post_U_m = [
            torch.rand(dim, self.R_U, 1).double().to(self.device)
            for dim in self.ndims
        ]  #  (dim, R_U, 1) * nmod
        self.post_U_v = [
            torch.eye(self.R_U).reshape(
                (1, self.R_U, self.R_U)).repeat(dim, 1,
                                                1).double().to(self.device)
            for dim in self.ndims
        ]  # (dim, R_U, R_U) * nmod

        # msg of U over all data-llk, use nature-paras, merge them will to be observation of LDS

        self.msg_U_lam = [
            1e-3 * torch.eye(self.R_U).reshape((1, self.R_U, self.R_U)).repeat(
                self.N, 1, 1).double().to(self.device)
            for i in range(self.nmods)
        ]  # (N*R_U*R_U)*nmod
        self.msg_U_eta = [
            1e-3 * torch.rand(self.N, self.R_U, 1).double().to(self.device)
            for i in range(self.nmods)
        ]  # (N*R_U*1)*nmod
        """llk-msg and post. of obs-noise"""

        # msg of tau
        self.msg_a = torch.ones(self.N, 1).double().to(self.device)  # N*1
        self.msg_b = torch.ones(self.N, 1).double().to(self.device)  # N*1

        self.E_tau = 1

        self.uid_table, self.data_table = utils_continues.build_id_key_table(
            nmod=self.nmods, ind=self.tr_ind_DISCT
        )  # nested-list of observed objects (and their associated entrie)
        # recall, idx in uid_table is sorted

        self.product_method = "hadamard"

        # some constant terms
        self.ones_const = torch.ones(self.N, 1).to(self.device)
        self.eye_const = torch.eye(self.R_U).to(self.device)

        # gamma in CP, we just set it as a all-one constant v
        self.post_gamma_m = torch.ones(self.R_U,
                                       1).double().to(self.device)  # (R^K)*1

    def product_with_gamma(self, E_z, E_z_2, mode):
        """product E_z / E_z_2 with gamma: for CP, gamma is constant all-one-vector, we actully do nothing here"""
        return E_z, E_z_2

    def msg_approx_U(self, mode):
        """approx the msg from the group of data-llk"""

        # for mode in range(self.nmods):

        condi_modes = [i for i in range(self.nmods)]
        condi_modes.remove(mode)

        E_z, E_z_2 = utils_continues.moment_product(
            modes=condi_modes,
            ind=self.tr_ind_DISCT,
            U_m=self.post_U_m,
            U_v=self.post_U_v,
            order="second",
            sum_2_scaler=False,
            device=self.device,
            product_method=self.product_method,
        )

        E_z, E_z_2 = self.product_with_gamma(E_z, E_z_2, mode)

        # use the nature-paras, easy to DAMPING
        S_inv = self.E_tau * E_z_2  # (N,R,R)
        S_inv_Beta = self.E_tau * E_z * self.tr_y.reshape(-1, 1, 1)  # (N,R,1)

        self.msg_U_lam[mode] = (self.DAMPING * self.msg_U_lam[mode] +
                                (1 - self.DAMPING) * S_inv)
        self.msg_U_eta[mode] = (self.DAMPING * self.msg_U_eta[mode] +
                                (1 - self.DAMPING) * S_inv_Beta)

    def msg_approx_tau(self):

        all_modes = [i for i in range(self.nmods)]

        E_z, E_z_2 = utils_continues.moment_product(
            modes=all_modes,
            ind=self.tr_ind_DISCT,
            U_m=self.post_U_m,
            U_v=self.post_U_v,
            order="second",
            sum_2_scaler=False,
            device=self.device,
            product_method=self.product_method,
        )

        self.msg_a = 1.5 * self.ones_const

        term1 = 0.5 * torch.square(self.tr_y)  # N*1

        term2 = self.tr_y.reshape(-1, 1) * torch.matmul(
            E_z.transpose(dim0=1, dim1=2), self.post_gamma_m).reshape(-1,
                                                                      1)  # N*1

        temp = torch.matmul(E_z_2, self.post_gamma_m)  # N*R*1
        term3 = 0.5 * torch.matmul(temp.transpose(dim0=1, dim1=2),
                                   self.post_gamma_m).reshape(-1, 1)  # N*1

        # alternative way to compute term3, where we have to compute and store E_gamma_2
        # term3 = torch.unsqueeze(0.5* torch.einsum('bii->b',torch.bmm(self.E_gamma_2,self.E_z_2)),dim=-1) # N*1

        self.msg_b = self.DAMPING_tau * self.msg_b + (1 - self.DAMPING_tau) * (
            term1.reshape(-1, 1) - term2.reshape(-1, 1) + term3.reshape(-1, 1)
        )  # N*1

    def LDS_update(self, mode):
        """merge the approx.msg as the observation, run filter and smooth"""
        LDS_traj = self.traj_class[mode]
        id_CONTI_list = LDS_traj.id_CONTI_list

        assert len(id_CONTI_list) == len(self.uid_table[mode])

        for i, CONTI_idx in enumerate(id_CONTI_list):

            DISCT_idx = self.CONTI_2_DISCT_dicts[mode][CONTI_idx]
            uid = self.uid_table[mode][i]  # id of embedding

            assert DISCT_idx == uid

            eid = self.data_table[mode][i]  # id of associated entries

            # merge the llk-msg (along with priors)
            U_V = torch.linalg.inv(
                self.msg_U_lam[mode][eid].sum(dim=0) +
                (1 / self.v) * torch.eye(self.R_U).to(self.device))  # (R,R)

            U_M = torch.mm(U_V, self.msg_U_eta[mode][eid].sum(dim=0))  # (R,1)

            LDS_traj.filter_predict(i)
            LDS_traj.filter_update(y=U_M, R=U_V)

        LDS_traj.smooth()

    def post_update_U(self, mode):
        """update the post.U based on the LDS-smoothed results"""
        # for mode in range(self.nmods):

        LDS_traj = self.traj_class[mode]
        id_DISCT_list = LDS_traj.id_DISCT_list

        H = LDS_traj.H

        U_m_smooth = torch.stack(
            [torch.mm(H, m) for m in LDS_traj.m_smooth_list], dim=0)  # (*,R,1)

        U_v_smooth = torch.stack(
            [torch.mm(torch.mm(H, P), H.T) for P in LDS_traj.P_smooth_list],
            dim=0)  # (*,R,R)

        self.post_U_m[mode][id_DISCT_list, :, :] = U_m_smooth
        self.post_U_v[mode][id_DISCT_list, :, :] = U_v_smooth

    def post_update_tau(self):
        """update post. factor of tau based on current msg. factors"""

        post_a = self.a0 + self.msg_a.sum() - self.N
        post_b = self.b0 + self.msg_b.sum()
        self.E_tau = post_a / post_b

    def post_merge_U(self):
        """get the post.U on never-seen idx for test by merging the smoothed states of the neighbro"""
        for mode in range(self.nmods):

            LDS_traj = self.traj_class[mode]

            for new_idx_DISCT in LDS_traj.never_seen_test_idx_DISCT:
                new_idx_CONTI = self.DISCT_2_CONTI_dicts[mode][new_idx_DISCT]
                merge_U_m, merge_U_v = LDS_traj.merge(new_idx_CONTI)

                self.post_U_m[mode][new_idx_DISCT, :, :] = merge_U_m
                self.post_U_v[mode][new_idx_DISCT, :, :] = merge_U_v

    def model_test(self, test_ind, test_y):

        MSE_loss = torch.nn.MSELoss()
        MAE_loss = torch.nn.L1Loss()

        loss_test = {}

        all_modes = [i for i in range(self.nmods)]

        pred = utils_continues.moment_product(
            modes=all_modes,
            ind=test_ind,
            U_m=self.post_U_m,
            U_v=self.post_U_v,
            order="first",
            sum_2_scaler=True,
            device=self.device,
            product_method=self.product_method,
        )

        loss_test["rmse"] = torch.sqrt(
            MSE_loss(pred.squeeze(),
                     test_y.squeeze().to(self.device)))
        loss_test["MAE"] = MAE_loss(pred.squeeze(),
                                    test_y.squeeze().to(self.device))

        return pred, loss_test

    def reset(self):
        for LDS_traj in self.traj_class:
            LDS_traj.reset_list()

    def post_update_U_CEP(self):
        # merge such msgs to get post.U

        for mode in range(self.nmods):
            for j in range(len(self.uid_table[mode])):

                uid = self.uid_table[mode][j]  # id of embedding
                eid = self.data_table[mode][j]  # id of associated entries

                self.post_U_v[mode][uid] = torch.linalg.inv(
                    self.msg_U_lam[mode][eid].sum(dim=0) + (1.0 / self.v) *
                    torch.eye(self.R_U).to(self.device))  # R_U * R_U
                self.post_U_m[mode][uid] = torch.mm(
                    self.post_U_v[mode][uid],
                    self.msg_U_eta[mode][eid].sum(dim=0))  # R_U *1


class Continues_Mode_Tensor_Tucker(Continues_Mode_Tensor_CP):

    def __init__(self, hyper_dict, data_dict):
        super().__init__(hyper_dict, data_dict)

        # posterior: init with rand on post_U in tucker seems better than randn
        self.post_U_m = [
            torch.rand(dim, self.R_U, 1).double().to(self.device)
            for dim in self.ndims
        ]  #  (dim, R_U, 1) * nmod

        self.DAMPING_gamma = hyper_dict["DAMPING_gamma"]

        self.product_method = "kronecker"

        self.nmod_list = [self.R_U for k in range(self.nmods)]
        """llk-msg and post. of vectorized Tucker-Core"""
        self.gamma_size = np.product([self.nmod_list])  # R_U^{K}

        self.msg_gamma_lam = 1e-3 * torch.eye(self.gamma_size).reshape(
            (1, self.gamma_size, self.gamma_size)).repeat(
                self.N, 1, 1).double().to(self.device)  # N*(R^K)*(R^K)
        self.msg_gamma_eta = 1e-3 * torch.rand(
            self.N, self.gamma_size, 1).double().to(self.device)  # N*(R^K)*1

        # post. of gamma
        self.post_gamma_m = (torch.rand(self.gamma_size,
                                        1).double().to(self.device))  # (R^K)*1
        self.post_gamma_v = (torch.eye(self.gamma_size).double().to(
            self.device))  # (R^K)*(R^K)

    def product_with_gamma(self, E_z, E_z_2, mode):
        """product E_z / E_z_2 with gamma: for tucker, gamma is the folded tucker core, we actully do tensor-matrix product here"""

        E_gamma_tensor = tl.tensor(self.post_gamma_m.reshape(
            self.nmod_list))  # (R^k *1)-> (R * R * R ...)
        E_gamma_mat_k = tl.unfold(E_gamma_tensor, mode).double()

        # some mid terms (to compute E_a_2 = gamma_fold * z\z\.T *  gamma_fold.T)

        term1 = torch.matmul(E_z_2, E_gamma_mat_k.T)  # N * R_U^{K-1} * R_U
        E_a_2 = torch.matmul(term1.transpose(dim0=1, dim1=2),
                             E_gamma_mat_k.T).transpose(
                                 dim0=1, dim1=2)  # N * R_U * R_U

        # to compute E_a = gamma_fold * z\
        E_a = torch.matmul(E_z.transpose(dim0=1, dim1=2),
                           E_gamma_mat_k.T).transpose(
                               dim0=1, dim1=2)  # num_eid * R_U * 1

        return E_a, E_a_2

    def msg_approx_gamma(self):

        all_modes = [i for i in range(self.nmods)]

        E_z, E_z_2 = utils_continues.moment_product(
            modes=all_modes,
            ind=self.tr_ind_DISCT,
            U_m=self.post_U_m,
            U_v=self.post_U_v,
            order="second",
            sum_2_scaler=False,
            device=self.device,
            product_method=self.product_method,
        )

        msg_gamma_lam_new = self.E_tau * E_z_2  # N*(R^K)*(R^K)

        msg_gamma_eta_new = self.E_tau * E_z * self.tr_y.reshape(
            -1, 1, 1)  # N*(R^K)*1

        self.msg_gamma_lam = (self.DAMPING_gamma * self.msg_gamma_lam +
                              (1 - self.DAMPING_gamma) * msg_gamma_lam_new
                              )  # N*(R^K)*(R^K)
        self.msg_gamma_eta = (self.DAMPING_gamma * self.msg_gamma_eta +
                              (1 - self.DAMPING_gamma) * msg_gamma_eta_new
                              )  # N*(R^K)*1

    def post_update_gamma(self):

        self.post_gamma_v = torch.linalg.inv(
            self.msg_gamma_lam.sum(dim=0))  # (R^K) * (R^K)

        self.post_gamma_m = torch.mm(
            self.post_gamma_v, self.msg_gamma_eta.sum(dim=0))  # (R^K) * 1

    def model_test(self, test_ind, test_y):

        MSE_loss = torch.nn.MSELoss()
        MAE_loss = torch.nn.L1Loss()

        loss_test = {}

        all_modes = [i for i in range(self.nmods)]

        E_z = utils_continues.moment_product(
            modes=all_modes,
            ind=test_ind,
            U_m=self.post_U_m,
            U_v=self.post_U_v,
            order="first",
            sum_2_scaler=False,
            device=self.device,
            product_method=self.product_method,
        )
        pred = torch.matmul(E_z.transpose(dim0=1, dim1=2),
                            self.post_gamma_m).squeeze()

        loss_test["rmse"] = torch.sqrt(
            MSE_loss(pred.squeeze(),
                     test_y.squeeze().to(self.device)))
        loss_test["MAE"] = MAE_loss(pred.squeeze(),
                                    test_y.squeeze().to(self.device))

        return pred, loss_test
