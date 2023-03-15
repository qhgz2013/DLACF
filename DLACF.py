import numpy as np
import scipy.sparse as sp
from UpdateSVD import UpdateSVD
from rec_model import RecModel


class DLACF(RecModel):
    def __init__(self, rating_matrix, social_matrix, r, lc, delta_phi, gamma_phi, gamma_u, gamma_v, min_r, max_r,
                 debug=False, pretrain_init=False, init_delta_u=None, init_delta_v=None):
        self.rating_matrix = rating_matrix.copy()
        # scale from [min_r, max_r] to [-r, r]
        self.rating_matrix.data = ((self.rating_matrix.data - min_r) / (max_r - min_r) * 2 - 1) * r
        self.social_matrix = social_matrix
        self.r = r
        self.lc = lc
        self.delta_phi = delta_phi
        self.gamma_phi = gamma_phi
        self.gamma_u = gamma_u
        self.gamma_v = gamma_v
        self.n_users, self.n_items = self.rating_matrix.shape
        self.phi = self.u = self.v = self.x = self.y = self.z = None
        self.debug = debug
        self.alpha = None
        self.min_r = min_r
        self.max_r = max_r
        self.pretrain_init = pretrain_init
        self.init_delta_u = init_delta_u
        self.init_delta_v = init_delta_v
        if pretrain_init:
            assert init_delta_u is not None and init_delta_v is not None, 'init_delta_* must be specified for pretrain'
        # debug field for tracking the ratio of limited attention, in format (k, n, alpha)
        self.la_tracking = []

    # noinspection DuplicatedCode
    def _compute_alpha_iu_u_u(self, return_nan_index=False, fill_nan=True):
        alpha_iu_u_u = np.full((self.n_users, self.r), np.nan, dtype=np.float32)
        nan_idx = []
        for i in range(self.n_users):
            if self.alpha.indptr[i] == self.alpha.indptr[i+1]:
                nan_idx.append(i)
                continue
            us = self.alpha.indices[self.alpha.indptr[i]:self.alpha.indptr[i+1]]
            alpha = self.alpha.data[self.alpha.indptr[i]:self.alpha.indptr[i+1]]
            u_u = self.u[us, :]
            value = np.sum(alpha.reshape(-1, 1) * u_u, 0)
            alpha_iu_u_u[i, :] = value
        # like OLA-DSR, fill empty value as mean value
        if fill_nan:
            mean = np.nanmean(alpha_iu_u_u, 0)
            alpha_iu_u_u[nan_idx, :] = mean
        if return_nan_index:
            return alpha_iu_u_u, nan_idx
        return alpha_iu_u_u

    # noinspection DuplicatedCode
    def _optimal_limited_attention(self):
        # for construction csr matrix
        ola_data, ola_indices, ola_indptr = [], [], [0]
        self.la_tracking.clear()
        for i in range(self.n_users):
            us = self.social_matrix.indices[self.social_matrix.indptr[i]:self.social_matrix.indptr[i+1]]
            if len(us) == 0:
                ola_indptr.append(ola_indptr[-1])
                continue
            # Hamming distance
            beta_i = self.lc * np.sum(np.abs(self.u[i, :].reshape(1, -1) - self.u[us, :]) / 2, -1)
            # beta_iu in asc order
            sorted_beta_i = np.argsort(beta_i)
            beta_i = beta_i[sorted_beta_i]
            us = us[sorted_beta_i]
            lambda_k = beta_i[0] + 1
            lambdas = []
            k = 0
            while k < len(sorted_beta_i) and lambda_k > beta_i[k]:
                k += 1
                term1 = np.sum(beta_i[:k])
                term2_inner = k + term1 ** 2 - k * np.sum(beta_i[:k] ** 2)
                lambda_k = (term1 + np.sqrt(term2_inner)) / k
                lambdas.append(lambda_k)
            alpha = np.array(lambdas) - beta_i[:k]
            alpha /= np.sum(np.maximum(alpha, 0))
            ola_indptr.append(ola_indptr[-1] + k)
            ola_indices.extend(us[:k])
            ola_data.extend(alpha)
            self.la_tracking.append((k, len(us), alpha))
        # alpha matrix
        self.alpha = sp.csr_matrix((ola_data, ola_indices, ola_indptr),
                                   shape=self.social_matrix.shape, dtype=np.float32)
        return self.alpha

    def init(self):
        self.phi = np.sign(np.random.randn(self.n_users, self.r))
        self.u = np.sign(np.random.randn(self.n_users, self.r))
        self.v = np.sign(np.random.randn(self.n_items, self.r))
        self.phi[self.phi == 0] = -1
        self.v[self.v == 0] = -1
        self.u[self.u == 0] = -1
        self.x = UpdateSVD(self.phi.T).T
        self.y = UpdateSVD(self.u.T).T
        self.z = UpdateSVD(self.v.T).T

    # noinspection DuplicatedCode
    def loss(self):
        term1 = 0.0
        # mae = rmse = 0
        for i in range(self.n_users):
            phi_i = self.phi[i, :].reshape(-1, 1)
            v_j = self.v[self.rating_matrix.indices[self.rating_matrix.indptr[i]:self.rating_matrix.indptr[i + 1]], :]
            mae_unscaled = self.rating_matrix.data[self.rating_matrix.indptr[i]:self.rating_matrix.indptr[i + 1]] \
                - np.dot(v_j, phi_i).reshape(-1)
            # mae_scaled = mae_unscaled / 2 / self.r * (self.max_r - self.min_r)
            # mae += np.sum(np.abs(mae_scaled))
            # rmse += np.sum(mae_scaled ** 2)
            term1 += np.sum(mae_unscaled ** 2) / 2
        alpha_iu_u_u = self._compute_alpha_iu_u_u()
        term2 = self.delta_phi * np.sum((self.phi - alpha_iu_u_u) ** 2) / 2
        term3 = self.delta_phi * np.sum((self.u - self.phi) ** 2) / 2
        term6 = self.gamma_u * np.sum(self.u * self.y)
        term7 = self.gamma_v * np.sum(self.v * self.z)
        term8 = self.gamma_phi * np.sum(self.phi * self.x)
        loss = term1 + term2 + term3 - term6 - term7 - term8
        return loss, term1  # , mae / len(self.rating_matrix.data), np.sqrt(rmse / len(self.rating_matrix.data))

    def _dcd_update_phi_v(self, update_var, another_var, rating_matrix, linear_term, max_itr):
        n = update_var.shape[0]
        for i in range(n):
            ratings = rating_matrix.data[rating_matrix.indptr[i]:rating_matrix.indptr[i+1]].reshape(-1, 1)
            relevant_idx = rating_matrix.indices[rating_matrix.indptr[i]:rating_matrix.indptr[i+1]]
            another_var_rel = another_var[relevant_idx, :]
            for it in range(max_itr):
                # BEGIN OPT2 - tracking updated terms without re-computing all values
                tracking_term = np.sum(update_var[i, :].reshape(1, -1) * another_var_rel, -1, keepdims=True)
                quad_term = ratings - tracking_term
                # END OPT2
                original = update_var[i, :].copy()
                for k in range(self.r):
                    # BEGIN OPT1 - re-compute all values
                    # quad_term = ratings - np.sum(update_var[i, :].reshape(1, -1) * another_var_rel, -1, keepdims=True)
                    # END OPT1
                    hat_value = np.sum(quad_term.reshape(-1) * another_var_rel[:, k]) + \
                        update_var[i, k] * len(ratings) + linear_term[i, k]
                    hat_value = np.sign(hat_value)
                    if hat_value != 0 and update_var[i, k] != hat_value:
                        update_var[i, k] = hat_value
                        # BEGIN OPT2
                        tracking_term += 2 * hat_value * another_var_rel[:, k].reshape(-1, 1)
                        quad_term = ratings - tracking_term
                        # END OPT2
                if np.all(np.equal(original, update_var[i, :])):
                    break
        return update_var

    def _dcd_update_u(self, max_itr):
        alpha_iu_u_u = self._compute_alpha_iu_u_u(fill_nan=False)  # [n_users, r]
        alpha_trans = self.alpha.T.tocsr()
        for t in range(self.n_users):
            # noinspection DuplicatedCode
            users = alpha_trans.indices[alpha_trans.indptr[t]:alpha_trans.indptr[t+1]]
            if len(users) == 0:
                continue
            # assert all(map(lambda x: x not in nan_idx, users))
            alpha_t = alpha_trans.data[alpha_trans.indptr[t]:alpha_trans.indptr[t+1]].reshape(-1, 1)
            alpha_iu_u_u_excluded = alpha_iu_u_u[users, :] - alpha_t * self.u[users, :]
            for it in range(max_itr):
                # vector-wise update is used here
                # hat_value = self.delta_phi * np.sum(alpha_t * (self.phi[users] - alpha_iu_u_u_excluded), 0) + \
                #     self.delta_phi * self.phi[t, :] + self.gamma_u * self.y[t, :]
                u_old = self.u[t, :].copy()
                # self.u[t, :] = np.where(np.equal(hat_value, 0), self.u[t, :], np.sign(hat_value))
                for k in range(self.r):
                    hat_value = self.delta_phi * np.sum(alpha_t * (self.phi[users, k] - alpha_iu_u_u_excluded[:, k])) +\
                                self.delta_phi * self.phi[t, k] + self.gamma_u * self.y[t, k]
                    hat_value = np.sign(hat_value)
                    if hat_value != 0 and hat_value != self.u[t, k]:
                        self.u[t, k] = hat_value
                        # should alpha_iu_u_u be updated?
                        # alpha_iu_u_u = self._compute_alpha_iu_u_u(fill_nan=False)
                        alpha_iu_u_u[users, k] += 2 * hat_value * alpha_t.reshape(-1)
                        # just alpha_iu_u_u[:, k] changed after updated
                        alpha_iu_u_u_excluded += 2 * hat_value * alpha_t  # = alpha_iu_u_u[users, :] - alpha_t * self.u[users, :]
                if np.all(np.equal(u_old, self.u[t, :])):
                    break

    # noinspection DuplicatedCode
    def train(self, max_itr, max_itr2, reset_param=True, init_only=False):
        if reset_param:
            if self.pretrain_init:
                # init by OLARec
                from OLARec import OLARec
                olarec = OLARec(rating_matrix=self.rating_matrix, social_matrix=self.social_matrix, r=self.r,
                                delta_phi=self.delta_phi, lc=self.lc, gamma_phi=self.gamma_phi, gamma_u=self.gamma_u,
                                gamma_v=self.gamma_v, delta_u=self.init_delta_u, delta_v=self.init_delta_v,
                                min_r=self.min_r, max_r=self.max_r, debug=self.debug)
                olarec.train(max_itr, reset_param=reset_param)
                self.phi, self.u, self.v = np.sign(olarec.phi), np.sign(olarec.u), np.sign(olarec.v)
                self.phi[self.phi == 0] = -1
                self.u[self.u == 0] = -1
                self.v[self.v == 0] = -1
                self.x = UpdateSVD(self.phi.T).T
                self.y = UpdateSVD(self.u.T).T
                self.z = UpdateSVD(self.v.T).T
            else:
                # random init
                self.init()
            if init_only:
                return  # just run initialization
        rating_matrix_t = self.rating_matrix.T.tocsr()
        for it in range(max_itr):
            phi0, u0, v0 = self.phi.copy(), self.u.copy(), self.v.copy()
            # E-step
            self._optimal_limited_attention()
            if self.debug:
                print('epoch %d, E-step loss: %s' % (it, self.loss()))

            # M-step
            alpha_iu_u_u = self._compute_alpha_iu_u_u()
            # DCD-phi
            self._dcd_update_phi_v(self.phi, self.v, self.rating_matrix,
                                   self.delta_phi*(alpha_iu_u_u+self.u)+self.gamma_phi*self.x, max_itr2)
            if self.debug:
                print('epoch %d, M-step loss phi-subproblem: %s' % (it, self.loss()))
            # DCD-V
            self._dcd_update_phi_v(self.v, self.phi, rating_matrix_t, self.gamma_v*self.z, max_itr2)
            if self.debug:
                print('epoch %d, M-step loss v-subproblem: %s' % (it, self.loss()))
            # DCD-U
            self._dcd_update_u(max_itr2)
            if self.debug:
                print('epoch %d, M-step loss u-subproblem: %s' % (it, self.loss()))
            self.x = UpdateSVD(self.phi.T).T
            self.y = UpdateSVD(self.u.T).T
            self.z = UpdateSVD(self.v.T).T
            if self.debug:
                print('epoch %d, M-step loss svd: %s' % (it, self.loss()))
            if np.all(np.equal(self.phi, phi0)) and np.all(np.equal(self.u, u0)) and np.all(np.equal(self.v, v0)):
                break

    def params(self):
        return self.phi, self.u, self.v

    def predict(self, x):
        users, items = x
        score = np.sum(self.phi[users, :] * self.v[items, :], 1)
        score = (score + self.r) / 2 / self.r  # [-r, r] -> [0, 1]
        return score * (self.max_r - self.min_r) + self.min_r  # [0, 1] -> [min_r, max_r]


# noinspection DuplicatedCode
def main():
    from Dataloader import Dataloader, get_train_test_data
    from evaluator import Metric
    data_name = 'filmtrust'
    user_filter = 10

    if data_name.endswith('filmtrust'):
        max_r = 4.0
        min_r = 0.5
    else:
        max_r = 5.0
        min_r = 1.0
    np.random.seed(123)  # dataset seed
    r_train, r_test, s_bin = get_train_test_data(Dataloader.load_data(data_name, user_filter))
    model = DLACF(r_train, s_bin, 32, min_r=min_r, max_r=max_r, lc=1e5, delta_phi=1, gamma_phi=0.1, gamma_u=0.1,
                  gamma_v=1000, debug=True, pretrain_init=True, init_delta_u=0.1, init_delta_v=1e6)  # filmtrust
    model.train(10, 10, reset_param=True, init_only=True)
    metric = Metric(r_test, r_train)
    with metric:
        print(metric.roc_metric(model, 10))
        print(metric.NDCG(model, 10))
        print(metric.HRandMRR(model, 10))
        print(metric.MAEandRMSE(model))
    for i in range(10):
        print(f'Itr {i}')
        model.train(1, 10, reset_param=False)
        with metric:
            print(metric.roc_metric(model, 10))
            print(metric.NDCG(model, 10))
            print(metric.HRandMRR(model, 10))
            print(metric.MAEandRMSE(model))


if __name__ == '__main__':
    main()
