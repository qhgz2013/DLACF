# SIGKDD 2019 Social Recommendation with OLA
# implementation detail:
# 1. use ALS optimization instead of SGD w.r.t. Eq. (22) - (24)
# 2. add balance and de-correlation constraint for U and V like DCF init
# 4. Eq. (23) ignored the related terms in term 2 of Eq. (21)
import numpy as np
import scipy.sparse as sp
from UpdateSVD import UpdateSVD
from rec_model import RecModel


class OLARec(RecModel):
    def __init__(self, rating_matrix, social_matrix, r, delta_phi, lc, gamma_phi, gamma_u, gamma_v, delta_u, delta_v,
                 min_r, max_r, debug=False, update_alg='als', lr=1e-3):
        assert update_alg in {'als', 'sgd'}, 'update_alg should be "als" or "sgd"'
        self.rating_matrix = rating_matrix.copy()
        self.rating_matrix.data = (self.rating_matrix.data - min_r) / (max_r - min_r)
        self.social_matrix = social_matrix
        self.r = r
        self.delta_phi = delta_phi
        self.n_users, self.n_items = self.rating_matrix.shape
        self.lc = lc
        self.gamma_phi = gamma_phi
        self.gamma_u = gamma_u
        self.gamma_v = gamma_v
        self.u = self.v = self.phi = self.alpha = self.x = self.y = self.z = None
        self.max_r = max_r
        self.min_r = min_r
        self.debug = debug
        self.delta_u = delta_u
        self.delta_v = delta_v
        self.update_alg = update_alg
        self.lr = lr  # only used when update_alg == 'sgd'

    # noinspection DuplicatedCode
    def _compute_alpha_iu_u_u(self):
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
        mean = np.nanmean(alpha_iu_u_u, 0)
        alpha_iu_u_u[nan_idx, :] = mean
        return alpha_iu_u_u

    # noinspection DuplicatedCode
    def _optimal_limited_attention(self):
        # for construction csr matrix
        ola_data, ola_indices, ola_indptr = [], [], [0]
        for i in range(self.n_users):
            us = self.social_matrix.indices[self.social_matrix.indptr[i]:self.social_matrix.indptr[i+1]]
            if len(us) == 0:
                ola_indptr.append(ola_indptr[-1])
                continue
            # Euclidean distance
            beta_i = self.lc * np.sqrt(np.sum((self.u[i, :].reshape(1, -1) - self.u[us, :]) ** 2, -1))
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
            ola_indices.extend(us[sorted_beta_i[:k]])
            ola_data.extend(alpha)
        # alpha matrix
        self.alpha = sp.csr_matrix((ola_data, ola_indices, ola_indptr),
                                   shape=self.social_matrix.shape, dtype=np.float32)
        return self.alpha

    # noinspection DuplicatedCode
    def loss(self):
        term1 = 0.0
        for i in range(self.n_users):
            phi_i = self.phi[i, :].reshape(-1, 1)
            v_j = self.v[self.rating_matrix.indices[self.rating_matrix.indptr[i]:self.rating_matrix.indptr[i+1]], :]
            dist = (self.rating_matrix.data[self.rating_matrix.indptr[i]:self.rating_matrix.indptr[i+1]]
                    - np.dot(v_j, phi_i)) ** 2
            term1 += np.sum(dist) / 2
        alpha_iu_u_u = self._compute_alpha_iu_u_u()
        term2 = self.delta_phi * np.sum((self.phi - alpha_iu_u_u) ** 2) / 2
        term3 = self.delta_phi * np.sum((self.u - self.phi) ** 2) / 2
        term4 = self.delta_u * np.sum(self.u ** 2) / 2
        term5 = self.delta_v * np.sum(self.v ** 2) / 2
        # balance and de-correlation term
        term6 = self.gamma_phi * np.sum(self.phi * self.x)
        term7 = self.gamma_u * np.sum(self.u * self.y)
        term8 = self.gamma_v * np.sum(self.v * self.z)
        loss = term1 + term2 + term3 + term4 + term5 - term6 - term7 - term8
        return loss, term1

    def init(self):
        self.phi = np.random.randn(self.n_users, self.r) * 0.1
        self.u = np.random.randn(self.n_users, self.r) * 0.1
        self.v = np.random.randn(self.n_items, self.r) * 0.1
        self.x = UpdateSVD(self.phi.T).T
        self.y = UpdateSVD(self.u.T).T
        self.z = UpdateSVD(self.v.T).T

    def _m_step_als(self, rating_matrix_t, it):
        alpha_iu_u_u = self._compute_alpha_iu_u_u()
        # phi-subproblem
        for i in range(self.n_users):
            # noinspection DuplicatedCode
            items = self.rating_matrix.indices[self.rating_matrix.indptr[i]:self.rating_matrix.indptr[i+1]]
            rating = self.rating_matrix.data[self.rating_matrix.indptr[i]:self.rating_matrix.indptr[i+1]]
            v_j = self.v[items, :]
            a = np.dot(v_j.T, v_j) + 2 * self.delta_phi * np.eye(self.r)
            b = np.sum(rating.reshape(-1, 1) * v_j, 0) + self.delta_phi * (alpha_iu_u_u[i, :] + self.u[i, :]) + \
                self.gamma_phi * self.x[i, :]
            self.phi[i, :] = np.linalg.solve(a, b)
        if self.debug:
            print('epoch %d, M-step loss phi-subproblem: %s' % (it, self.loss()))

        # u-subproblem
        alpha_trans = self.alpha.T.tocsr()  # for reverse lookup
        for t in range(self.n_users):
            # users = E(t)
            # noinspection DuplicatedCode
            users = alpha_trans.indices[alpha_trans.indptr[t]:alpha_trans.indptr[t+1]]
            if len(users) == 0:
                continue
            # alpha_{it}
            alpha_t = alpha_trans.data[alpha_trans.indptr[t]:alpha_trans.indptr[t+1]].reshape(-1, 1)
            denominator = self.delta_phi * (1 + np.sum(alpha_t ** 2)) + self.delta_u
            phi_i = self.phi[users, :]
            alpha_iu_u_u_excluded = alpha_iu_u_u[users, :] - alpha_t * self.u[users, :]
            numerator = self.delta_phi * (np.sum((phi_i - alpha_iu_u_u_excluded) * alpha_t, 0) + self.phi[t, :])
            numerator += self.gamma_u * self.y[t, :]
            self.u[t, :] = numerator / denominator
            # self.u[t, :] = self.phi[t, :] - self.delta_u / self.delta_phi * self.u[t, :]
        if self.debug:
            print('epoch %d, M-step loss u-subproblem: %s' % (it, self.loss()))

        # v-subproblem
        for j in range(self.n_items):
            users = rating_matrix_t.indices[rating_matrix_t.indptr[j]:rating_matrix_t.indptr[j+1]]
            ratings = rating_matrix_t.data[rating_matrix_t.indptr[j]:rating_matrix_t.indptr[j+1]]
            if len(users) == 0:
                continue
            phi_i = self.phi[users, :]
            a = np.dot(phi_i.T, phi_i) + self.delta_v * np.eye(self.r)
            b = np.sum(ratings.reshape(-1, 1) * phi_i, 0) + self.gamma_v * self.z[j, :]
            # self.v[j, :] = np.linalg.solve(a, b)
            self.v[j, :] = np.linalg.lstsq(a, b, rcond=None)[0]
        if self.debug:
            print('epoch %d, M-step loss v-subproblem: %s' % (it, self.loss()))

        self.x = UpdateSVD(self.phi.T).T
        self.y = UpdateSVD(self.u.T).T
        self.z = UpdateSVD(self.v.T).T
        if self.debug:
            print('epoch %d, M-step loss svd-update: %s' % (it, self.loss()))

    def _m_step_sgd(self, it):
        # balance and de-correlation is disabled for SGD, only implemented the original OLA-Rec
        alpha_iu_u_u = self._compute_alpha_iu_u_u()
        for i in range(self.n_users):
            for idx in range(self.rating_matrix.indptr[i], self.rating_matrix.indptr[i+1]):
                phi_i = self.phi[i, :]
                j = self.rating_matrix.indices[idx]
                r = self.rating_matrix.data[idx]
                v_j = self.v[j, :]
                term1 = np.sum(phi_i * v_j) - r
                # formula (22)
                grad_phi_i = term1 * v_j + self.delta_phi * ((phi_i - alpha_iu_u_u[i, :]) - (self.u[i, :] - phi_i))
                self.phi[i, :] -= self.lr * grad_phi_i
                # formula (23)
                grad_u_i = self.delta_phi * (self.u[i, :] - phi_i) + self.delta_u * self.u[i, :]
                self.u[i, :] -= self.lr * grad_u_i
                # formula (24)
                grad_v_j = term1 * phi_i + self.delta_v * self.v[j, :]
                self.v[j, :] -= self.lr * grad_v_j
        if self.debug:
            print('epoch %d, M-step loss: %s' % (it, self.loss()))

    def train(self, max_itr, _=None, reset_param=True):
        if reset_param:
            self.init()
        rating_matrix_t = self.rating_matrix.T.tocsr()
        for it in range(max_itr):
            # E-step: obtain social relation via OLA
            self._optimal_limited_attention()
            if self.debug:
                print('epoch %d, E-step loss: %s' % (it, self.loss()))

            # M-step: solve optimization via ALS
            if self.update_alg == 'als':
                self._m_step_als(rating_matrix_t, it)
            else:
                self._m_step_sgd(it)

    def predict(self, x):
        users, items = x
        score = np.sum(self.phi[users, :] * self.v[items, :], 1)
        return score * (self.max_r - self.min_r) + self.min_r

    def params(self):
        return self.phi, self.u, self.v


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
    np.random.seed(123)
    r_train, r_test, s_bin = get_train_test_data(Dataloader.load_data(data_name, user_filter))
    r_train.eliminate_zeros()
    r_test.eliminate_zeros()
    np.random.seed(123)
    model = OLARec(r_train, s_bin, 32, 1, 0.1, 1e-2, 10, 1e-5, 10, 1, min_r, max_r, True, 'sgd', 1e-3)
    model.train(5)
    metric = Metric(r_test, r_train)
    with metric:
        print(metric.roc_metric(model, 10))
        print(metric.NDCG(model, 10))
        print(metric.HRandMRR(model, 10))
        print(metric.MAEandRMSE(model))


if __name__ == '__main__':
    main()
