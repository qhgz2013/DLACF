import heapq
import numpy as np
import warnings

# change this if using K > 100

denominator_table = np.log2(np.arange(2, 102))


def deprecated(f):
    import functools
    import warnings
    warning_calls = set()

    @functools.wraps(f)
    def wrap_func(*args, **kwargs):
        if f not in warning_calls:
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(f'Calling method {f.__name__} is deprecated', DeprecationWarning)
            warnings.simplefilter('default', DeprecationWarning)
            warning_calls.add(f)
        return f(*args, **kwargs)
    # noinspection PyDeprecation
    return wrap_func


def sigmoid(x):
    """Sigmoid function"""
    return 1. / (1. + np.exp(-x))


def dcg_at_k(r, method=1):
    r = np.asfarray(r)
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            # return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            return np.sum(r / denominator_table[:r.shape[0]])
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def get_ndcg(r, k, method=1):
    """
    Score is normalized discounted cumulative gain (ndcg)
    	Relevance orignally was positive real values
    	Args:
    		r: Relevance scores (list or numpy) in rank order
    			(first element is the first item)
    		method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
    				If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    	Returns:
    		Normalized discounted cumulative gain
    """

    dcg_max = dcg_at_k(sorted(r, reverse=True)[:k], method)
    # dcg_min = dcg_at_k(sorted(r), method)
    # assert( dcg_max >= dcg_min )

    # if not dcg_max:
    #     return 0.

    dcg = dcg_at_k(r[:k], method)

    # print dcg_min, dcg, dcg_max
    # if dcg_max == dcg_min:
    #     return 1
    # else:
    #     return (dcg - dcg_min) / (dcg_max - dcg_min)
    return dcg / dcg_max  # dcg_max is idcg


def topk_sorted_idx(a, k):
    if a.size <= k:
        return np.argsort(-a)

    topk_index = np.argpartition(-a, k)[:k]
    topk_data = a[topk_index]
    topk_index_sort = np.argsort(-topk_data)
    topk_index_sort = topk_index[topk_index_sort]

    return topk_index_sort


class _NegItemIndexer:
    def __init__(self, n_items, pos_items):
        self.n_items = n_items
        self.pos_items = pos_items
        self.n_users = len(pos_items)
        self.items = np.arange(n_items, dtype=np.int32)

    def __getitem__(self, item):
        assert isinstance(item, int)
        assert 0 <= item < self.n_users
        return np.setdiff1d(self.items, self.pos_items[item], assume_unique=True)


class _MetricBase:
    def __init__(self, num_users, num_items, train_pos_items, test_pos_items, test_pos_ratings, maxR, minR):
        # self.R_test = R_test
        self.num_users = num_users
        self.num_items = num_items
        self.train_pos_items = train_pos_items
        self.test_pos_items = test_pos_items
        self.test_pos_ratings = test_pos_ratings
        self.test_items = _NegItemIndexer(num_items, train_pos_items)
        pos_items = [np.concatenate([train_pos_items[u], test_pos_items[u]]) for u in range(num_users)]
        self.test_neg_items = _NegItemIndexer(num_items, pos_items)  # changed: generated on the fly
        self.maxR = maxR
        self.minR = minR

    def roc_metric(self, model, k):
        pass

    def HRandMRR(self, model, k):
        pass

    def Recall(self, model, k):
        pass

    def Precision(self, model, k):
        pass

    def F1(self, model, k):
        pass

    def FPRAndTPR(self, model, k):
        pass

    def NDCG(self, model, k):
        pass

    def MAEandRMSE(self, model):
        pass


class _MetricImpl(_MetricBase):
    def __init__(self, num_users, num_items, train_pos_items, test_pos_items, test_pos_ratings, maxR, minR):
        super(_MetricImpl, self).__init__(num_users, num_items, train_pos_items, test_pos_items, test_pos_ratings,
                                          maxR, minR)
        # for speed up computation (and use more memory)
        self._cache_prediction = False
        self._cached_predictions = None

    def __enter__(self):
        self._cache_prediction = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cache_prediction = False
        self._cached_predictions = None

    def _predict_all(self, model):
        arr = np.empty((self.num_users, self.num_items), dtype=np.float32)
        for u in range(self.num_users):
            arr[u, :] = model.predict([np.full(self.num_items, u, dtype=np.int32),
                                       np.arange(self.num_items, dtype=np.int32)])
        return arr

    def _model_predict(self, model, x):
        if self._cache_prediction:
            if self._cached_predictions is None:
                self._cached_predictions = self._predict_all(model)
            return self._cached_predictions[tuple(x)]
        return model.predict(x)

    def roc_metric(self, model, k, threshold=0.5, policy='man'):
        policy = policy.lower()
        if policy == 'mau':
            return self._roc_metric_missing_as_unknown(model, k, threshold)
        elif policy == 'man':
            return self._roc_metric_missing_as_negative(model, k, threshold)
        else:
            raise ValueError('Invalid evaluation policy, it should be one of MAU (Missing As Unknown) or MAN'
                             ' (Missing As Negative)')

    # changed: treats top-k items as positive, no matter what rating it predicted (i.e., even when RecSys predicts a
    # zero for this item, it will be regarded as a positive item once it appears in top-k list)
    # todo: test evaluation code
    def _roc_metric_missing_as_negative(self, model, k, threshold=0.5):
        tp = np.empty(self.num_users, dtype=np.int32)
        p = np.empty_like(tp)
        mask = np.ones_like(tp, dtype=np.bool)
        rating_threshold = self.minR + (self.maxR - self.minR) * threshold
        for u in range(self.num_users):
            if len(self.test_pos_items[u]) == 0:
                mask[u] = 0
                continue
            u_test_items = self.test_items[u]
            users = np.full(len(u_test_items), u, dtype=np.int32)
            pred_ratings = self._model_predict(model, [users, u_test_items])
            top_k_idx = topk_sorted_idx(pred_ratings, k)
            top_k_items = u_test_items[top_k_idx]
            test_pos_items = self.test_pos_items[u]
            test_pos_items = test_pos_items[self.test_pos_ratings[u] >= rating_threshold]
            top_k_items_gt_label = np.isin(top_k_items, test_pos_items, assume_unique=True)
            tp[u] = np.sum(top_k_items_gt_label)
            p[u] = len(test_pos_items)
        tp, p = tp[mask], p[mask]
        fn = p - tp
        n = self.num_items - p
        fp = k - tp
        tn = n - fp
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered')
            pre = np.nanmean(tp / (tp + fp))
            rec = np.nanmean(tp / p)
            fpr = np.nanmean(fp / n)
            acc = np.nanmean((tp + tn) / (p + n))
        sum_tp, sum_fp, sum_fn = np.sum(tp), np.sum(fp), np.sum(fn)
        micro_rec = sum_tp / (sum_tp + sum_fn)
        return {
            'precision_macro': pre, 'precision_micro': pre, 'precision': pre,
            'recall_macro': rec, 'tpr': rec,  # tpr == recall (by definition)
            'recall_micro': micro_rec, 'fpr': fpr, 'accuracy': acc,
            'f1_macro': 0.0 if pre + rec == 0 else 2 * pre * rec / (pre + rec),
            'f1_micro': 0.0 if pre + micro_rec == 0 else 2 * pre * micro_rec / (pre + micro_rec)
        }

    def _roc_metric_missing_as_unknown(self, model, k, threshold=0.5):
        tp = np.empty(self.num_users, dtype=np.int32)
        fp = np.empty_like(tp)
        p = np.empty_like(tp)  # tp + fn = p
        n = np.empty_like(tp)
        mask = np.ones_like(tp, dtype=np.bool)
        rating_threshold = self.minR + (self.maxR - self.minR) * threshold
        for u in range(self.num_users):
            if len(self.test_pos_items[u]) == 0:
                mask[u] = 0
                continue
            u_test_items = self.test_pos_items[u]
            users = np.full(len(u_test_items), u, dtype=np.int32)
            pred_ratings = self._model_predict(model, [users, u_test_items])
            top_k_idx = topk_sorted_idx(pred_ratings, k)
            test_items_gt_label = self.test_pos_ratings[u] >= rating_threshold
            tp[u] = np.sum(test_items_gt_label[top_k_idx])
            # fp[u] = np.sum(np.logical_and(top_k_items_pred_label, np.logical_not(top_k_items_gt_label)))
            fp[u] = len(top_k_idx) - tp[u]  # this may not equal to k when len(u_test_items) < k
            p[u] = np.sum(test_items_gt_label)
            n[u] = len(u_test_items) - p[u]
        tp, fp, p, n = tp[mask], fp[mask], p[mask], n[mask]
        fn = p - tp
        tn = n - fp
        # tn = self.num_items - fp - fn
        # n = self.num_items - p
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered')
            pre = np.nanmean(tp / (tp + fp))
            rec = np.nanmean(tp / p)
            fpr = np.nanmean(fp / n)
            acc = np.nanmean((tp + tn) / (p + n))
        sum_tp, sum_fp, sum_fn = np.sum(tp), np.sum(fp), np.sum(fn)
        micro_pre = sum_tp / (sum_tp + sum_fp)
        micro_rec = sum_tp / (sum_tp + sum_fn)
        return {
            'precision_macro': pre, 'precision_micro': micro_pre,  # 'precision': pre,
            'recall_macro': rec, 'tpr': rec,  # tpr == recall (by definition)
            'recall_micro': micro_rec, 'fpr': fpr, 'accuracy': acc,
            'f1_macro': 2 * pre * rec / (pre + rec), 'f1_micro': 2 * pre * micro_rec / (pre + micro_rec)
        }

    def HRandMRR(self, model, k, threshold=0.5):
        hr = mrr = count = 0
        rating_threshold = self.minR + (self.maxR - self.minR) * threshold
        for u in range(self.num_users):
            if len(self.test_pos_items[u]) == 0:
                continue
            u_test_items = self.test_items[u]
            users = np.full(len(u_test_items), u, dtype=np.int32)
            pred_ratings = self._model_predict(model, [users, self.test_items[u]])
            top_k_idx = topk_sorted_idx(pred_ratings, k)
            top_k_items = self.test_items[u][top_k_idx]
            test_pos_items = self.test_pos_items[u]
            test_pos_items = test_pos_items[self.test_pos_ratings[u] >= rating_threshold]  # filter low rating items
            gt_label = np.isin(top_k_items, test_pos_items)
            first_index = np.argwhere(gt_label).reshape(-1)
            if len(first_index) > 0:
                hr += 1
                mrr += 1 / (1 + first_index[0])
            count += 1
        return hr / count, mrr / count

    @deprecated
    def Recall(self, model, k):
        recall_sum = 0
        count = 0
        for u in range(self.num_users):
            u_test_pos_items = self.test_pos_items[u]
            u_test_items = self.test_items[u]

            if u_test_pos_items.size == 0:
                continue
            else:
                count += 1

            users = np.full(len(u_test_items), u, dtype='int32')

            pred_ratings = self._model_predict(model, [users, u_test_items])

            # Evaluate top rank list
            ranklist = u_test_items[topk_sorted_idx(pred_ratings, k=k)]
            tp = np.intersect1d(u_test_pos_items, ranklist).size
            recall_sum += tp / u_test_pos_items.size

        return recall_sum / count

    @deprecated
    def Precision(self, model, k):
        count = 0
        precision_sum = 0
        for u in range(self.num_users):
            u_test_pos_items = self.test_pos_items[u]
            u_test_items = self.test_items[u]

            if u_test_pos_items.size == 0:
                continue
            else:
                count += 1

            users = np.full(len(u_test_items), u, dtype='int32')
            pred_ratings = self._model_predict(model, [users, u_test_items])

            # Evaluate top rank list
            ranklist = u_test_items[topk_sorted_idx(pred_ratings, k=k)]
            tp = np.intersect1d(u_test_pos_items, ranklist).size

            precision_sum += tp / k

        return precision_sum / count

    @deprecated
    def F1(self, model, k):
        p = self.Precision(model, k)
        r = self.Recall(model, k)
        return 2 * p * r / (p + r)

    @deprecated
    def FPRAndTPR(self, model, k):
        TPR_sum = 0
        FPR_sum = 0
        count = 0
        for u in (range(self.num_users)):
            u_test_pos_items = self.test_pos_items[u]
            u_test_neg_items = self.test_neg_items[u]
            u_test_items = self.test_items[u]

            if u_test_pos_items.size == 0:
                continue
            else:
                count += 1

            users = np.full(len(u_test_items), u, dtype='int32')
            pred_ratings = self._model_predict(model, [users, u_test_items])

            map_item_rating = {}
            for i in range(len(u_test_items)):
                item = u_test_items[i]
                map_item_rating[item] = pred_ratings[i]

            # Evaluate top rank list
            ranklist = heapq.nlargest(k, map_item_rating, key=map_item_rating.get)

            tp = np.intersect1d(u_test_pos_items, ranklist).size
            fp = len(ranklist) - tp
            TPR_sum += tp / u_test_pos_items.size
            FPR_sum += fp / u_test_neg_items.size

        FPR = FPR_sum / count
        TPR = TPR_sum / count
        return FPR, TPR

    def NDCG(self, model, k):
        ndcgs = []
        for u in range(self.num_users):
            u_test_pos_items = self.test_pos_items[u]

            if u_test_pos_items.size < k:
                continue

            users = np.full(len(u_test_pos_items), u, dtype='int32')
            pred_ratings = self._model_predict(model, [users, u_test_pos_items])
            gt_ratings = self.test_pos_ratings[u]

            ranked_ratings = gt_ratings[topk_sorted_idx(pred_ratings, k=pred_ratings.size)]
            ndcgs.append(get_ndcg(ranked_ratings, k))

        return np.mean(np.array(ndcgs))

    def MAEandRMSE(self, model):
        mae = rmse = count = 0
        for u in range(self.num_users):
            u_test_pos_items = self.test_pos_items[u]
            if len(u_test_pos_items) == 0:
                continue
            users = np.full(len(u_test_pos_items), u, dtype='int32')
            pred_ratings = self._model_predict(model, [users, u_test_pos_items])
            gt_ratings = self.test_pos_ratings[u]
            mae += np.sum(np.abs(pred_ratings - gt_ratings))
            rmse += np.sum((pred_ratings - gt_ratings) ** 2)
            count += len(u_test_pos_items)
        return mae / count, np.sqrt(rmse / count)

    @deprecated
    def dump_shared_memory(self, filename: str = None):
        import struct
        import os
        import sys
        import mmap
        assert sys.platform == 'linux', 'current implementation only support linux environment'
        os.makedirs('/dev/shm/shared_memory_dlacf', exist_ok=True)
        shm_file = f'/dev/shm/shared_memory_dlacf/{os.getpid() if filename is None else filename}'
        # file_no = os.open(shm_file, os.O_RDWR | os.O_CREAT | os.O_TRUNC)
        file = open(shm_file, 'w+b')
        file_no = file.fileno()

        def _compute_bytes_needed_ndarray(arr: np.ndarray) -> int:
            return 8 + arr.ndim * 8 + 2 + len(bytes(arr.dtype.name, 'utf8')) + arr.nbytes

        # compute bytes
        n_bytes_needed = 16
        for u in range(self.num_users):
            n_bytes_needed += _compute_bytes_needed_ndarray(self.test_pos_items[u])
            n_bytes_needed += _compute_bytes_needed_ndarray(self.test_pos_ratings[u])
            n_bytes_needed += _compute_bytes_needed_ndarray(self.train_pos_items[u])
        n_bytes_needed += 16  # maxR, minR
        file.truncate(n_bytes_needed)
        mmap_file = mmap.mmap(file_no, n_bytes_needed, prot=mmap.PROT_WRITE)
        buf = memoryview(mmap_file)

        def _dump_data(array: np.ndarray):
            nonlocal offset
            length = _compute_bytes_needed_ndarray(array)
            buf[offset:offset+8] = struct.pack('L', length)
            offset += 8
            buf[offset:offset+1] = struct.pack('B', array.ndim)
            offset += 1
            buf[offset:offset+(8*array.ndim)] = struct.pack('L' * array.ndim, *array.shape)
            offset += 8 * array.ndim
            dtype_str = bytes(array.dtype.name, 'utf8')
            buf[offset:offset+1] = struct.pack('B', len(dtype_str))
            offset += 1
            buf[offset:offset+len(dtype_str)] = dtype_str
            offset += len(dtype_str)
            np_arr = np.ndarray(array.shape, array.dtype, buffer=buf[offset:offset+array.nbytes])
            np_arr[:] = array[:]
            offset += array.nbytes
        # dump data
        buf[0:16] = struct.pack('LL', self.num_users, self.num_items)
        offset = 16
        for u in range(self.num_users):
            _dump_data(self.test_pos_items[u])
            _dump_data(self.test_pos_ratings[u])
            _dump_data(self.train_pos_items[u])
        buf[offset:offset+16] = struct.pack('dd', self.maxR, self.minR)
        buf.release()
        mmap_file.close()
        file.close()
        return shm_file


class Metric(_MetricImpl):
    def __init__(self, R_test, R_train):
        num_users, num_items = R_test.shape
        train_pos_items = [R_train[u, :].indices for u in range(num_users)]
        test_pos_items = [R_test[u, :].indices for u in range(num_users)]
        test_pos_ratings = [R_test[u, test_pos_items[u]].toarray().flatten() for u in range(num_users)]
        maxR = np.maximum(np.max(R_test.data), np.max(R_train.data))
        minR = np.minimum(np.min(R_test.data), np.min(R_train.data))
        super(Metric, self).__init__(num_users, num_items, train_pos_items, test_pos_items, test_pos_ratings, maxR,
                                     minR)


class MetricSM(_MetricImpl):
    def __init__(self, sm_name: str):
        import struct
        import mmap
        file = open(sm_name, 'rb')
        mmap_file = mmap.mmap(file.fileno(), 0, prot=mmap.PROT_READ)
        buf = memoryview(mmap_file)
        num_users, num_items = struct.unpack('LL', buf[0:16])
        offset = 16

        def _restore_data():
            nonlocal offset
            length = struct.unpack('L', buf[offset:offset+8])[0]
            ndim = struct.unpack('B', buf[offset+8:offset+9])[0]
            tmp_offset = 9
            shape = struct.unpack('L' * ndim, buf[offset+tmp_offset:offset+tmp_offset+(8 * ndim)])
            tmp_offset += 8 * ndim
            dtype_len = struct.unpack('B', buf[offset+tmp_offset:offset+tmp_offset+1])[0]
            tmp_offset += 1
            dtype_str = str(bytes(buf[offset+tmp_offset:offset+tmp_offset+dtype_len]), 'utf8')
            tmp_offset += dtype_len
            nbytes = length - tmp_offset
            arr = np.ndarray(shape, np.dtype(dtype_str), buf[offset+tmp_offset:offset+tmp_offset+nbytes])
            offset += length
            return arr

        test_pos_items = []
        test_pos_ratings = []
        train_pos_items = []

        for u in range(num_users):
            test_pos_items.append(_restore_data())
            test_pos_ratings.append(_restore_data())
            train_pos_items.append(_restore_data())
        max_r, min_r = struct.unpack('dd', buf[offset:offset+16])
        super(MetricSM, self).__init__(num_users, num_items, train_pos_items, test_pos_items, test_pos_ratings,
                                       max_r, min_r)
        self.memoryview_buffer = buf
        self.mmap_file = mmap_file
        self.shared_memory_file = file

    def close(self):
        self.test_pos_ratings = self.test_neg_items = self.test_pos_items = self.test_items = None
        self.memoryview_buffer.release()
        self.mmap_file.close()
        self.shared_memory_file.close()
