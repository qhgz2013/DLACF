import multiprocessing as mp
import multiprocessing.pool as pool
from evaluator import Metric
import numpy as np
from Dataloader import Dataloader, get_train_test_data, CVDataloader
from functools import partial
import os
import json
import argparse
import typing
from collections import defaultdict
import warnings


def run_metric(dataset, model, k, evaluate_policy='man', q=None):
    train_set, test_set = dataset
    metric = Metric(test_set, train_set)
    with metric:
        roc = metric.roc_metric(model, k=k, policy=evaluate_policy)
        ndcg = metric.NDCG(model, k=k)
        hr, mrr = metric.HRandMRR(model, k=k)
        mae, rmse = metric.MAEandRMSE(model)
    result = {'pre': roc['precision_macro'], 'rec': roc['recall_macro'], 'fpr': roc['fpr'], 'acc': roc['accuracy'],
              'f1_macro': roc['f1_macro'], 'f1_micro': roc['f1_micro'], 'ndcg': ndcg, 'mae': mae, 'rmse': rmse,
              'hr': hr, 'mrr': mrr, 'rec_micro': roc['recall_micro']}
    if q is None:
        return result
    q.put(result)


def run_model(args):
    import threading
    model, dataset, seed, best_metric, evaluate_policy, verbose, max_itr, max_itr2, k = args
    np.random.seed(seed)  # fix seed before param initialization
    metric_proc = None
    result_queue = mp.SimpleQueue()
    if best_metric:
        results = []
        old_params = None
        for i in range(max_itr):
            exc = None
            try:
                if verbose:
                    model.train(1, max_itr2, i == 0)
                else:
                    # if verbose is set to false, RuntimeWarning is also suppressed
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        model.train(1, max_itr2, i == 0)
            except Exception as e:
                exc = e
            finally:
                if metric_proc is not None:
                    metric_proc.join()
                    results.append(result_queue.get())
                    metric_proc = None
            if verbose:
                print(f'    PID {os.getpid()} - TID {threading.current_thread().ident} - ITR {i} '
                      f'{"finished" if exc is None else "failed"}')
            if exc is not None:
                # exception while training, abort training and return nan metric value for this run
                break
            metric_proc = mp.Process(target=run_metric, args=(dataset, model, k, evaluate_policy, result_queue))
            metric_proc.start()
            if old_params is not None:
                converged = True
                for old_param, new_param in zip(old_params, model.params()):
                    if np.any(old_param != new_param):
                        converged = False
                        break
                if converged:
                    break
            old_params = [param.copy() for param in model.params()]
        if metric_proc is not None:
            metric_proc.join()
            results.append(result_queue.get())
        result_dict = defaultdict(list)
        for result in results:
            for metric_name in result:
                result_dict[metric_name].append(result[metric_name])
        del result_queue
        return dict(result_dict)
    else:
        exc = None
        try:
            if verbose:
                model.train(max_itr, max_itr2)
            else:
                # if verbose is set to false, RuntimeWarning is also suppressed
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    model.train(max_itr, max_itr2)
        except Exception as e:
            exc = e
        if verbose:
            print(f'    PID {os.getpid()} - TID {threading.current_thread().ident} - ITR {max_itr} '
                  f'{"finished" if exc is None else "failed"}')
        if exc is not None:
            return {k: np.nan for k in ['pre', 'rec', 'fpr', 'acc', 'f1_micro', 'f1_macro', 'ndcg', 'mae', 'rmse', 'hr',
                                        'mrr', 'rec_micro']}
        metric_proc = mp.Process(target=run_metric, args=(dataset, model, k, evaluate_policy, result_queue))
        metric_proc.start()
        metric_proc.join()
        result = result_queue.get()
        del result_queue
        return result


def rankmin(x):
    # >>> rankmin(np.array((1,1,1,2,2,3,3,3,3)))
    # array([0, 0, 0, 3, 3, 5, 5, 5, 5])
    x = np.asarray(x)
    nan_idx = np.isnan(x)
    nan_cnt = np.sum(nan_idx)
    if nan_cnt == 0:
        u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    else:
        u, inv_non_nan, counts = np.unique(x[~nan_idx], return_inverse=True, return_counts=True)
        # add nan values at the end as maximum value
        nan_inv_idx = len(u)
        inv = np.full(len(x), nan_inv_idx, dtype=inv_non_nan.dtype)
        inv[~nan_idx] = inv_non_nan
        counts = np.concatenate([counts, [nan_cnt]])
    csum = np.zeros_like(counts)
    csum[1:] = counts[:-1].cumsum()
    return csum[inv]


def _load_dataset_and_dump_shm(splits, q, pid):
    for i, (r_train, r_test) in enumerate(splits):
        metric = Metric(r_test, r_train)
        q.put(metric.dump_shared_memory(f'{pid}_{i}'))


# wrap daemon so that process created from pool can still fork child process
class NonDaemonProcess(mp.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, _):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NestablePool(pool.Pool):
    Process = NonDaemonProcess


def import_rec_model_from_str(model_str):
    import importlib
    module_name, class_name = model_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_seed', type=int, default=123)
    parser.add_argument('--param_seed', type=int, default=123)
    parser.add_argument('--alg', type=str, choices=['DCF', 'DSR', 'OLARec', 'DLACF', 'SoRec', 'DLACF+Init', 'DLACFInit',
                                                    'DSoRec', 'DTMF'], default='DLACF+Init')
    parser.add_argument('--r', type=int, default=8)
    parser.add_argument('--k', type=int, default=10)  # metric parameter
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['CiaoDVD', 'filmtrust'], default='filmtrust')
    parser.add_argument('--max_process', type=int, default=20)
    parser.add_argument('--user_filter', type=int, default=10)
    parser.add_argument('--max_epoch', type=int, default=5)  # for SoRec, it refers to max_iter
    parser.add_argument('--max_dcd_itr', type=int, default=10)
    parser.add_argument('--tune_itr', type=int, default=3)
    parser.add_argument('--rank_func', type=str, choices=['reciprocal', 'log2'], default='reciprocal')
    parser.add_argument('--best_metric', default=False, action='store_true')
    parser.add_argument('--disable_cache', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--evaluate_policy', choices=['mau', 'man'], default='man')  # missing as unknown / negative
    parser.add_argument('--n_folds', type=int, default=5)  # for cross validation, n_folds<=1 to disable
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # other user-defined params
    tune_range = np.power(10.0, np.arange(-5, 6))
    tune_range2 = 1 - np.power(10, np.arange(-4, -1, dtype=np.float))
    tune_range3 = np.power(10.0, np.arange(-2, 9))  # added for CiaoDVD, we still use tune_range when training filmtrust
    # what metric should be taken into consideration, all available metrics are list as follows:
    # pre, pre_micro, rec, rec_micro, fpr, acc, f1_macro, f1_micro, ndcg, mae, rmse, hr, mrr
    rank_metric = ['pre', 'rec', 'fpr', 'acc', 'f1_macro', 'f1_micro', 'ndcg', 'mae', 'rmse', 'hr', 'mrr']

    use_cache = not args.disable_cache
    np.random.seed(args.dataset_seed)
    alg = args.alg
    r = args.r
    max_process = args.max_process if args.max_process != 0 else mp.cpu_count()
    max_process = min(max_process, len(tune_range) * max(1, args.n_folds))
    # noinspection DuplicatedCode
    data_name = args.dataset
    user_filter = args.user_filter
    max_itr = args.max_epoch
    max_itr2 = args.max_dcd_itr
    tune_itr = args.tune_itr
    rank_func_name = args.rank_func
    param_init_seed = args.param_seed
    best_metric = args.best_metric
    verbose = args.verbose

    alg_tune_params = {
        'DCF': {'alpha': tune_range, 'beta': tune_range},
        'DSR': {'alpha0': tune_range, 'beta1': tune_range, 'beta2': tune_range, 'beta3': tune_range},
        'DLACFInit': {'delta_phi': tune_range, 'gamma_u': tune_range, 'gamma_v': tune_range, 'lc': tune_range,
                      'delta_u': tune_range, 'delta_v': tune_range3, 'gamma_phi': tune_range},
        'DLACF': {'delta_phi': tune_range, 'gamma_phi': tune_range, 'gamma_u': tune_range, 'gamma_v': tune_range,
                  'lc': tune_range},
        'DLACF+Init': {'init_delta_u': tune_range, 'init_delta_v': tune_range3, 'delta_phi': tune_range,
                       'gamma_phi': tune_range, 'gamma_u': tune_range, 'gamma_v': tune_range, 'lc': tune_range},
        'SoRec': {'learning_rate': tune_range, 'lambda_c': tune_range, 'lambda_reg': tune_range, 'gamma': tune_range2},
        'OLARec': {'lr': tune_range, 'delta_phi': tune_range, 'delta_u': tune_range, 'delta_v': tune_range3,
                   'lc': tune_range},
        'DSoRec': {'learning_rate': tune_range, 'lambda_c': tune_range, 'lambda_reg': tune_range, 'gamma': tune_range2},
        'DTMF': {'lambda_': tune_range, 'alpha': tune_range, 'beta': tune_range},
    }  # type: typing.Dict[str, typing.Dict[str, typing.Any]]

    if data_name == 'filmtrust':
        max_r = 4.0
        min_r = 0.5
    else:
        max_r = 5.0
        min_r = 1.0

    n_folds = args.n_folds
    if n_folds <= 1:
        n_folds = 1

    # <<<<< load data
    if n_folds == 1:
        split = Dataloader.load_data(data_name, user_filter)
        r_train_dataset = [split.train_set]
        r_train, r_test, s_bin = get_train_test_data(split)
        dataset = [(r_train, r_test)]
    else:
        split = CVDataloader.load_data(data_name, user_filter, args.n_folds)
        r_train_dataset = []
        dataset = []
        s_bin = None
        for i in range(n_folds):
            r_train, r_test, s_bin = get_train_test_data(split, folds=i)
            r_train_dataset.append(split.train_set)  # get_train_test_data must be called first to obtain "train_set"
            dataset.append((r_train, r_test))

    # <<<<< setting up algorithm params
    alg_init_params = {
        'DCF': {'r': r, 'maxR': max_r, 'minR': min_r},
        'DSR': {'r': r, 'maxR': max_r, 'minR': min_r},
        'DLACFInit': {'r': r, 'min_r': min_r, 'max_r': max_r, 'debug': False},
        'DLACF': {'r': r, 'min_r': min_r, 'max_r': max_r, 'debug': False},
        'DLACF+Init': {'r': r, 'min_r': min_r, 'max_r': max_r, 'debug': False, 'pretrain_init': True},
        'SoRec': {'k': r},
        'OLARec': {'r': r, 'min_r': min_r, 'max_r': max_r, 'debug': False, 'gamma_phi': 0.0, 'gamma_u': 0.0,
                   'gamma_v': 0.0, 'update_alg': 'sgd'},
        'DSoRec': {'k': r},
        'DTMF': {'r': r, 'min_r': min_r, 'max_r': max_r, 'debug': False},
    }
    alg_cv_init_params = {
        'DCF': {'rating_matrix': 'R', 'social_matrix': None},
        'DSR': {'rating_matrix': 'R', 'social_matrix': 'S'},
        'SoRec': {'cornac_train_set': 'train_set'},
        'DSoRec': {'cornac_train_set': 'train_set'},
    }
    alg_class_mapper = {'DCF': 'DCF.DCF', 'DSR': 'DSR.DSR', 'DLACFInit': 'OLARec.OLARec', 'DLACF': 'DLACF.DLACF',
                        'DLACF+Init': 'DLACF.DLACF', 'SoRec': 'SoRec.MySoRec', 'OLARec': 'OLARec.OLARec',
                        'DSoRec': 'SoRec_2stage.MyDSoRec', 'DTMF': 'DTMF_D.DTMF_D'}
    rank_func_mapper = {'reciprocal': lambda x: 1. / (x + 1), 'log2': lambda x: 1. / np.log2(x + 1)}
    rank_func = rank_func_mapper[rank_func_name]
    rank_metric_sorting_kind = {'max': ['pre', 'pre_micro', 'rec', 'rec_micro', 'acc', 'f1_micro', 'f1_macro', 'ndcg',
                                        'hr', 'mrr'],
                                'min': ['fpr', 'mae', 'rmse']}
    sort_func_mapper = {'max': lambda x: rankmin(-np.array(x, dtype=np.float)), 'min': rankmin}
    sort_func = {metric_name: sort_func_mapper[sort_kind]
                 for sort_kind, metric_names in rank_metric_sorting_kind.items() for metric_name in metric_names}

    params = alg_tune_params[alg]
    best_guessed_params = {k: v[0] for k, v in params.items()}
    best_guessed_params_metric = None

    # <<<<< setting up evaluation metric cache
    k = args.k
    # for back compatibility, use "cache" instead of "cache_10" when k = 10
    cache_dir = args.cache_dir or ('cache' if k == 10 else f'cache_{k}')
    print('cache_dir:', cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f'{cache_dir}/{alg}_{data_name}_{r}_{args.dataset_seed}_{args.param_seed}' \
                 f'{"_mau" if args.evaluate_policy == "mau" else ""}{"_best" if best_metric else ""}.json'
    if use_cache and os.path.isfile(cache_file):
        with open(cache_file, 'r', encoding='utf8') as f:
            cached_param_metrics = json.load(f)
    else:
        cached_param_metrics = {}

    def _get_param_str(param):
        return ';'.join(map(lambda x: f'{x[0]}={x[1]}', param.items()))

    # <<<<< main search loop
    with NestablePool(max_process) as p:
        for itr in range(tune_itr):
            print(f'Tune iteration {itr}')
            param_changed = False
            for param_name, param_values in params.items():
                print(f'  Tuning param {param_name}')  # delta, gamma, ...
                alg_instances = []
                results = [None] * len(param_values)  # type: typing.List[typing.Optional[dict]]
                alg_instance_index_params = defaultdict(list)
                # create model instances
                for i, param_value in enumerate(param_values):
                    run_params = best_guessed_params.copy()
                    run_params[param_name] = param_value
                    run_param_str = _get_param_str(run_params)
                    if run_param_str in cached_param_metrics:
                        results[i] = cached_param_metrics[run_param_str]
                        continue
                    # cross validation
                    for fold in range(n_folds):
                        rating_param_name = alg_cv_init_params.get(alg, {}).get('rating_matrix', 'rating_matrix')
                        social_param_name = alg_cv_init_params.get(alg, {}).get('social_matrix', 'social_matrix')
                        cornac_data_param_name = alg_cv_init_params.get(alg, {}).get('cornac_train_set', None)
                        run_fold_params = run_params.copy()
                        if cornac_data_param_name is not None:
                            run_fold_params[cornac_data_param_name] = r_train_dataset[fold]
                        else:
                            if rating_param_name is not None:
                                run_fold_params[rating_param_name] = dataset[fold][0]
                            if social_param_name is not None:
                                run_fold_params[social_param_name] = s_bin
                        model = import_rec_model_from_str(alg_class_mapper[alg])
                        alg_instance = partial(model, **alg_init_params[alg])(**run_fold_params)
                        # run param => (run index, fold, param index)
                        alg_instance_index_params[run_param_str].append((len(alg_instances), fold, i))
                        alg_instances.append((alg_instance, fold))
                # dispatch model training job to worker process, and obtain the results
                new_results = p.map(run_model,
                                    map(lambda x: (x[0], dataset[x[1]], param_init_seed, best_metric,
                                                   args.evaluate_policy, verbose, max_itr, max_itr2, k),
                                        alg_instances), chunksize=1)
                # merge cross validation results
                for run_param_str, run_info in alg_instance_index_params.items():
                    run_results = []
                    for run_index, fold, i in run_info:
                        run_results.append(new_results[run_index])
                    # merge results from different folds
                    merged_results = {}
                    for metric_name in rank_metric:
                        metric_values = []
                        for result in run_results:
                            metric_value_or_values = result[metric_name]
                            if isinstance(metric_value_or_values, list):
                                if len(metric_value_or_values) == 0:
                                    # ignore empty result
                                    continue
                                elif len(metric_value_or_values) == 1:
                                    metric_value_or_values = metric_value_or_values[0]
                                else:
                                    # rank best first
                                    metric_value_sorted_idx = sort_func[metric_name](metric_value_or_values)
                                    metric_value_or_values = metric_value_or_values[np.argmin(metric_value_sorted_idx)]
                            metric_values.append(metric_value_or_values)
                        if len(metric_values) == 0:
                            raise ValueError(f'No result for metric {metric_name}')
                        merged_results[metric_name] = np.mean(metric_values)
                    results[i] = merged_results
                    cached_param_metrics[run_param_str] = merged_results
                # rank the results from different hyperparameters
                result_rank = np.zeros(len(param_values), dtype=np.float)
                with open(cache_file, 'w', encoding='utf8') as f:
                    json.dump(cached_param_metrics, f, separators=(',', ':'))  # save temporary result
                for metric_name in rank_metric:
                    metric_values = [result[metric_name] for result in results]
                    metric_sorted_idx = sort_func[metric_name](metric_values)
                    metric_ranked_value = rank_func(metric_sorted_idx)
                    result_rank += metric_ranked_value
                best_idx = np.argmax(result_rank)
                best_param_value = param_values[best_idx]
                if best_guessed_params[param_name] != best_param_value:
                    print(f'    Setting param {param_name} from {best_guessed_params[param_name]} '
                          f'to {best_param_value}')
                    best_guessed_params[param_name] = best_param_value
                    param_changed = True
                else:
                    print(f'    Param {param_name} = {best_param_value} unchanged')
                best_guessed_params_metric = results[best_idx]
            if not param_changed:
                break
    print('Best guessed params:', best_guessed_params)
    print('Metric under this param:', best_guessed_params_metric)
    with open(cache_file, 'w', encoding='utf8') as f:
        json.dump(cached_param_metrics, f, separators=(',', ':'))
    print(f'All results are saved to {cache_file}')

    export_metrics = ['ndcg', 'pre', 'rec', 'fpr', 'f1_macro', 'f1_micro', 'hr', 'mrr', 'acc', 'mae', 'rmse']
    export_entries = [alg, data_name, str(r)]
    if best_metric:
        export_entries.extend(map(
            lambda x: format(best_guessed_params_metric[x]
                             [np.argmin(sort_func[x](best_guessed_params_metric[x]))], '.6g'), export_metrics))
    else:
        export_entries.extend(map(lambda x: format(best_guessed_params_metric[x], '.6g'), export_metrics))
    export_entries.append(_get_param_str(best_guessed_params))
    export_entries.append(f'dataset_seed={args.dataset_seed};param_seed={param_init_seed}'
                          f'{";best" if best_metric else ""}')
    print(','.join(export_entries))


if __name__ == '__main__':
    main()
