# grid search version of hyper-param tuning
import multiprocessing as mp
from evaluator import Metric
import numpy as np
from Dataloader import Dataloader, get_train_test_data, CVDataloader
import os
import json
import argparse
from typing import *
import warnings
import threading
import bayes_opt

# type checking hints
if TYPE_CHECKING:
    import scipy.sparse as sps
    from rec_model import RecModel


# noinspection DuplicatedCode
def run_metric(dataset: 'Tuple[sps.csr_matrix, sps.csr_matrix]', model: 'RecModel', k: 'int',
               evaluate_policy: str = 'man', q: 'Optional[mp.Queue]' = None):
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


# noinspection DuplicatedCode
def run_model(args: tuple) -> Tuple[Dict[str, float], Dict[str, float]]:
    hp, model, dataset, seed, evaluate_policy, verbose, max_itr, max_itr2, k = args
    np.random.seed(seed)  # fix seed before param initialization
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
        return hp, {k: np.nan for k in ['pre', 'rec', 'fpr', 'acc', 'f1_micro', 'f1_macro', 'ndcg', 'mae', 'rmse', 'hr',
                                        'mrr', 'rec_micro']}
    result = run_metric(dataset, model, k, evaluate_policy)
    return hp, result


def import_rec_model_from_str(model_str):
    import importlib
    module_name, class_name = model_str.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class SearchSpaceConverter:
    def __init__(self, range_definitions: Dict[str, np.ndarray]):
        self._range_definitions = range_definitions

    def get_pbounds(self) -> Dict[str, Tuple[float, float]]:
        return {k: (0, len(v)) for k, v in self._range_definitions.items()}

    def convert(self, x: Dict[str, float]) -> Dict[str, float]:
        return {k: float(self._range_definitions[k][min(int(v), len(self._range_definitions[k])-1)])
                for k, v in x.items()}


# noinspection DuplicatedCode
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
    parser.add_argument('--max_epoch', type=int, default=10)  # for SoRec, it refers to max_iter
    parser.add_argument('--max_dcd_itr', type=int, default=10)

    # bayes_opt related parameters
    parser.add_argument('--util_func', type=str, default='ucb', choices=['ucb', 'ei', 'poi'])
    parser.add_argument('--kappa', type=float, default=3.0)  # lower: exploitation, higher: exploration
    parser.add_argument('--xi', type=float, default=1.0)  # lower: exploitation, higher: exploration
    parser.add_argument('--bayes_opt_seed', type=int, default=123)
    parser.add_argument('--init_points', type=int, default=20)
    parser.add_argument('--tune_iter', type=int, default=200)  # max trials

    parser.add_argument('--disable_cache', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--evaluate_policy', choices=['mau', 'man'], default='man')  # missing as unknown / negative
    parser.add_argument('--n_folds', type=int, default=5)  # for cross validation, n_folds<=1 to disable
    parser.add_argument('--low_priority', default=False, action='store_true')
    args = parser.parse_args()
    return args


def _get_param_str(param: Dict[str, Any]) -> str:
    return ';'.join(map(lambda x: f'{x[0]}={x[1]}', param.items()))


def _get_param_dict(param_str: str) -> Dict[str, float]:
    param_dict = {}
    for param in param_str.split(';'):
        k, v = param.split('=')
        param_dict[k] = float(v)
    return param_dict


def low_priority():
    """ Set the priority of the process to low."""
    import sys
    try:
        sys.getwindowsversion()
    except AttributeError:
        is_windows = False
    else:
        is_windows = True

    if is_windows:
        # Based on:
        #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
        #   http://code.activestate.com/recipes/496767/
        try:
            import win32api
            import win32process
            import win32con

            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.BELOW_NORMAL_PRIORITY_CLASS)
        except ImportError:
            pass
    else:
        import os
        os.nice(5)


# noinspection DuplicatedCode
def main():
    args = parse_args()
    if args.low_priority:
        low_priority()

    # other user-defined params
    tune_range = np.power(10.0, np.arange(-5, 6))
    tune_range2 = 1 - np.power(10, np.arange(-4, -1, dtype=np.float))
    tune_range3 = np.power(10.0, np.arange(-2, 9))  # added for CiaoDVD, we still use tune_range when training filmtrust
    # what metric should be taken into consideration, all available metrics are list as follows:
    # pre, pre_micro, rec, rec_micro, fpr, acc, f1_macro, f1_micro, ndcg, mae, rmse, hr, mrr
    rank_metric = 'f1_macro'
    assert rank_metric in {'pre', 'rec', 'fpr', 'acc', 'f1_macro', 'f1_micro', 'ndcg', 'mae', 'rmse', 'hr', 'mrr'},\
        'Invalid metric'

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
    # rank_func_name = args.rank_func
    param_init_seed = args.param_seed
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
    }  # type: Dict[str, Dict[str, Any]]

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
    # rank_func_mapper = {'reciprocal': lambda x: 1. / (x + 1), 'log2': lambda x: 1. / np.log2(x + 1)}
    # rank_func = rank_func_mapper[rank_func_name]
    rank_metric_sorting_kind = {'max': ['pre', 'pre_micro', 'rec', 'rec_micro', 'acc', 'f1_micro', 'f1_macro', 'ndcg',
                                        'hr', 'mrr'],
                                'min': ['fpr', 'mae', 'rmse']}
    # sort_func_mapper = {'max': lambda x: rankmin(-np.array(x, dtype=np.float)), 'min': rankmin}
    # sort_func = {metric_name: sort_func_mapper[sort_kind]
    #              for sort_kind, metric_names in rank_metric_sorting_kind.items() for metric_name in metric_names}

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
                 f'{"_mau" if args.evaluate_policy == "mau" else ""}.json'
    if use_cache and os.path.isfile(cache_file):
        with open(cache_file, 'r', encoding='utf8') as f:
            cached_param_metrics = json.load(f)
    else:
        cached_param_metrics = {}

    # <<<<< setting up bayesian optimization
    util_fn_kwargs = {'kind': args.util_func, 'kappa': args.kappa, 'xi': args.xi}
    print('Utility function for Bayesian optimization:', util_fn_kwargs)
    util_fn = bayes_opt.util.UtilityFunction(**util_fn_kwargs)
    space = SearchSpaceConverter(alg_tune_params[alg])

    def _get_metric_result(*args_, **kwargs):
        hp, metrics = run_model(*args_, **kwargs)
        value = metrics[rank_metric]
        if rank_metric in rank_metric_sorting_kind['min']:
            value = -value
        return hp, value
    optimizer = bayes_opt.BayesianOptimization(f=_get_metric_result, pbounds=space.get_pbounds(),
                                               random_state=args.bayes_opt_seed)
    # target function "f" is only used in "probe" method in Bayesian optimizer, naturally we do not rely on it
    # <<<<< load points from cache
    distinct_runs = set()
    for hyper_param_str, logs in cached_param_metrics.items():
        metric = logs['metric']
        point = logs['bayes_point']
        if hyper_param_str not in distinct_runs:
            metric_value = metric[rank_metric]
            if rank_metric in rank_metric_sorting_kind['min']:
                metric_value = -metric_value
            optimizer.register(params=point, target=metric_value)
            distinct_runs.add(hyper_param_str)
    last_best = ''
    if len(cached_param_metrics) > 0:
        # log the previous best result
        best_results = optimizer.max
        best_guessed_params = space.convert(best_results['params'])
        best_guessed_params_metric = cached_param_metrics[_get_param_str(best_guessed_params)]['metric'].copy()
        # keep first three digits
        best_guessed_params_metric = {k: round(float(v), 6) for k, v in best_guessed_params_metric.items()}
        iteration = len(optimizer.space.params)
        last_best = _get_param_str(best_guessed_params)
        print(f'Last best: iteration {iteration}/{args.tune_iter}: Metric: {best_guessed_params_metric}; '
              f'Param: {last_best}')
    optimizer_lock = threading.Lock()
    result_updated = threading.Event()

    def _suggest_no_dup():
        for _ in range(50):
            if len(optimizer.space.params) >= args.tune_iter:
                raise ValueError('tune_iter reached.')
            pt_str = ''
            for _ in range(300):  # loop 300 times and exit if no new point is found
                with optimizer_lock:
                    pt = optimizer.suggest(util_fn)
                pt_converted = space.convert(pt)
                pt_str = _get_param_str(pt_converted)
                if pt_str not in distinct_runs:
                    distinct_runs.add(pt_str)
                    return pt, pt_converted
            if pt_str not in cached_param_metrics:
                print(f'_suggest_no_dup: waiting for result of point {pt_str}')
                result_updated.clear()
                result_updated.wait()
            if pt_str in cached_param_metrics:
                with optimizer_lock:
                    optimizer.register(params=pt, target=cached_param_metrics[pt_str]['metric'][rank_metric])
                print(f'Iteration {len(optimizer.space.params)}/{args.tune_iter}: reusing cached point {pt_str}')
        print('Training terminated due to no new point is found.')
        raise ValueError('No new point is found.')

    def _gen_suggest_no_dup():
        # sample initial points at first (before any results are registered to optimizer)
        exist_points = len(optimizer.space.params)
        init_points = []
        for _ in range(exist_points, args.init_points):
            try:
                init_points.append(_suggest_no_dup())
            except ValueError:
                break
        yield from init_points
        while len(optimizer.space.params) < args.tune_iter:
            try:
                yield _suggest_no_dup()
            except ValueError:
                break

    def _gen_suggest_cv():
        for pt, param_dict_converted in _gen_suggest_no_dup():
            param_dict_converted = param_dict_converted.copy()
            for fold in range(n_folds):
                param_dict_converted['fold'] = fold
                yield pt, param_dict_converted

    def _gen_model_args(hyper_param_generator: Generator[Tuple[Dict[str, Any], Dict[str, Any]], None, None]):
        for pt, hyper_param in hyper_param_generator:
            fold = hyper_param.pop('fold', 0)
            # hp, model, dataset, seed, evaluate_policy, verbose, max_itr, max_itr2, k
            alg_init_param = alg_init_params[alg].copy()
            alg_cls = import_rec_model_from_str(alg_class_mapper[alg])
            hyper_param_str = _get_param_str(hyper_param)
            if hyper_param_str in cached_param_metrics:
                continue  # skip evaluated hyper-parameters
            # cross validation params
            rating_param_name = alg_cv_init_params.get(alg, {}).get('rating_matrix', 'rating_matrix')
            social_param_name = alg_cv_init_params.get(alg, {}).get('social_matrix', 'social_matrix')
            cornac_data_param_name = alg_cv_init_params.get(alg, {}).get('cornac_train_set', None)
            if cornac_data_param_name is not None:
                alg_init_param[cornac_data_param_name] = r_train_dataset[fold]
            else:
                if rating_param_name is not None:
                    alg_init_param[rating_param_name] = dataset[fold][0]
                if social_param_name is not None:
                    alg_init_param[social_param_name] = s_bin
            # hyper params
            alg_init_param.update(hyper_param)
            model = alg_cls(**alg_init_param)
            yield (pt, hyper_param), model, dataset[fold], param_init_seed, args.evaluate_policy, \
                verbose, max_itr, max_itr2, k

    # <<<<< main search loop
    cv_result_cache = {}
    # with mp.Pool(max_process, maxtasksperchild=1) as p:
    with mp.Pool(max_process) as p:
        for (pt, run_param), run_result in p.imap_unordered(run_model, _gen_model_args(_gen_suggest_cv())):
            param_str = _get_param_str(run_param)
            if param_str not in cv_result_cache:
                cv_result_cache[param_str] = [run_result]
            else:
                cv_result_cache[param_str].append(run_result)

            if len(cv_result_cache[param_str]) < n_folds:
                continue

            results = cv_result_cache.pop(param_str)
            cv_result = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)  # ignore nanmean warning
                for metric in results[0].keys():
                    metric_values = [r[metric] for r in results]
                    metric_value = np.nanmean(metric_values)
                    if np.isnan(metric_value):
                        if metric in rank_metric_sorting_kind['min']:  # mae etc
                            metric_value = 10.0
                        else:
                            metric_value = 0.0
                    cv_result[metric] = metric_value

            cached_param_metrics[param_str] = {'metric': cv_result, 'bayes_point': pt}
            with open(cache_file, 'w', encoding='utf8') as f:
                json.dump(cached_param_metrics, f, separators=(',', ':'))

            target_metric_value = cv_result[rank_metric]
            if rank_metric in rank_metric_sorting_kind['min']:
                target_metric_value = -target_metric_value
            with optimizer_lock:
                optimizer.register(params=pt, target=target_metric_value)
                # print out current best
                best_results = optimizer.max
                iteration = len(optimizer.space.params)
            result_updated.set()
            best_guessed_params = space.convert(best_results['params'])
            best_guessed_params_str = _get_param_str(best_guessed_params)
            best_guessed_params_metric = cached_param_metrics[best_guessed_params_str]['metric'].copy()
            cv_result_rounded = {k: round(float(v), 6) for k, v in cv_result.items()}
            print(f'Iteration {iteration}/{args.tune_iter}: Metric: {cv_result_rounded}; Param: {param_str};')
            if last_best != best_guessed_params_str:
                last_best = best_guessed_params_str
                best_guessed_params_metric = {k: round(float(v), 6) for k, v in best_guessed_params_metric.items()}
                print(f'New best: Metric: {best_guessed_params_metric}; Param: {best_guessed_params_str}')

    export_metrics = ['ndcg', 'pre', 'rec', 'fpr', 'f1_macro', 'f1_micro', 'hr', 'mrr', 'acc', 'mae', 'rmse']
    export_entries = [alg, data_name, str(r)]
    export_entries.extend(map(lambda x: format(best_guessed_params_metric[x], '.6g'), export_metrics))
    export_entries.append(_get_param_str(best_guessed_params))
    export_entries.append(f'dataset_seed={args.dataset_seed};param_seed={param_init_seed}')
    print(','.join(export_entries))


if __name__ == '__main__':
    main()
