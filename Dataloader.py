import os
import sys


def _cornac_pre_init():
    origin_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        import cornac
    finally:
        sys.stdout.close()
        sys.stdout = origin_stdout


class DataloaderBase:
    @staticmethod
    def load_data(data_name, user_filter):
        """

        :param data_name: filmtrust/CiaoDVD/Epinions
        :param user_filter: Filter out users whose interaction times are less than the set value
        """
        _cornac_pre_init()
        assert user_filter >= 0
        from dataset import filmtrust, ciaodvd, epinions

        if data_name.endswith('filmtrust'):
            rating = filmtrust.load_feedback(min_user_freq=user_filter)
            trust = filmtrust.load_trust()
        elif data_name.endswith('CiaoDVD'):
            rating = ciaodvd.load_feedback(min_user_freq=user_filter)
            trust = ciaodvd.load_trust()
        elif data_name.endswith('Epinions'):
            rating = epinions.load_feedback(min_user_freq=user_filter)
            trust = epinions.load_trust()
        else:
            raise ValueError("{} is not defined".format(data_name))
        return rating, trust


class Dataloader(DataloaderBase):
    @staticmethod
    def load_data(data_name, user_filter, frac=0.8):
        """

        :param data_name: filmtrust/CiaoDVD/Epinions
        :param user_filter: Filter out users whose interaction times are less than the set value
        :param frac: ratio of training set
        """
        _cornac_pre_init()
        from cornac.data import GraphModality
        from cornac.eval_methods import RatioSplit

        rating, trust = DataloaderBase.load_data(data_name, user_filter)
        user_graph = GraphModality(data=trust)
        split = RatioSplit(
            test_size=1 - frac, val_size=None,
            data=rating,
            user_graph=user_graph,
            rating_threshold=0.0,
            seed=0, exclude_unknowns=True, verbose=False
        )

        # if iscornac:
        #     return split.train_set, split.test_set
        # else:
        #     return split.train_set.matrix, split.test_set.matrix, split.train_set.user_graph.matrix
        return split


class CVDataloader(DataloaderBase):
    @staticmethod
    def load_data(data_name, user_filter, folds=5):
        """

        :param data_name: filmtrust/CiaoDVD/Epinions
        :param user_filter: Filter out users whose interaction times are less than the set value
        :param folds: folds of cross validation
        """
        _cornac_pre_init()
        from cornac.data import GraphModality
        from cornac.eval_methods import CrossValidation

        rating, trust = DataloaderBase.load_data(data_name, user_filter)
        user_graph = GraphModality(data=trust)
        split = CrossValidation(
            n_folds=folds,
            data=rating,
            user_graph=user_graph,
            rating_threshold=0.0,
            seed=0, exclude_unknowns=True, verbose=False
        )
        return split


# util functions for cross validation
def get_train_test_data(split, folds=None):
    if folds is not None:
        split.current_fold = folds
        # noinspection PyProtectedMember
        split._get_train_test()
    r_train = split.train_set.matrix
    r_test = split.test_set.matrix
    s_bin = split.train_set.user_graph.matrix
    r_train.eliminate_zeros()
    r_test.eliminate_zeros()
    s_bin.eliminate_zeros()
    return r_train, r_test, s_bin
