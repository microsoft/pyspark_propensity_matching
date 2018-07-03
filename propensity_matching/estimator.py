"""module holding PropensityEstimator (ml.Estimator)."""
import logging
from typing import Optional, Tuple

import pyspark.ml as ml
import pyspark.ml.classification as mlc
import pyspark.ml.feature as mlf
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from .config import MINIMUM_DF_COUNT, MINIMUM_POS_COUNT, SAMPLES_PER_FEATURE
from .model import PropensityModel
from .utils import _persist_if_unpersisted, _time_log, bin_features, reduce_dimensionality, remove_redundant_features


class PropensityEstimator:
    """
    ml.Estimator to fit and return a PropensityModel.

    Instance Attributes
    -------------------
    fit_data_prep_args : dict
        args for class balance and test/train split during fitting
    probability_estimator : pyspark.ml.Model = LogisticRegression
    response_col : str = 'response'
        column containing response variable
    train_set : pyspark.sql.DataFrame
        training set used by probability estimator.
        created by _split_test_train
    test_set : pyspark.sql.DataFrame
        test set used by probability estimator.
        created by _split_test_train
    rebalanced_df : pyspark.sql.DataFrame
        dataframe with class balance given in fit_data_prep_args

    Class Attributes
    ----------------
    default_probability_estimator_args
    default_fit_data_prep_args


    Methods
    -------
    __init__(
            pred_cols: List[str],
            fit_data_prep_args: dict = default_fit_data_prep_args,
            probability_estimator_args=default_probability_estimator_args,
            probability_estimator=mlc.LogisticRegression,
            response_col='response' )
    fit(df: pyspark.sql.DataFrame)
        return PropensityModel
    """

    default_probability_estimator_args = {
        "featuresCol": "features",
        "labelCol": "label",
        "predictionCol": "prediction",
        "maxIter": 10,
        "regParam": .2,
        "elasticNetParam": 0,
        # tol":1e-6,
        "fitIntercept": True,
        # "threshold":0.5,
        # "thresholds":None,
        "probabilityCol": "probability",
        # "rawPredictionCol":"rawPrediction",
        # "standardization":True,
        # "weightCol":None,
        # "aggregationDepth":2,
        "family": "binomial"
    }

    default_fit_data_prep_args = {
        'class_balance': 1,
        'train_prop': .8,
        'bin_features': True,
        'remove_redundant_features': True,
        }

    @_time_log
    def __init__(self,
                 fit_data_prep_args: Optional[dict] = None,
                 probability_estimator: Optional[ml.Estimator] = None,
                 response_col: str ='response'):
        r"""
        Parameters
        ----------
        fit_data_prep_args: Optional[dict] = None,
            arguments around preparing the data to be fit
            default args are
            default_fit_data_prep_args = {
                'class_balance': 1,
                'train_prop': .8,
                'bin_features':True,
                'remove_redundant_features':True,
            }

            'class balance' is ratio of control_candidates : treatment
            to train the model on

           train_prop is the proportion of the population (post-rebalance)
           that is in the training set

           'bin_features' can be bool, dict, or absent.
            if you do not want to bin them here, they MUST be binned
            prior. Unbinned features will undermine validity of outcome.
            if bin_features is absent or True, bin_features will be run
            with default args. If it is a dict, it will be passed as
            kwargs to bin_features. see utils.bin_features for arg details

            'remove_redundant_features' can be bool, dict or absent
            True or absent will run remove redundant features with default
            args. Dict will passed as kwargs instead.
            see utils.remove_redundant_features for arg details

        probability_estimator: ml.Estimator = mlc.LogisticRegression

                default args are
                default_probability_estimator_args = {
                "featuresCol": "features",
                "labelCol": "label",
                "predictionCol": "prediction",
                "maxIter": 10,
                "regParam": .2,
                "elasticNetParam": 0,
                "fitIntercept": True,
                "probabilityCol": "probability",
                "family": "binomial"
            }   Correct labelCol and featuresCol
            are crucial so special attention should be paid


        response_col: str ='response'
            column in df containt the response

        Raises
        ------
        UncaughtExceptions
        """

        if probability_estimator is None:
            probability_estimator = mlc.LogisticRegression(**self.default_probability_estimator_args)

        if fit_data_prep_args is None:
            fit_data_prep_args = self.default_fit_data_prep_args

        self.fit_data_prep_args = fit_data_prep_args
        self.probability_estimator = probability_estimator
        self.response_col = response_col

        # set vals to None - will be correctly assigned in fit

        self.train_set = None
        self.test_set = None
        self.rebalanced_df = None

    @_time_log
    def fit(self, df: DataFrame) -> Tuple[PropensityModel, DataFrame]:
        """
        Fit propensity model and return.

        Must prepare df and fit probability model from estimator.
        df is rebalanced and, if necessary, features are adjusted.
        will fail if df is too small or has too few positive samples

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            dataframe containing desired data. Must have predictor columns
            as well as features, label column specificed in
            propensity_estimator_args and response col given in __init__

        Returns
        -------
        model: PropensityModel
            ml.Model object for propensity matching.
        df: DataFrame
            adjusted dataframe

        Raises
        ------
        AssertionError
            df too small
            too few positive samples


        Uncaught Errors:
            invalid param args.
        """
        _persist_if_unpersisted(df)

        df_count = df.count()
        if df_count <= MINIMUM_DF_COUNT:
            logging.getLogger(__name__).critical("df size {s:,} < {MDC:,} is too small to fit model".format(s=df_count, MDC=MINIMUM_DF_COUNT))
            raise ValueError("df is too small to fit model")

        label_col = self.probability_estimator.getOrDefault('labelCol')
        pos_count = df.where(F.col(label_col) == 1).count()

        if pos_count <= MINIMUM_POS_COUNT:
            logging.getLogger(__name__).critical("not enough positive samples {ps:,} < {MPC:,} in df to fit".format(ps=pos_count, MPC=MINIMUM_POS_COUNT))
            raise ValueError("not enough positive samples in df to fit")


        logging.getLogger(__name__).info("fitting to df with total size {n:,} and pos size {pos_n:,}".format(n=df_count, pos_n=pos_count))

        self._prep_data(df)
        self._split_test_train()
        self._prepare_probability_model()
        model = PropensityModel(
            prob_mod=self.prob_mod,
            df=self.df,
            train_set=self.train_set,
            test_set=self.test_set,
            response_col=self.response_col
        )
        return model, self.df

    @_time_log
    def _prep_data(self,
                   df: DataFrame):
        r"""
        remove highly collinear features, bin the features, and
        reduce the dimensionality if necessary and in that order

        Parameters
        ----------
        df : pyspark.sql.DataFrame
        self.fit_data_prep_args : dict
            arguments around preparing the data to be fit
            default args are
            default_fit_data_prep_args = {
                'class_balance': 1,
                'train_prop': .8,
                'bin_features':True,
                'remove_redundant_features':True,
            }

            'class balance' is ratio of control_candidates : treatment
            to train the model on

           train_prop is the proportion of the population (post-rebalance)
           that is in the training set

           'bin_features' can be bool, dict, or absent.
            if you do not want to bin them here, they MUST be binned
            prior. Unbinned features will undermine validity of outcome.
            if bin_features is absent or True, bin_features will be run
            with default args. If it is a dict, it will be passed as
            kwargs to bin_features. see utils.bin_features for arg details

            'remove_redundant_features' can be bool, dict or absent
            True or absent will run remove redundant features with default
            args. Dict will passed as kwargs instead.
            see utils.remove_redundant_features for arg details


        Returns
        -------
        df : pyspark.sql.DataFrame
            prepared dataframe


        Raises
        ------
        UncaughtExceptions

        See Also
        --------
        remove_redundant_features
        bin_features
        reduce_dimensionality
        """

        features_col = self.probability_estimator.getOrDefault('featuresCol')
        label_col = self.probability_estimator.getOrDefault('labelCol')

        if ('remove_redundant_features' not in self.fit_data_prep_args) | (self.fit_data_prep_args['remove_redundant_features'] is True):
            logging.getLogger(__name__).info("removing redundant features with default args")
            df, pred_cols = remove_redundant_features(df=df, features_col=features_col)
        elif isinstance(self.fit_data_prep_args['remove_redundant_features'], dict):
            logging.getLogger(__name__).info("removing redundant features with specified args")
            df, pred_cols = remove_redundant_features(df=df, **self.fit_data_prep_args['remove_redundant_features'])
        elif self.fit_data_prep_args['remove_redundant_features'] is False:
            logging.getLogger(__name__).info("not removing redundant features")
        else:
            logging.getLogger(__name__).critical("illegal arg for remove_redundant_features")
            raise ValueError('illegal argument for "remove_redundant_features" in fit_data_prep_args')

        if ('bin_features' not in self.fit_data_prep_args) | (self.fit_data_prep_args['bin_features'] is True):
            logging.getLogger(__name__).info("binning features with default args")
            df, pred_cols = bin_features(df=df, features_col=features_col)
        elif isinstance(self.fit_data_prep_args['bin_features'], dict):
            logging.getLogger(__name__).info("binning features with specified args")
            df, pred_cols = bin_features(df=df, **self.fit_data_prep_args['bin_features'])
        elif self.fit_data_prep_args['bin_features'] is False:
            logging.getLogger(__name__).info("not binning features")
        else:
            logging.getLogger(__name__).critical("illegal arg for bin_features")
            raise ValueError('illegal argument for "bin_features" in fit_data_prep_args')

        # leakage note: evaluation of informativeness of predictors includes test set
        # not ideal but minimal impact and is expedient for architecture right now.

        # num cols is limited by size of training set. To get it we must first rebalance, and multiply by train prop.
        # reduce dim on whole pop df, then apply the same transform to the rebalanced df
        self.df = df
        self._rebalance_df()
        ncols = int((self.rebalanced_df.where(F.col(label_col) == 1).count() * self.fit_data_prep_args['train_prop'])//SAMPLES_PER_FEATURE)
        red_dim_args = {'df': self.df,
                        'label_col': label_col,
                        'binned_features_col': features_col,
                        'ncols': ncols}
        logging.getLogger(__name__).info("reducing dimensionality of df")
        self.df, pred_cols = reduce_dimensionality(args=red_dim_args)

        assembler = mlf.VectorAssembler(inputCols=pred_cols, outputCol=features_col)
        self.rebalanced_df = assembler.transform(self.rebalanced_df.drop(features_col))

        return True

    @_time_log
    def _rebalance_df(self,) -> bool:
        """
        Create new df with forced class balance for label to help with training.

        Raises
        ------
        NotImplementedError
            where there is more of class 1 than class 0

        Uncaught Errors
            where class balance is less than 1

        """
        label_col = self.probability_estimator.getOrDefault('labelCol')
        num_1 = self.df.where(F.col(label_col) == 1).count()
        num_0 = self.df.where(F.col(label_col) == 0).count()

        logging.getLogger(__name__).info("rebalancing df to {class_balance} with pos count {n_pos:,} and neg count {n_neg:,}".format(n_pos=num_1, n_neg=num_0, class_balance=self.fit_data_prep_args['class_balance']))

        if num_1 > num_0:
            logging.getLogger(__name__).critical("class rebalance not implemented for class 1 > class 0")
            raise NotImplementedError("class rebalance not implemented for class 1 > class 0")

        # should have already failed out in fit if num_1 would have returned 0
        max_ratio = num_0 / num_1
        # if desired class ratio is impossible, take max possible ratio, reassign class balance and warn
        if self.fit_data_prep_args['class_balance'] > max_ratio:
            logging.getLogger(__name__).warning("Maximum class balance is {max_ratio} but requested is {class_balance} \
            Changing to {max_ratio}".format(max_ratio=max_ratio, class_balance=self.fit_data_prep_args['class_balance']))
            self.fit_data_prep_args['class_balance'] = max_ratio

        desired_num_0 = self.fit_data_prep_args['class_balance'] * num_1
        sample_frac_0 = min(1, float(desired_num_0 / num_0))  # protect against non-float types (numpy) & floating point error
        rebalanced_df_0 = self.df.where(F.col(label_col) == 0).sample(withReplacement=False, fraction=float(sample_frac_0), seed=42)
        rebalanced_df = rebalanced_df_0.select(self.df.columns).union(self.df.where(F.col(label_col) == 1).select(self.df.columns))
        self.rebalanced_df = rebalanced_df
        logging.getLogger(__name__).info("rebalanced df into size {n}".format(n=rebalanced_df.count()))

        return True

    @_time_log
    def _split_test_train(self) -> bool:
        """Create test, train set attributes based on fit_data_prep_args."""
        self.train_set, self.test_set = self.rebalanced_df.randomSplit([self.fit_data_prep_args['train_prop'], 1 - self.fit_data_prep_args['train_prop']])
        return True

    @_time_log
    def _prepare_probability_model(self):
        """Fit probability model"""
        prob_mod = self.probability_estimator.fit(self.train_set)
        # guard against overfit happened in fit before _rebalance_df and _split_test_train were called
        self.prob_mod = prob_mod
        return True
