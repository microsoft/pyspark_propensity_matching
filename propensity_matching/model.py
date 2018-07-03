"""Defines PropensityModel"""
from collections import namedtuple
from typing import Optional, Tuple

import pyspark.ml.classification as mlc
from pyspark.sql import DataFrame

from .evaluate import evaluate as _evaluate
from .impact import impact as _impact
from .transform import transform as _transform
from .utils import _time_log


class PropensityModel:
    r"""The entry point for transform, impact, and evaluate workflows.

    Parameters / Attributes
    -----------------------
    prob_mod : pyspark.ml.classification.LogisticRegressionModel
        Model obj to predict probability of being in label class 1
        prob_mod.pred_cols houses feature columns names
        getters are also used to for label and assembled features col
    df : pyspark.sql.DataFrame
        The actual data
    train_set : pyspark.sql.DataFrame
        data used to train prob_mod
    test_set : pyspark.sql.DataFrame
        data used to test prob_mod
    response_col : str
        col holding the response variable

    Methods
    -------
    transform(df)
    determine_impact(df, matched_treatment, matched_control)
    evaluate_performance(pre_df, post_df, transform_df, by_col_group)
    """

    @_time_log
    def __init__(self,
                 prob_mod: mlc.Model,
                 df: DataFrame,
                 train_set: DataFrame,
                 test_set: DataFrame,
                 response_col: str,
                 ):

        self.prob_mod = prob_mod
        self.df = df
        self.train_set = train_set
        self.test_set = test_set
        self.response_col = response_col

    @_time_log
    def transform(self,
                  df: DataFrame) ->Tuple[DataFrame, dict]:
        r"""
        creates a dataframe with appropriate controls for the treatment
        group

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            full dataframe to propensity_match on 
        Returns
        -------
        df: pyspark.sql.DataFrame
            matched observations
        match_info : dict
            depending on matching, contains information about the match

        Raises
        ------
        UncaughtExceptiosn

        See Also
        --------
        transform in transform.py
        """
        df, match_info= _transform(df, self.prob_mod)
        return df, match_info

    @_time_log
    def determine_impact(self,
                         df: DataFrame)-> Tuple[float, float, float]:
        r"""Calculates effect of label col on response col, controlling
        for covariates

        Parameters
        ----------
        df : pyspark.sql.DataFrame


        Returns
        -------
        treatment_rate : float
            % of matched class 1s that have response 1 (as opposed to 0)
        control_rate : float
            % of matched class 1s that have response 1 (as opposed to 0)
        adjusted_response : float
            impact of label on reponse col, with further adjustments for bias

        Raises
        ------

        See Also
        --------
        impact in impact.py

        Examples
        --------
        """
        treatment_rate, control_rate, adjusted_response = _impact(df=df,
                                                                  response_col=self.response_col,
                                                                  prob_mod=self.prob_mod)
        return treatment_rate, control_rate, adjusted_response

    @_time_log
    def evaluate_performance(self,
                             pre_df: DataFrame,
                             post_df: DataFrame,
                             transform_df: DataFrame,
                             sample_size: Optional[int] = 10**6,
                             metrics_args: Optional[dict] = None)-> namedtuple:
        r"""provides goodness metrics for propensity match

        Considers both the probability model as well as the matching itself


        Parameters
        ----------
        pre_df : pyspark.sql.DataFrame
            dataframe before the propensity matching. used to calculate 
            starting standard bias
        post_df : pyspark.sql.DataFrame
            dataframe after propensity matching. used to calculate ending
            standard bias
        transform_df : pyspark.sql.DataFrame
            df transformed by probability model. used to calculate model
            goodness metrics on whole dataframe, as opposed to class 
            balances test and train sets
        sample_size: int = 10**6
            size of sample to evaluate goodness of match
        metrics_args: Optional[dict] = None
            which model metrics to calculate for each dataframe

        Returns
        -------
        performance_summary : namedtuple
            'test_prob_mod_perf': propensity_model_performance_summary
            'train_prob_mod_perf' : propensity_model_performance_summary
            'transform_prob_mod_perf' : propensity_model_performance_summary
            'bias_df': pd.DataFrame
                for each col has pre, post, absolute reduce, relative
                reduced bias
            'total_bias_reduced': float
                1 - (sum postbias of features/ sum rebias of features)
            'starting_bias_mean': float
                mean of prebias
            'starting_bias_var': float
                var of prebias
            where
                propensity_model_performance_summary : namedtuple
                    'auc' : float
                    'auprc' : float
                        area under precision recall curve
                    'threshold' : float
                    'informativeness' (f1) : float
                    'precision' : float
                    'recall' : float
                    'accuracy'  : float

        Raises
        ------
        
        See Also
        --------
        evaluate in evaluate.py


        Examples
        --------

        """
        performance_summary = _evaluate(prob_mod=self.prob_mod,
                                        pre_df=pre_df,
                                        post_df=post_df,
                                        test_df=self.test_set,
                                        train_df=self.train_set,
                                        transform_df=transform_df,
                                        sample_size=sample_size,
                                        metrics_args=metrics_args)
        return performance_summary
