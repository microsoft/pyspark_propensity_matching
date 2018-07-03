import warnings
import logging
from collections import namedtuple
from typing import Optional, Tuple, Dict

import pandas as pd
import numpy as np

import pyspark
from pyspark import StorageLevel
from pyspark.sql import DataFrame, Window
from pyspark.ml import classification as mlc
from pyspark.ml import evaluation as mle
import pyspark.sql.functions as F
import pyspark.sql.types as T

from .utils import _persist_if_unpersisted, _get_pred_cols, _time_log, _sample_df

# container for model performance
propensity_model_performance_summary = namedtuple('propensity_model_performance_summary', [
    'auc',
    'auprc',
    'threshold',
    'informativeness',
    'precision',
    'recall',
    'accuracy'
    ])

# container for whole operation performance
performance_summary = namedtuple('performance_summary', [
    'train_prob_mod_perf',
    'test_prob_mod_perf',
    'transform_prob_mod_perf',
    'bias_df',
    'total_bias_reduced',
    'starting_bias_mean',
    'starting_bias_var'
    ])


@_time_log
def evaluate(prob_mod: pyspark.ml.Model,
             pre_df: Optional[DataFrame] = None,
             post_df: Optional[DataFrame] = None,
             test_df: Optional[DataFrame] = None,
             train_df: Optional[DataFrame] = None,
             transform_df: Optional[DataFrame] = None,
             sample_size: Optional[int] = 10**6,
             metrics_args: Optional[dict] = None
             ) -> performance_summary:
    r"""evaluates the goodness of match and the power of the propensity
    model

    Parameters
    ----------
    prob_mod : pyspark.ml.Model
        pyspark model that predict probability row is in postive class
        1 in label col
    pre_df : pyspark.sql.DataFrame
        dataframe before matching. used to assess starting bias. If there
        are no positive samples, all goodness of match metrics return None
    post_df : pyspark.sql.DataFrame
        dataframe after matching used to assess final bias. If there
        are no positive samples, all goodness of match metrics return None
    test_df : pyspark.sql.DataFrame
        test dataset for `prob_mod`. if `None`, test_prob_mod_perf returns
        `None`
    train_df: pyspark.sql.DataFrame
        dataframe `prob_mod` was trained on. if `None`, train_prob_mod_perf
        returns `None`
    transform_df : pyspark.sql.DataFrame
        the whole dataframe. Often a superset of `train_df` and `test_df`
        also frequently has different class balance
        if `None`, transform_prob_mod_perf returns `None`
    sample_size: int = 10**6
        passed to eval match performance. Indicates size of sample to
        evaluate match performance - used to improve speed of calculation.
        Maybe be left as None for no sampling.
    metrics_args : dict
        passed to _eval_propenisty_model - not to match performance
        dict where keys are df var names and value is bool of whether to
        retrieve acc, prec, rec, thresh, informativeness for them. default
        value is {"transform_df":True}. Keys not in metrics_arg are treated
        as false. can take several min to calc per df so the "test_df" and
        "train_df" are left out for performance. Note that auc and auprc
        are always retrieved.

    Returns
    -------
    performance_summary : performance_summary named tuple

    Notes
    -----
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
    propensity_model_performance_summary : namedtuple
        'auc' : float
        'auprc' : float
            area under precision recall curve
        'threshold' : float
        'informativeness' (f1) : float
        'precision' : float
        'recall' : float
        'accuracy'  : float

    Bias is calculated on predictors which may or may not be the originally
    provided data (e.g. binning, dropping predictors). This is wad but user
    should be aware.

    Raises
    ------
    ValueError
        if all model metric dfs are none
    UncaughtErrors

    See Also
    --------
    _eval_propensity_model
    _eval_match_performance



    """

    test_prob_mod_perf, train_prob_mod_perf, transform_prob_mod_perf = \
        _eval_propensity_model(prob_mod=prob_mod,
                               test_df=test_df,
                               train_df=train_df,
                               transform_df=transform_df,
                               metrics_args=metrics_args)

    label_col = prob_mod.getOrDefault('labelCol')
    features_col = prob_mod.getOrDefault('featuresCol')
    if test_df is not None:
        pred_col_df = test_df
    elif train_df is not None:
        pred_col_df = train_df
    elif transform_df is not None:
        pred_col_df = transform_df
    else:
        logging.getLogger(__name__).critical("all model metric dfs are None")
        raise ValueError("All model metric dfs are None")

    pred_cols = _get_pred_cols(df=pred_col_df, features_col=features_col)

    cols = pred_cols + [label_col]
    try:
        bias_df, total_bias_reduced, starting_bias_mean, starting_bias_var \
            = _eval_match_performance(pre_df.select(cols), post_df.select(cols), label_col, sample_size=sample_size)
    except ValueError as e:
        logging.getLogger(__name__).warning("eval_match_performance failed with warning {e}".format(e=str(e)))
        warnings.warn(e)
        bias_df = None
        total_bias_reduced = None
        starting_bias_mean = None
        starting_bias_var = None

    perf_sum = performance_summary(
            train_prob_mod_perf=train_prob_mod_perf,
            test_prob_mod_perf=test_prob_mod_perf,
            transform_prob_mod_perf=transform_prob_mod_perf,
            bias_df=bias_df,
            total_bias_reduced=total_bias_reduced,
            starting_bias_mean=starting_bias_mean,
            starting_bias_var=starting_bias_var
    )
    return perf_sum


@_time_log
def _eval_propensity_model(prob_mod: pyspark.ml.Model,
                           test_df: Optional[DataFrame],
                           train_df: Optional[DataFrame],
                           transform_df: Optional[DataFrame],
                           metrics_args: Optional[dict])\
        ->Tuple[Optional[propensity_model_performance_summary],
                Optional[propensity_model_performance_summary],
                Optional[propensity_model_performance_summary]]:
    r"""For test, train, and transform df, evaluate the propensity
    predicting model.


    Parameters
    ----------
    prob_mod : pyspark.ml.Model
        pyspark model that predict probability row is in postive class
        1 in label col
    test_df : pyspark.sql.DataFrame
        test dataset for `prob_mod`. if `None`, test_prob_mod_perf returns
        `None`
    train_df: pyspark.sql.DataFrame
        dataframe `prob_mod` was trained on. if `None`, train_prob_mod_perf
        returns `None`
    transform_df : pyspark.sql.DataFrame
        the whole dataframe. Often a superset of `train_df` and `test_df`
        also frequently has different class balance
        if `None`, transform_prob_mod_perf returns `None`
    metrics_args : dict
        dict where keys are df var names and value is bool of whether to
        retrieve acc, prec, rec, thresh, informativeness for them. default
        value is {"transform_df":True}. Keys not in metrics_arg are treated
        as false. can take several min to calc per df so the "test_df" and
        "train_df" are left out for performance.


    Returns
    -------
    test_prob_mod_perf : performance_summary
    train_prob_mod_perf : performance_summary
    transform_prob_mod_perf : performance_summary


    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _eval_df_model


    Notes
    -----
    if a df is None, it's corresponding per tuple will instead be None

    propensity_model_performance_summary : namedtuple
        'auc' : float
        'auprc' : float
            area under precision recall curve
        'threshold' : float
        'informativeness' (f1) : float
        'precision' : float
        'recall' : float
        'accuracy'  : float
    """
    if metrics_args is None:
        metrics_args = {"transform_df" : True}

    if train_df is None:
        train_prob_mod_perf = None
        logging.getLogger(__name__).info("train_df is None")
    else:
        train_prob_mod_perf = _eval_df_model(df=train_df,
                                             prob_mod=prob_mod,
                                             additional_metrics=metrics_args.get("train_df", False))
        logging.getLogger(__name__).info("train_prob_mod_perf: {perf}".format(perf=str(train_prob_mod_perf)))

    if test_df is None:
        test_prob_mod_perf = None
        logging.getLogger(__name__).info("test_df is None")
    else:
        test_prob_mod_perf = _eval_df_model(df=test_df,
                                            prob_mod=prob_mod,
                                            additional_metrics=metrics_args.get("test_df", False))
        logging.getLogger(__name__).info("test_prob_mod_perf: {perf}".format(perf=str(test_prob_mod_perf)))

    if transform_df is None:
        transform_prob_mod_perf = None
        logging.getLogger(__name__).info("transform_df is None")
    else:
        transform_prob_mod_perf = _eval_df_model(df=transform_df,
                                                 prob_mod=prob_mod,
                                                 additional_metrics=metrics_args.get("transform_df", False))
        logging.getLogger(__name__).info("transform_prob_mod_perf: {perf}".format(perf=str(transform_prob_mod_perf)))

    return test_prob_mod_perf, train_prob_mod_perf, transform_prob_mod_perf


@_time_log
def _eval_df_model(df: DataFrame,
                   prob_mod: pyspark.ml.Model,
                   sample_size: Optional[int] = 10 ** 5,
                   additional_metrics: bool = False) -> propensity_model_performance_summary:
    r"""calculate binary classification model metrics on provided dataframe

    Calculate accuracy, precision, and recall at maximum value for
    informativeness (f1) also provide auc and auprc(precision-recall curve)


    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    prob_mod : pyspark.ml.Model
        The type above can either refer to an actual Python type
        (e.g. ``int``), or describe the type of the variable in more
        detail, e.g. ``(N,) ndarray`` or ``array_like``.
    sample_size
    additional_metrics


    Returns
    -------
    prob_mod_perf_sum : propensity_model_performance_summary

    Other Parameters
    ----------------
    sample_size: int
        max sample size to calculate performance. Defaults to 10**6, can be
        left as None to avoid sampling

    Notes
    -----
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
    UncaughtExceptions

    See Also
    --------
    _calc_auc_auprc
    _calc_model_metrics
    """
    sample_df = _sample_df(df=df, sample_size=sample_size)
    _persist_if_unpersisted(sample_df)

    label_col = prob_mod.getOrDefault('labelCol')
    prob_col = prob_mod.getOrDefault('probabilityCol')
    features_col = prob_mod.getOrDefault('featuresCol')
    sample_df = prob_mod.transform(sample_df.select(features_col, label_col))

    prob_1_col = "{prob_col}_1".format(prob_col=prob_col)
    sample_df = sample_df.withColumn(prob_1_col, F.udf(lambda x: float(x[1]), T.DoubleType())(F.col(prob_col)))
    sample_df.persist(StorageLevel(False, True, False, False, 1))

    auc, auprc = _calc_auc_auprc(df=sample_df, prob_col=prob_1_col, label_col=label_col)

    metric_keys = ['threshold', 'informativeness', 'precision', 'recall', 'accuracy']
    if additional_metrics:
        metrics_dict = _calc_model_metrics(df=sample_df, prob_col=prob_1_col, label_col=label_col)
        metrics_dict = {x: metrics_dict[x] for x in metrics_dict.keys() if x in metric_keys}
    else:
        metrics_dict = {x: None for x in metric_keys}

    prob_mod_per_sum = propensity_model_performance_summary(auc=auc, auprc=auprc, **metrics_dict)

    return prob_mod_per_sum


@_time_log
def _calc_auc_auprc(df: DataFrame,
                    prob_col: str,
                    label_col: str) -> Tuple[float, float]:
    r"""
    Given a df, labels, and probabilities, return auc and auprc (area under
    precision recall curve)

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    prob_col : str
        colname w/ raw probabilities of being in class 1
    label_col : str

    Returns
    -------
    auc :float
    aucprc : float
        area under precision recall curve

    Raises
    ------
    UncaughExceptions

    See Also
    --------
    pyspark.ml.evaluation.BinaryClassificationEvaluator
    """

    auc_eval = mle.BinaryClassificationEvaluator(rawPredictionCol=prob_col, labelCol=label_col,
                                                 metricName='areaUnderROC')
    auc = auc_eval.evaluate(df)

    auprc_eval = mle.BinaryClassificationEvaluator(rawPredictionCol=prob_col, labelCol=label_col,
                                                   metricName='areaUnderPR')
    auprc = auprc_eval.evaluate(df)

    return auc, auprc


@_time_log
def _calc_model_metrics(df: DataFrame,
                        prob_col: str,
                        label_col: str) -> Dict[str, float]:
    r"""calc model metrics at max f1 given probabilities and labels

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    prob_col : str
        colname w/ raw probabilities of being in class 1
    label_col : str

    Returns
    -------
    max_metrics : dict
        dict with keys : 'tp', 'tn', 'fp', 'fn', 'f1', 'accuracy',
        'precision', 'recall' and corresponding values as floats

    Raises
    ------
    UncaughtException
    """
    _persist_if_unpersisted(df)

    metrics_df = df.groupby(prob_col).pivot(label_col).count().fillna(value=0)
    metrics_df.persist(StorageLevel(False, True, False, False))

    window = Window.orderBy(prob_col).rowsBetween(Window.unboundedPreceding, -1)
    metrics_df = metrics_df.withColumn('fn', F.sum(F.col(str(1))).over(window))
    metrics_df = metrics_df.withColumn('tn', F.sum(F.col(str(0))).over(window)).fillna(value=0)
    metrics_df.persist(StorageLevel(False, True, False, False, 1))

    all_count = df.count()
    pos_count = df.where(F.col(label_col) == 1).count()
    neg_count = all_count - pos_count

    metrics_df = metrics_df.withColumn('tp', pos_count - F.col('fn'))
    metrics_df = metrics_df.withColumn('fp', neg_count - F.col('tn'))

    metrics_df = metrics_df.withColumn('precision', (F.col('tp')) / (F.col('tp') + F.col('fp')))
    metrics_df = metrics_df.withColumn('recall', F.col('tp') / pos_count)

    metrics_df = metrics_df.withColumn('informativeness',
                                       2 * (F.col('precision') * F.col('recall')) / (F.col('precision') + F.col('recall')))
    metrics_df.persist(StorageLevel(False, True, False, False))

    max_metrics = metrics_df.where(
        F.col('informativeness') == metrics_df.select(F.max(F.col('informativeness'))).take(1)[0][0]).take(1)[
        0].asDict()
    max_metrics['accuracy'] = (max_metrics['tp'] + max_metrics['tn']) / all_count
    max_metrics['threshold'] = max_metrics[prob_col]

    return max_metrics


@_time_log
def _eval_match_performance(pre_df: DataFrame,
                            post_df: DataFrame,
                            label_col: str,
                            sample_size: Optional[int]) -> Tuple[pd.DataFrame, float, float, float]:
    r""" evaluate propensity match performance given the pre and post
    match samples

    For each predictor evaluate the starting and ending bias, absolute and
    relative bias reduced. On aggregate, evaluate relative bias reduced,
    as well as the avg starting bias and its variance

    Parameters
    ----------
    pre_df : pyspark.sql.DataFrame
        dataframe before matching occured. each row should be a sample.
        every column except the label col is considered a predictor
    post_df : pyspark.sql.DataFrame
        dataframe after matching occured. each row should be a sample.
        every column except the label col is considered a predictor
    label_col : str


    Returns
    -------
    bias_df : pd.DataFrame
        each row is a predictor and the columns are normal and
        standard pre/post bias. Also includes absolute and relative
        bias reduced
    total_bias_reduced : float
        1 - (sum postbias of features/ sum rebias of features)
    starting_bias_mean : float
        mean of prebias
    starting_bias_var : float
        var of prebias


    Raises
    ------
    AssertionError
        when either the pre or post df do not have any positive samples
    UncaughtExceptions

    Notes
    -----
    some calculations are duplicated e.g. when calculating bias and
    standard bias, root bias calculations are done twice. this has not been
    removed for ease of maintenance but an opportunity for performance
    increase should the need arise.

    See Also
    --------
    _calc_standard_bias_reduced
    _calc_bias_reduced
    _calc_standard_bias
    _calc_bias
    _calc_var
    """
    _persist_if_unpersisted(pre_df)
    if pre_df.where(F.col(label_col) == 1).count() <= 0:
        logging.getLogger(__name__).critical("somehow dont have positive samples in pre_df, this shouldnt happen")
        raise ValueError('no positive samples in pre_df')

    _persist_if_unpersisted(post_df)
    if post_df.where(F.col(label_col) == 1).count() <= 0:
        logging.getLogger(__name__).critical("somehow dont have positive samples in post_df, this shouldnt happen")
        raise ValueError("no positive samples in post df")

    sampled_pre_df = _sample_df(df=pre_df, sample_size=sample_size)
    _persist_if_unpersisted(sampled_pre_df)

    sampled_post_df = _sample_df(df=post_df, sample_size=sample_size)
    _persist_if_unpersisted(sampled_post_df)

    stan_bias_red_df, total_bias_reduced = _calc_standard_bias_reduced(pre_df, post_df, label_col)
    logging.getLogger(__name__).info("total bias reduced: {tbr:.2f}".format(tbr=total_bias_reduced))

    bias_red_df, starting_bias_mean, starting_bias_var = _calc_bias_reduced(pre_df, post_df, label_col)
    bias_df = bias_red_df.join(stan_bias_red_df, how='outer')
    logging.getLogger(__name__).info("starting mean: {sbm:,.2f}   starting var: {sbv:,.2f}".format(sbm=starting_bias_mean, sbv=starting_bias_var))

    return bias_df, total_bias_reduced, starting_bias_mean, starting_bias_var


@_time_log
def _calc_bias(df: DataFrame,
               label_col: str) -> pd.DataFrame:
    r""" given labels and features, calculate bias for each feature


    Parameters
    ----------
    df : pyspark.sql.DataFrame
        df where rows are observations, all columns except `label_col` are
        predictors. note: label col must be numeric binary 1 or 0
    label_col : str

    Returns
    -------
    bias_df : pd.DataFrame
        pandas dataframe where predictors are index and only column is
        bias

    Raises
    ------
    UncaughtExceptions

    Notes
    -----
    bias is defined as mean(x | label=1) - mean(x | label=0)
    """
    bias_df = df.groupby(label_col).mean().toPandas().transpose()
    bias_df['bias'] = bias_df[1] - bias_df[0]
    bias_df = bias_df.reset_index()
    bias_df['index'] = bias_df['index'].str.replace(r'avg\(', '').str.replace(r')', '')
    bias_df = bias_df.set_index('index')[['bias']]
    bias_df = bias_df.loc[bias_df.index != 'label', :]
    return bias_df


@_time_log
def _calc_standard_bias(df: DataFrame,
                        label_col: str) -> pd.DataFrame:
    r"""given labels and features, calculate standard bias for each feature


    Parameters
    ----------
    df : pyspark.sql.DataFrame
        df where rows are observations, all columns except `label_col` are
        predictors. note: label col must be numeric binary 1 or 0
    label_col : str

    Returns
    -------
    bias_df : pd.DataFrame
        pandas dataframe where predictors are index and only column is
        standard bias

    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _calc_var
    _calc_bias

    Notes
    -----
    standard bias is defined as
    (B1-B0)/(((VAR0+VAR1)/2)**.5)
    """
    var_df = _calc_var(df, label_col)
    bias_df = _calc_bias(df, label_col)

    bias_red_df = var_df.join(bias_df)
    bias_red_df['denominator'] = np.sqrt((bias_red_df['var_1'] + bias_red_df['var_0']) / 2)
    bias_red_df['standard_bias'] = bias_red_df['bias'] / bias_red_df['denominator'] * 100
    bias_red_df = bias_red_df[['standard_bias']]
    return bias_red_df


@_time_log
def _calc_var(df: pyspark.sql.DataFrame,
              label_col: str) -> pd.DataFrame:
    r"""calculate variance for each column that isnt the label_col

     Parameters
    ----------
    df : pyspark.sql.DataFrame
        df where rows are observations, all columns except `label_col` are
        predictors.
    label_col : str

    Returns
    -------
    bias_df : pd.DataFrame
        pandas dataframe where predictors are index and only column is
        variance

    Raises
    ------
    UncaughtExceptions

    Notes
    -----


    """
    pred_cols = [x for x in df.columns if x != label_col]
    s_var_df = df.groupby(label_col).agg({x: 'variance' for x in pred_cols}).toPandas().transpose()
    s_var_df = s_var_df.reset_index()
    s_var_df['index'] = s_var_df['index'].str.replace(r')', '').str.replace(r'variance\(', '')
    s_var_df = s_var_df.set_index('index')
    s_var_df.columns = ["var_{0}".format(x) for x in s_var_df.columns]
    s_var_df = s_var_df.loc[s_var_df.index != 'label', :]
    return s_var_df


@_time_log
def _calc_bias_reduced(pre_df: DataFrame,
                       post_df: DataFrame,
                       label_col: str) -> Tuple[pd.DataFrame, float, float]:
    r"""find the pre/post bias and their difference

    Parameters
    ----------
    pre_df : pyspark.sql.DataFrame
        dataframe before matching occured. each row should be a sample.
        every column except the label col is considered a predictor
    post_df : pyspark.sql.DataFrame
        dataframe after matching occured. each row should be a sample.
        every column except the label col is considered a predictor
    label_col : str


    Returns
    -------
    bias_df : pd.DataFrame
        each row is a predictor and the columns are pre-bias, post-bias,
        absolute bias reduced, and relative bias reduced
    starting_bias_./mean : float
        mean of prebias
    starting_bias_var : float
        var of prebias

    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _calc_bias

    Notes
    -----

    """
    pre_bias = _calc_bias(pre_df, label_col)
    pre_bias.columns = ['pre_bias']

    post_bias = _calc_bias(post_df, label_col)
    post_bias.columns = ['post_bias']

    b_red_df = pre_bias.join(post_bias, how='outer')
    b_red_df['bias_reduced_absolute'] = b_red_df['pre_bias'] - b_red_df['post_bias']
    b_red_df['bias_reduced_relative'] = b_red_df['bias_reduced_absolute'] / b_red_df['pre_bias'] * 100

    # output of pandas method may need to be converted to native float
    starting_bias_mean = float(b_red_df['pre_bias'].mean())
    starting_bias_var = float(b_red_df['pre_bias'].var())

    return b_red_df, starting_bias_mean, starting_bias_var


@_time_log
def _calc_standard_bias_reduced(pre_df: DataFrame,
                                post_df: DataFrame,
                                label_col: str) -> Tuple[pd.DataFrame, float]:
    r"""given the pre and post match dfs, calculate the standard bias for
    each, and how much was reduced.


    Parameters
    ----------
    pre_df : pyspark.sql.DataFrame
        dataframe before matching occured. each row should be a sample.
        every column except the label col is considered a predictor
    post_df : pyspark.sql.DataFrame
        dataframe after matching occured. each row should be a sample.
        every column except the label col is considered a predictor
    label_col : str
        note: label column in dfs should be binary numeric 1 or 0

    Returns
    -------
    bias_df : pd.DataFrame
        each row is a predictor and the columns are
        standard pre/post bias. Also includes absolute and relative
        standard bias reduced
    total_bias_reduced : float
        measure of total relative bias reduced.
        1- (sum of post-bias)/(sum or pre bias)

    Raises
    ------
    UncaughtExceptions


    See Also
    --------
    _calc_standard_bias

    Notes
    -----

    """
    pre_standard_bias = _calc_standard_bias(pre_df, label_col)
    pre_standard_bias.columns = ['pre_standard_bias']

    post_standard_bias = _calc_standard_bias(post_df, label_col)
    post_standard_bias.columns = ['post_standard_bias']

    stan_bias_red_df = pre_standard_bias.join(post_standard_bias, how='outer')
    stan_bias_red_df['standard_bias_reduced_absolute'] = stan_bias_red_df['pre_standard_bias'] \
                                                         - stan_bias_red_df['post_standard_bias']
    stan_bias_red_df['standard_bias_reduced_relative'] = stan_bias_red_df['standard_bias_reduced_absolute'] \
                                                         / stan_bias_red_df['pre_standard_bias'] * 100

    total_bias_reduced = (1 - stan_bias_red_df.post_standard_bias.abs().sum() / stan_bias_red_df.pre_standard_bias.abs().sum()) * 100

    # output of pandas method may need to be converted to native float
    total_bias_reduced = float(total_bias_reduced)
    return stan_bias_red_df, total_bias_reduced
