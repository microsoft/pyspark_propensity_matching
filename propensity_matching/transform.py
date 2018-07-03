import logging
import math
from typing import  Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import pyspark
from pyspark.sql import DataFrame
import pyspark.ml.classification as mlc
import pyspark.sql.functions as F
import pyspark.sql.types as T



from .utils import _persist_if_unpersisted, _time_log, _sample_df
from .config import UTIL_BOOST_THRESH_1, UTIL_BOOST_THRESH_2, UTIL_BOOST_THRESH_3,\
                    SMALL_MATCH_THRESHOLD, MINIMUM_POS_COUNT


@_time_log
def transform(df: DataFrame,
              prob_mod: mlc.Model,
              method: Optional[str] = None,
              metric: Optional[str] = None,
              match_kwargs: Optional[dict] = None) ->Tuple[DataFrame, dict]:

    r"""Create a propensity matched dataset

    Public entry point point for transform. Depending on arguments
    will point to the appropriate metric and matching methodology

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe with population in question. Must have featureCol and
        labelCol used in `prob_mod`
    prob_mod : mlc.Model
        the model predicting the probability that the row is in class 1
        in the label col.
    method : {'auto', 'quantile', 'assignment'}
        how matching occurs. auto will select according to the number of
        rows specified in config as `SMALL_MATCH_THRESHOLD`

        Quantile does stratified sampling on predicted probability.
        It guarantees similar population sizes and may drop some treatments
        non-symmetrically in order to fulfill that guarantee. match_info
        contains 'scale', what proportion of treatment users were used, and
        'dropped', proportion of sample dropped asymmetrically. The
        algorithm tries to maintain a balance between sample size and
        bias in deciding scale and droppped
    metric : {'probability'}
        the number that is being matched. Currently only support predicted
        probability but may add more in the future
    match_kwargs : dict, optional
        additional kwargs for match algorithm.
        

    Returns
    -------
    df : pyspark.sql.DataFrame
        df with only matched populations (so dont overwrite your parent
        dataframe if you need it!)
    match_info : dict
        information about that particular match depending on the algorithm
        chosen. 

    Raises
    ------
    UncaughtExceptions


    Notes
    -----
    Currently just a wrapper for _transform. Implemented like this so
    multiple potential _transform outputs could be run and the best one
    returned by transform. Collecting, coordinating, and evaluation those
    _transform outputs would be done in transform.
    """
    df, match_info = _transform(df, prob_mod, method, metric, match_kwargs)
    return df, match_info


@_time_log
def _transform(df: DataFrame,
               prob_mod: mlc.Model,
               method: Optional[str],
               metric: Optional[str],
               match_kwargs: Optional[dict] = None) -> Tuple[DataFrame, dict]:
    r""" execute one propensity match transform

    based on input vars, grab match col and run through algorithm to
    produce matched populations.


    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe with population in question. Must have featureCol and
        labelCol used in `prob_mod`
    prob_mod : mlc.Model
        the model predicting the probability that the row is in class 1
        in the label col.
    method : {'auto', 'quantile', 'assignment'}
        how matching occurs. auto will select according to the number of
        rows specified in config as `SMALL_MATCH_THRESHOLD`

        Quantile does stratified sampling on predicted probability.
        It guarantees similar population sizes and may drop some treatments
        non-symmetrically in order to fulfill that guarantee. match_info
        contains 'scale', what proportion of treatment users were used, and
        'dropped', proportion of sample dropped asymmetrically. The
        algorithm tries to maintain a balance between sample size and
        bias in deciding scale and droppped
    metric : {'probability'}
        the number that is being matched. Currently only support predicted
        probability but may add more in the future
    match_kwargs : dict, optional
        additional kwargs for match algorithm.


    Returns
    -------
    df : pyspark.sql.DataFrame
        df with only matched populations ( so dont overwrite your parent
        dataframe if you need it!)
    match_info : dict
        information about that particular match depending on the algorithm
        chosen.

    Raises
    ------
    UncaughtExceptions
    NotImplementedError
        illegal values for `method` and `metric`.

    See Also
    --------
    _get_metric
    _match

    Notes
    -----

    """

    # interpret input args:
    # only support quantile or assignment matching right now
    if match_kwargs is None:
        match_kwargs = {}

    logging.getLogger(__name__).info("method is {method}".format(method=str(method)))

    if method is None:
        method = 'auto'
        logging.getLogger(__name__).info("assigning default arg 'auto'")
    elif method not in ['assignment', 'quantile', 'auto']:
        logging.getLogger(__name__).critical("invalid method argument")
        raise NotImplementedError("method {method} not implemented".format(method=method))
    if method == 'auto':
        label_col = prob_mod.getOrDefault('labelCol')

        _persist_if_unpersisted(df)
        pos_count = df.where(F.col(label_col) == 1).count()
        neg_count = df.where(F.col(label_col) == 0).count()
        if ((pos_count**2)*neg_count) <= SMALL_MATCH_THRESHOLD:
            method = 'assignment'
            logging.getLogger(__name__).info("auto method is assignment")
        else:
            method = 'quantile'
            logging.getLogger(__name__).info("auto method is quantile")

    logging.getLogger(__name__).info("metric is {metric}".format(metric=str(metric)))
    if metric is None:
        metric = 'probability'
        logging.getLogger(__name__).info("assigning default metric 'probability'")
    elif metric not in ['probability']:
        logging.getLogger(__name__).critical("invalid metric argument")
        raise NotImplementedError("metric {metric} not implemented".format(metric=metric))

    # step 1 calculate match metric
    df, metric_col = _get_metric(df, prob_mod, metric)
    # step 2 match
    df, match_info = _match(df, prob_mod, method, metric_col, match_kwargs)

    return df, match_info


@_time_log
def _get_metric(df: DataFrame,
                prob_mod: pyspark.ml.Model,
                metric: str) ->Tuple[DataFrame, str]:
    r"""create the to-be-matched column


    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe with population in question. Must have featureCol and
        labelCol used in `prob_mod`
    prob_mod : mlc.Model
        the model predicting the probability that the row is in class 1
        in the label col.
    metric : {'probability'}
        the number that is being matched. Currently only support predicted
        probability but may add more in the future

    Returns
    -------
    df : pyspark.sql.DataFrame
        input dataframe but with new metric column
    metric_col : str
        name of the metric column now in `df`

    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _get_probability

    Notes
    -----
    just an entry point. right now only points to _get_probability
    but will be useful once more metrics are added

    """
    functions_dict = {'probability': _get_probability}
    df, metric_col = functions_dict[metric](df, prob_mod)
    return df, metric_col


@_time_log
def _get_probability(df: DataFrame,
                     prob_mod: pyspark.ml.Model) -> Tuple[DataFrame, str]:
    r"""given a df w/ featureCol, binary 0,1 labelCol and a model, predict
    probability of being in class 1 & return w/ df

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe with population in question. Must have featureCol and
        labelCol used in `prob_mod`
    prob_mod : mlc.Model
        the model predicting the probability that the row is in class 1
        in the label col.
    Returns
    -------
    df : pyspark.sql.DataFrame
        input dataframe but with new metric column
    prob_1_col : str
        name of the metric column now in `df`

    Raises
    ------
    UncaughtExceptions


    Notes
    -----

    """

    _persist_if_unpersisted(df)
    scored_df = prob_mod.transform(df)
    prob_col = prob_mod.getOrDefault('probabilityCol')
    prob_1_col = prob_col + "_1"
    scored_df = scored_df.withColumn(prob_1_col, F.udf(lambda x: float(x[1]), T.FloatType())(F.col(prob_col)))
    return scored_df, prob_1_col


@_time_log
def _match(df: DataFrame,
           prob_mod: mlc.Model,
           method: str,
           metric_col: str,
           match_kwargs: Optional[dict] = None):
    r"""private entry point for matching

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Dataframe with population in question. Must have featureCol and
        labelCol used in `prob_mod`
    prob_mod : mlc.Model
        the model predicting the probability that the row is in class 1
        in the label col.
    method : {'auto', 'quantile', 'assignment'}
        how matching occurs. auto will select according to the number of
        rows specified config as `SMALL_MATCH_THRESHOLD`

        Quantile does stratified sampling on predicted probability.
        It guarantees similar population sizes and may drop some treatments
        non-symmetrically in order to fulfill that guarantee. match_info
        contains 'scale', what proportion of treatment users were used, and
        'dropped', proportion of sample dropped asymmetrically. The
        algorithm tries to maintain a balance between sample size and
        bias in deciding scale and droppped
    metric_col : str
        the number that is being matched.
    match_kwargs : dict, optional
        additional kwargs for match algorithm.
        

    Returns
    -------
    df : pyspark.sql.DataFrame
        df with only matched populations ( so dont overwrite your parent
        dataframe if you need it!)
    match_info : dict
        information about that particular match depending on the algorithm
        chosen. 

    Raises
    ------
    UncaughtExceptions


    See Also
    --------
    _quantile_match
    _assignment_match

    """

    functions_dict = {
        'assignment': _assignment_match,
        'quantile': _quantile_match
        }
    # _assignment_match doesnt currently have any kwargs, so match_kwargs should be empty
    df, match_info = functions_dict[method](df, prob_mod, metric_col, **match_kwargs)

    return df, match_info


@_time_log
def _quantile_match(df: DataFrame,
                    prob_mod: mlc.Model,
                    metric_col: str,
                    ntile: int = 10,
                    quantile_error_scale: int = 5,
                    sample_size: int = 10**5) -> Tuple[DataFrame, dict]:
    r"""match by stratified sampling on probability bins. guarantee similar
    populations.

    match by stratified sampling on probability bins. guarantee similar
    populations. may scale treatment curve down and drop treatments
    unevenly to uphold guarantee

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    prob_mod : pyspark.ml.classification.Model
    metric_col : str
        name of col to be matched
    ntile : int
        how many buckets to make out of the metric col and then
        stratify sample
        defaults to 10
    quantile_error_scale: Union[int, float]
        error tolerance for calculating boundries for ntiles
        relativeError passed to approxQuantile is calculated as
        1/ntile/quantile_error_scale
        in other words 1/quantile_error_scale is how much error is ok
        as a fraction of the bin size
        be cognizant of ntile, and this value, as passing a small
        relativeError can increase compute time dramatically
        defaults to 5
    sample_size: Optional[int]
        size of sample used to calculate quantile bin boundaries
        no sampling if None, not recommended
        defauts to 10**5


    Returns
    -------
    df
        Explanation of anonymous return value of type ``type``.
    match_info : dict
        contains scale and dropped
        scale describes what proportion of the treatment group was used
        and dropped describes what proportion of the treatment group, after
        scaling, was dropped due to inadequate control candidates
        Explanation

    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _make_quantile_match_col
    _execute_quantile_match

    Notes
    -----

    """
    logging.getLogger(__name__).info("starting _quantile_match with args ntile={ntile}, quantile_error_scale={qes}, /"
                                     "sample_size={sn}".format(ntile=ntile, qes=quantile_error_scale, sn=sample_size))

    label_col = prob_mod.getOrDefault('labelCol')
    df, match_col = _make_quantile_match_col(df, metric_col, label_col, ntile, quantile_error_scale, sample_size)
    df, match_info = _execute_quantile_match(df, match_col, label_col)
    match_info['type'] = 'quantile'

    return df, match_info


@_time_log
def _make_quantile_match_col(df: DataFrame,
                             metric_col: str,
                             label_col: str,
                             ntile: int,
                             quantile_error_scale: Optional[Union[int, float]],
                             sample_size: Optional[int]) -> Tuple[DataFrame, str]:
    r"""bin probability column and return it to be matched


    Parameters
    ----------
    df : pyspark.sql.DataFrame
    metric_col : str
        name of col to be matched
    ntile : int
        how many buckets to make out of the metric col and then
        stratify sample
        defaults to 10
    quantile_error_scale: Union[int, float]
        error tolerance for calculating boundries for ntiles
        relativeError passed to approxQuantile is calculated as
        1/ntile/quantile_error_scale
        in other words 1/quantile_error_scale is how much error is ok
        as a fraction of the bin size
        be cognizant of ntile, and this value, as passing a small
        relativeError can increase compute time dramatically
    sample_size: Optional[int]
        size of sample used to calculate quantile bin boundaries
        no sampling if None, not recommended


    Returns
    -------
    df : pyspark.sql.DataFrame
        input df but with `match_col`
    match_col : type
        colname of col to be matched
        Explanation

    Raises
    ------
    UncaughtException

    See Also
    --------
    _quantile_match: calls it, sets default args for `ntile`, `sample_size`,
        and `quantile_error_scale`
    """

    t_df = df.where(F.col(label_col) == 1)
    _persist_if_unpersisted(t_df)

    t_sample_df = _sample_df(df=t_df, sample_size=sample_size)
    _persist_if_unpersisted(t_sample_df)
    # create thresholds for ntiles, convert to native float from numpy float for use w/ pyspark
    probs = [float(x) for x in np.linspace(start=0, stop=1, num=ntile, endpoint=False)][1:]

    quantile_error_tolerance = 1/ntile/quantile_error_scale
    threshs = t_sample_df.approxQuantile(col=metric_col, probabilities=probs, relativeError=quantile_error_tolerance)

    def make_udf(threshs):
        return F.udf(lambda x: sum([x > y for y in threshs]), T.ShortType())

    # add 1 to conform to mathematical indexing of ntiling
    _persist_if_unpersisted(df)
    match_col = "quantile_match_col_{metric_col}".format(metric_col=metric_col)
    df = df.withColumn(match_col, make_udf(threshs)(F.col(metric_col)) + 1)

    return df, match_col


@_time_log
def _execute_quantile_match(df: DataFrame,
                            match_col: str,
                            label_col: str) ->Tuple[DataFrame, dict]:

    r"""stratified sample and return matched populations & match_info

    _calc_sample_fracs handles logic for scale/drop

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    match_col : str
    label_col : str

    Returns
    -------
    df : pyspark.sql.DataFrame
    match_info : dict

    Raises
    ------
    Uncaught Exceptions

    See Also
    --------
    _calc_sample_fracs
    _sample_dfs

    Notes
    -----

    """

    t_df = df.where(F.col(label_col) == 1)
    c_can_df = df.where(F.col(label_col) == 0)

    t_fracs, c_fracs, scaled, dropped = _calc_sample_fracs(t_df, c_can_df, match_col)
    t_out, c_out = _sample_dfs(t_df, t_fracs, c_can_df, c_fracs, match_col)
    df = t_out.union(c_out.select(t_out.columns))
    match_info = {'scaled': scaled, 'dropped': dropped}

    return df, match_info


@_time_log
def _calc_sample_fracs(t_df: DataFrame,
                       c_can_df: DataFrame,
                       match_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:

    r"""given treatment and control_candidates, calculate optimal
    stratified sample fractions for balance population

    `scale` is using fewer treatment all across the board, while `drop`
    is dropping specific treatments without adequate controls.
    _calc_sample_fracs tries to balance sample size (decreased by scale)
    and validity (decreased by unbalanced dropping)


    Parameters
    ----------
    t_df : pyspark.sql.DataFrame
        the treatment group
    c_can_df : pyspark.sqlDataFrame
        the control candidates
    match_col : str
        binned col to be matched

    Returns
    -------
    pd.DataFrame
        the sample fractions for the treatment group w/ `match_col` and
        fraction columns
    pd.DataFrame
        the sample fractions for the control group.
    float
        the scale
    float
        the drop

    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _calc_optimal_subset

    Notes
    -----

    """
    _persist_if_unpersisted(t_df)
    _persist_if_unpersisted(c_can_df)

    t_counts = t_df.groupby(match_col).count().withColumnRenamed('count', 'treatment')
    c_can_counts = c_can_df.groupby(match_col).count().withColumnRenamed('count', 'control')
    fracs = t_counts.join(c_can_counts, on=[match_col])
    fracs = fracs.toPandas()
    sample_fracs, scale, drop = _calc_optimal_subset(fracs=fracs, match_col=match_col)
    logging.getLogger(__name__).info("scale = {scale:.2f}     drop: = {drop:.2f}".format(scale=scale, drop=drop))


    return sample_fracs[[match_col, 'treatment_scaled_sample_fraction']],\
           sample_fracs[[match_col, 'control_scaled_sample_fraction']],\
           scale, drop


@_time_log
def _calc_optimal_subset(fracs: pd.DataFrame,
                         match_col: str) -> Tuple[pd.DataFrame, float, float]:
    r""" return best sample fractions for given population

    iterate over range of possible scales and drops & assign utility
    return one of the fraction with max utility

    Parameters
    ----------
    fracs : pd.DataFrame
        columns `match_col` and 'treatment' and 'control' (
    match_col : str


    Returns
    -------
    pd.DataFrame
        columns `match_col`, 'treatment_scaled_sample_fraction',
        'control_scaled_sample_fraction'

    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _calc_sample_fracs
    _create_options_grid

    Notes
    -----
    """

    fracs = fracs.copy(deep=True)
    fracs['control_sample_fraction_naive'] = fracs['treatment']/fracs['control']
    scale_factor = fracs.control_sample_fraction_naive.max()**-1
    logging.getLogger(__name__).info("scale factor  is {scale_factor:.2f} (coeffs for treatment w/ no drops".format(scale_factor=scale_factor))

    # if no subscaling is necessary return fracs as is
    if scale_factor >= 1:
        logging.getLogger(__name__).info("can use all treatments safely, returning early")
        fracs['control_scaled_sample_fraction'] = fracs['control_sample_fraction_naive']
        fracs['treatment_scaled_sample_fraction'] = 1
        fracs = fracs[[match_col, 'treatment_scaled_sample_fraction', 'control_scaled_sample_fraction']]
        return fracs, float(1), float(0)

    options = _create_options_grid(fracs, scale_factor)
    options['utility'] = options.apply(_calc_util_wrapper, axis=1)

    # pick best
    max_util = options.utility.max()
    best_row = options[options.utility == max_util].iloc[0]
    winning_scale = float(best_row['scale'])
    winning_drop = float(best_row['percent_dropped'])

    logging.getLogger(__name__).info("max_util:{mu:.2f}\twinning_scale:{ws:.2f}\twinning_drop:{wd:.2f}".format(mu=max_util, ws=winning_scale, wd=winning_drop))

    fracs['control_scaled_sample_fraction'] = np.min([(fracs['treatment'] * winning_scale/fracs['control']).values, [1]*len(fracs)], axis=0)
    fracs['treatment_scaled_sample_fraction'] = fracs['control_scaled_sample_fraction'] * fracs['control']/fracs['treatment']
    fracs = fracs[[match_col, 'treatment_scaled_sample_fraction', 'control_scaled_sample_fraction']]

    return fracs, winning_scale, winning_drop


@_time_log
def _create_options_grid(fracs: pd.DataFrame, scale_factor: float) -> pd.DataFrame:
    r"""create 100 scale options & calc drop

    Parameters
    ----------
    fracs: pd.DataFrame
        Array_like means all those objects -- lists, nested lists, etc. --
        that can be converted to an array.  We can also refer to
        variables like `var1`.
    scale_factor: float
        the multiplication factor if no treatment were dropped (e.g. the
        fraction of treatment at which dropping is no longer neccesary
        because the number of control candidates at each match col
        bucket is greater than or equal to the treatments)


    Returns
    -------
    fracs : pd.DataFrame
        columns 'scale', 'percent_dropped', and 'number' (count of
        treatment)


    Raises
    ------
    UncaughtExceptions

    """
    fracs = fracs.copy(deep=True)
    scales = np.linspace(1, scale_factor, num=100, endpoint=True)
    options = pd.DataFrame(columns=['scale', 'percent_dropped', 'number'])

    for scale in scales:
        # calc new frac samples, maximum of 1
        fracs['control_scaled_sample_fraction'] = np.min([(fracs['treatment'] * scale/fracs['control']).values, [1]*len(fracs)], axis=0)
        fracs['treatment_scaled_sample_fraction'] = fracs['control_scaled_sample_fraction'] * fracs['control']/fracs['treatment']

        # calc %drop as difference of scale and actual ( e.g. where we pinned max at 1 in control scaled sample fraction)
        num_dropped = (fracs['treatment'] * (np.array([scale] * len(fracs)) - fracs['treatment_scaled_sample_fraction'])).sum()
        percent_dropped = num_dropped/(fracs['treatment'] * scale).sum()

        # calc new total
        number = (fracs['treatment']*fracs['treatment_scaled_sample_fraction']).sum()
        options = options.append({'scale': scale, 'percent_dropped': percent_dropped, 'number': number}, ignore_index=True)

    return options


def _calc_util_wrapper(row):
    return _calc_util(row['number'], row['percent_dropped'])


def _calc_util(number, dropped):
    # base utility
    log_value = math.log10(number/1000 + 1)
    # additional incentive to hit certain thresholds - chosen due to implementation factors
    threshold_boost = _logistic_function(L=math.log10(number / UTIL_BOOST_THRESH_1 + 1) / 10, x=number, x0=UTIL_BOOST_THRESH_1) \
                      + _logistic_function(L=math.log10(number / UTIL_BOOST_THRESH_2 + 1) / 10, x=number, x0=UTIL_BOOST_THRESH_2) \
                      + _logistic_function(L=math.log10(number / UTIL_BOOST_THRESH_3 + 1) / 10, x=number, x0=UTIL_BOOST_THRESH_3)
    # discount for unbalanced dropping
    dropped_penalty = 1-min(math.exp(dropped)-1, 1)
    utility = dropped_penalty * (log_value + threshold_boost)
    return utility


def _logistic_function(x, L, k=1, x0=0):
    try:
        return L / (1 + math.exp(-k * (x - x0)))
    except OverflowError:
        if x >= x0:
            return L
        if x < x0:
            return 0


@_time_log
def _sample_dfs(t_df: pyspark.sql.DataFrame,
                t_fracs: pd.DataFrame,
                c_can_df: pyspark.sql.DataFrame,
                c_fracs: pd.DataFrame,
                match_col: str) ->Tuple[DataFrame, DataFrame]:
    r"""given treatment and control pops and their stratified sample
    fracs, return balanced pops

    Parameters
    ----------
    t_df : pyspark.DataFrame
        treatment pop
    t_fracs: pd.DataFrame
        with columns `match_col` and 'treatment_scaled_sample_fraction'
    c_can_df : pyspark.DataFrame
        control can pop
    c_fracs : pd.DataFrame
        with columns `match_col` and control_scaled_sample_fraction

    Returns
    -------
    t_out : pyspark.sql.DataFrame
    c_out : pyspark.sql.DataFrame

    Raises
    ------
    UncaughtExceptions

    """
    _persist_if_unpersisted(t_df)
    _persist_if_unpersisted(c_can_df)

    t_fracs = t_fracs.set_index(match_col).treatment_scaled_sample_fraction.to_dict()
    t_dict = {}
    for key, value in t_fracs.items():
        t_dict[int(key)] = min(float(value), 1)
    t_out = t_df.sampleBy(col=match_col, fractions=t_dict, seed=42)

    c_fracs = c_fracs.set_index(match_col).control_scaled_sample_fraction.to_dict()
    c_dict = {}
    for key, value in c_fracs.items():
        c_dict[int(key)] = float(value)
    c_out = c_can_df.sampleBy(col=match_col, fractions=c_dict, seed=42)

    return t_out, c_out


@_time_log
def _assignment_match(df: DataFrame,
                      prob_mod: mlc.Model,
                      metric_col: str) ->Tuple[DataFrame, dict]:
    r"""match treatment to controls 1:1

    Use Hungarian/Munkres algorithm in `metric_col` (typically probability)
    to find controls for your treatments with the least cost - the distance
    between a treatment's metric and its control's metric

    Parameters
    ----------
    df: DataFrame
        dataframe in question, must have input columns specified by
        prob_mod
    prob_mod: mlc.Model
        propenisty predicting model. used here mostly to grab label/feature
        columns. metric col should have been constructed by another method
        prior
    metric_col: str
        the column values to matched on 
    Returns
    -------
    df
        new dataframe with just the matched population
    match_info: dict
        dict of potentially handy metrics from the matching process
            scaled: proportion of treatments used in matching
            init_t_mean: mean treatment val of treatment candidates
            init_c_can_mean: mean metric val of control candidates
            init_t_count: number of treatments in input
            init_c_can_count: number of control candidates
            adj_t_count: number of treatments after adjusting population
                size to accomodate difference in probability distribution
                between treatment and control.
            total_cost : total_cost
            average_cost : average_cost
            dropped : frac of treatments dropped unbalanced

    Raises
    ------
    ValueError
        when the treatment population is too small and or unabalanced
        to produce a good match
    UncaughtExceptions

    See Also
    --------
    _adjust_balance
    _make_cost_matrix
    _execute_assignment_match
    _get_assigned_rows

    Notes
    -----
    In order to produce good matching, if the probability distributions
    are significantly different (i.e. treatment is significantly right
    shifted), the control candidate pop needs to be much greater.
    _adjust_balance achieves this by taking
    max num_control_candidates/(treatment_mean/control_can_mean). this
    will progressively make the treatment pop smaller as the right
    shift becomes greater. However, this created the danger of being left
    with a very small treatment population from which conclusions shouldnt
    be drawn. additionaly this method was devised, and not taken from a
    peer reviewed paper. This is a point which merits further investigation
    before it is considered canon.

    """

    label_col = prob_mod.getOrDefault('labelCol')
    t_df = df.where(F.col(label_col) == 1)
    c_can_df = df.where(F.col(label_col) == 0)

    t_adjusted_df, c_can_adjusted_df, match_info = _adjust_balance(t_df, c_can_df, metric_col)

    t_vals = t_adjusted_df.select(metric_col).toPandas()
    c_can_vals = c_can_adjusted_df.select(metric_col).toPandas()

    cost_matrix = _make_cost_matrix(t_vals=t_vals, c_can_vals=c_can_vals,)

    c_ind, t_ind, total_cost, average_cost = _execute_assignment_match(cost_matrix)
    match_info['total_cost'] = total_cost
    match_info['average_cost'] = average_cost
    df = _get_assigned_rows(t_ind=t_ind, t_df=t_adjusted_df, c_ind=c_ind, c_can_df=c_can_adjusted_df)

    logging.getLogger(__name__).info("matched df size is {n:,}".format(n=df.count()))
    match_info['dropped'] = 0

    return df, match_info


@_time_log
def _adjust_balance(t_df: DataFrame,
                    c_can_df: DataFrame,
                    metric_col: str) -> Tuple[pyspark.sql.DataFrame, pyspark.sql.DataFrame, dict]:
    r"""given treatment, control can pops and the metric columns
        produce a class balance which can have a good matching result

    Parameters
    ----------
    t_df: pyspark.sql.DataFrame
        treatment group
    c_can_df: pyspark.sql.DataFrame
        the control candidate group
    metric_col: str
        name of the column in `t_df` and `c_can_df` whose difference
        will be the disutility during matching

    Returns
    -------
    t_adjusted_df: pyspark.sql.DataFrame
        treatment group - potentially scaled down to facilitate a better
        match
    c_can_df: pyspark.sql.DataFrame
        the control candidate group - passthrough
    match_info: dict
        dict with some potentially useful info about the pops and
        what processing was done
            scaled: proportion of treatments used in matching
            init_t_mean: mean treatment val of treatment candidates
            init_c_can_mean: mean metric val of control candidates
            init_t_count: number of treatments in input
            init_c_can_count: number of control candidates
            adj_t_count: number of treatments after adjusting population
                size to accomodate difference in probability distribution
                between treatment and control.


    Raises
    ------
    ValueError
        when the treatment population is too small and or unabalanced
        to produce a good match
    UncaughtExceptions

    See Also
    --------
    _assignment_match
    _make_cost_matrix
    _execute_assignment_match
    _get_assigned_rows

    Notes
    -----
    In order to produce good matching, if the probability distributions
    are significantly different (i.e. treatment is significantly right
    shifted), the control candidate pop needs to be much greater.
    _adjust_balance achieves this by taking
    max num_control_candidates/(treatment_mean/control_can_mean). this
    will progressively make the treatment pop smaller as the right
    shift becomes greater. However, this created the danger of being left
    with a very small treatment population from which conclusions shouldnt
    be drawn. additionaly this method was devised, and not taken from a
    peer reviewed paper. This is a point which merits further investigation
    before it is considered canon.


    """
    _persist_if_unpersisted(t_df)
    _persist_if_unpersisted(c_can_df)

    t_mean = t_df.select(F.col(metric_col)).agg({metric_col: 'mean'}).take(1)[0][0]
    c_can_mean = c_can_df.select(F.col(metric_col)).agg({metric_col: 'mean'}).take(1)[0][0]
    mean_ratio = t_mean/c_can_mean

    t_count = t_df.count()
    c_can_count = c_can_df.count()

    logging.getLogger(__name__).info("t_mean={tm:.2f}   c_can_mean={ccm:.2f}   mean_ratio={mr:.2f}   t_count={tc:.2f}   c_can_count={ccc:.2f}".format(
            tm=t_mean, ccm=c_can_mean, mr=mean_ratio, tc=t_count, ccc=c_can_count)
            )

    desired_t_count = c_can_count/mean_ratio
    if t_count > desired_t_count:
        logging.getLogger(__name__).info("adjustment of treatment necessary")
        frac = desired_t_count/t_count
        t_adjusted_df = t_df.sample(fraction=frac, withReplacement=False, seed=42)
        t_adj_count = t_adjusted_df.count()
        scaled = t_adj_count/t_count
        logging.getLogger(__name__).info("scaled: {s:.2f}   adjusted treatment count {tac:,}".format(s=scaled, tac=t_adj_count))

        if t_adj_count < MINIMUM_POS_COUNT:
            logging.getLogger(__name__).critical("treatment count {tac:,} less than MINIMUM_POS_COUNT {MPC:,}".format(tac=t_adj_count, MPC=MINIMUM_POS_COUNT))
            raise ValueError('population is too unbalanced to match')
    else:
        logging.getLogger(__name__).info("adjustment of treatment NOT necessary")
        t_adjusted_df = t_df
        scaled = 1
    t_adjusted_count = t_adjusted_df.count()
    match_info = {
        'scaled': scaled,
        'init_t_mean': t_mean,
        'init_c_can_mean': c_can_mean,
        'init_t_count': t_count,
        'init_c_can_count': c_can_count,
        'adj_t_count': t_adjusted_count
    }
    if t_adjusted_count > c_can_count:
        logging.getLogger(__name__).critical("more treatments thatn controls, this shouldnt have happened")
        raise ValueError("more treatments that controls")

    return t_adjusted_df, c_can_df, match_info


@_time_log
def _make_cost_matrix(t_vals: pd.Series,
                      c_can_vals: pd.Series) -> np.matrix:
    r"""given a series of values for treatment and control candidate
    make an i by j matrix of the cost for matching each treatment j
    with each control candidate i

    Parameters
    ----------
    t_vals: pd.Series
        metric values for the treatment group
    c_can_vals: pd.Series
        metric values for the control candidates

    Returns
    -------
    cost_matrix: np.matrix
        an i by j matrix of the cost for matching each treatment j
        with each control candidate i


    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _assignment_match
    _adjust_balance
    _execute_assignment_match
    _get_assigned_rows
    """

    t_vals_matrix = np.matrix(np.repeat(a=np.array([t_vals.values, ]), repeats=len(c_can_vals), axis=0))
    c_can_vals_matrix = np.matrix(np.repeat(a=np.array([c_can_vals.values, ]), repeats=len(t_vals), axis=0)).transpose()

    # rows are control can, columns are treatment
    cost_matrix = np.abs(t_vals_matrix-c_can_vals_matrix)
    return cost_matrix


@_time_log
def _execute_assignment_match(cost_matrix: np.matrix)->Tuple[np.ndarray, np.ndarray, float, float]:
    r"""given a cost matrix, use the hungarian algorithm to find the best
    solution to the assignmetn problem


    Parameters
    ----------
    cost_matrix: np.matrix
        an i by j matrix of the cost for matching each treatment j
        with each control candidate i


    Returns
    -------
    c_ind
        indices of the chosen controls
    t_ind : type
        indices of the chosen treatments ( typicall all of them)
    total_cost : float
        total difference in metric between treatments and chosen controls
    average_cost : float
        average difference in metric between treatment and chosen control

    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _assignment_match
    _adjust_balance
    _make_cost_matrix
    _get_assigned_rows

    """
    c_ind, t_ind = linear_sum_assignment(cost_matrix)
    total_cost = float(cost_matrix[c_ind, t_ind].sum())
    average_cost = float(total_cost/len(t_ind))

    logging.getLogger(__name__).info("chose {tnum:,} treatments and {cnum:,} controls".format(tnum=len(t_ind), cnum=len(c_ind)))
    logging.getLogger(__name__).info("total_cost:{tc:,.2f}   average_cost:{ac:.2f}".format(tc=total_cost, ac=average_cost))

    return c_ind, t_ind, total_cost, average_cost


@_time_log
def _get_assigned_rows(t_ind: np.ndarray,
                       t_df: DataFrame,
                       c_ind: np.ndarray,
                       c_can_df: DataFrame) -> DataFrame:
    r"""A one-line summary that does not use variable names or the
    function name.

    Several sentences providing an extended description. Refer to
    variables using back-ticks, e.g. `var`.

    Parameters
    ----------
    t_ind: np.ndarray
        indices of the chosen treatments. passthrough right now since
        they are all chosen (given logic in _adjust_balance)
    t_df: pyspark.sql.DataFrame
        dataframe of treatment group put through matching algorithm
    c_ind: np.ndarray
        indices of chosen control candidates
    c_can_df: pyspark.sql.DataFrame
        dataframe of control candidates put through matching algorithm

    Returns
    -------
    df: pyspark.sql.DataFrame
        the matched population all together

    Raises
    ------
    UncaughtExceptions

    See Also
    --------
    _assignment_match
    _adjust_balance
    _make_cost_matrix
    _execute_assignment_match

    """
    _persist_if_unpersisted(t_df)
    _persist_if_unpersisted(c_can_df)

    spark = pyspark.sql.SparkSession.builder.getOrCreate()

    c_bool_idx = [False] * c_can_df.count()
    for idx in c_ind:
        c_bool_idx[idx] = True
    c_bool_idx = zip(range(1, len(c_bool_idx) + 1), c_bool_idx)
    c_bool_idx = spark.createDataFrame(data=c_bool_idx)
    c_bool_idx = c_bool_idx.select(F.col('_1').alias('rownum'), F.col('_2').alias('chosen'))

    c_can_df = c_can_df.withColumn('monid', F.monotonically_increasing_id())
    c_can_df = c_can_df.withColumn('rownum', F.row_number().over(pyspark.sql.Window.orderBy('monid')))

    c_can_df = c_can_df.join(c_bool_idx, on='rownum', how='inner')
    c_matched_df = c_can_df.where(c_can_df.chosen)

    df = t_df.union(c_matched_df.select(t_df.columns))

    return df
