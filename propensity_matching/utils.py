"""Utility functions that may be useful at any stage."""

from collections import defaultdict
import math
import logging
import time
from typing import List, Tuple, Optional, Dict, Union, Callable

import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

from pyspark import StorageLevel
import pyspark.ml.feature as mlf
import pyspark.ml.classification as mlc
from pyspark.mllib import stat as mllibs
from pyspark.mllib.linalg import Vectors
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T

from .config import SAMPLES_PER_FEATURE


def _time_log(func: Callable):
    def _time_logger(*args, **kwargs):
        start_time = time.time()
        logging.getLogger(__name__).debug("starting {name}".format(name=func.__name__))
        out = func(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time)/60
        logging.getLogger(__name__).debug("finished {name}, took {duration:,.2f} minutes".format(name=func.__name__,
                                                                                                 duration=duration))
        return out
    return _time_logger

def _get_pred_cols(df: DataFrame, features_col: str, approved_types: Optional[List[str]]=None) -> List[str] :
    r""" given a dataframe and an assembled feature column, return the
    feature names in order

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        The dataframe in question.
    features_col : str
        colname, must be vector assembled and in df
    approved_types : List[str], optional
        list of types to return if they are in features_col
        defaults to ['numeric', 'binary']


    Returns
    -------
    pred_cols : List[str]
        list of columns names assembled into `features_col` in the order
        they were assembled
    """
    if approved_types is None:
        approved_types = ['numeric', 'binary']

    meta = [f.metadata
            for f in df.schema.fields
            if f.name == features_col][0]

    types = list(meta['ml_attr']['attrs'].keys())

    construction_list = list()

    for type in [x for x in types if x in approved_types]:
        construction_list += meta['ml_attr']['attrs'][type]

    construction_list_2 = [(x['idx'], x['name']) for x in construction_list]
    sorted_construction_list_2 = sorted(construction_list_2, key=lambda x: x[0])
    pred_cols = [x[1] for x in sorted_construction_list_2]

    return pred_cols


def _compare_stor_levels(first: StorageLevel, second: StorageLevel) -> bool:
    r""" see if two pyspark storage levels are equal

    __eq__ is not implemented for StorageLevel. this method goes through
    each attribute and compares, returning true if all are equal

    """
    atts = ['useDisk', 'useMemory', 'useOffHeap', 'deserialized', 'replication']
    for att in atts:
        if getattr(first, att) != getattr(second, att):
            return False
    return True


def _persist_if_unpersisted(df: DataFrame) -> bool:
    r"""if dataframe is completely unpersisted, set persistence to mem only
    return true if persistence is set, false if persistence was already set
    """

    if _compare_stor_levels(df.storageLevel, StorageLevel(False, False, False, False, 1)):
        logging.getLogger(__name__).info("df was unpersisted, persisting")
        df.cache()
        return True

    logging.getLogger(__name__).info("df was already persisted")
    return False

@_time_log
def _sample_df(df: DataFrame,
               sample_size: Optional[int]= None):
    _persist_if_unpersisted(df)
    all_count = df.count()
    if (sample_size is None) | (all_count <= sample_size):
        if isinstance(sample_size, int):
            logging.getLogger(__name__).info("no sampling necessary {c:,} <= {sn:,}".format(c=all_count, sn=sample_size))
        else:
            logging.getLogger(__name__).info("requested not to sample, df of size {c:,}".format(c=all_count))
        sample_df = df
    else:  # all_count > sample_size:
        frac = sample_size / all_count
        logging.getLogger(__name__).info("sampling {c:,} by {f:.2f} to {sn:,}".format(c=all_count, f=frac, sn=sample_size))
        sample_df = df.sample(withReplacement=False, fraction=frac, seed=42)
    return sample_df


@_time_log
def remove_redundant_features(df: DataFrame,
                              features_col: str,
                              sample_size: int = 10 ** 5,
                              method: str = 'ward',
                              cluster_thresh: float = .1,
                              ) -> Tuple[DataFrame, List[str]]:
    r"""Remove redundant or uninformative columns

    -Drop cols w/ 0 variance
    -Perform hierarchical agglomerative clustering on columns
        where method = `method` and distance = 1-spearman's corr
    -Create clusters w/ threshold `cluster_thresh`
        & select one column to represent whole cluster.


    Parameters
    ----------
    df : pyspark.sql.DataFrame
    pred_cols : List[str]
        List of columns that may be predictors. Must be numeric.
    features_col : str
        Name of column that will be the feature column.
        May already exist in df, in which case it will be dropped
    sample_size : int=10**5, optional
        size of sample df used to evaluate columns
    method : {'ward', 'single', 'average', 'weighted', median'} , optional
        see scipy.cluster.hierarchy.linkage for more
    cluster_thresh : float=.1
        distance threshold below which columns are considered a cluster
        may be interpreted as 1-correlation

    Returns
    -------
    df : pyspark.sql.DataFrame
        df with `features_col` assembled from chosen cols
    out_cols : List[str]
        list of chosen cols

    Raises
    ------
    Uncaught Exceptions

    See Also
    --------
    scipy.cluster.hierarchy : package with core functionality
    """

    # sample dataframe of max size sample_size
    logging.getLogger(__name__).info("removing redundant features with params sample_size={sn:,}   method={m}   "
                                     "cluster_thresh={ct:.2f}".format(sn=sample_size, m=method, ct=cluster_thresh))

    pred_cols = _get_pred_cols(df=df, features_col=features_col)


    sample_df = _sample_df(df=df, sample_size=sample_size)
    _persist_if_unpersisted(sample_df)

    # create dict of col:variance
    variances_frame = sample_df.select([F.variance(F.col(x)).alias(x) for x in pred_cols])
    variances_dict = variances_frame.toPandas().transpose().to_dict()[0]

    # exclude cols w/o variance (completely degenerate)
    cols = []
    for col, var in variances_dict.items():
        if var != 0:
            cols.append(col)
    logging.getLogger(__name__).info("{n:,} of {m:,} cols had variance and were kept".format(n=len(cols), m=len(variances_dict)))

    # prep rdd w/ where vals for each row are in a dense Vector
    corr_assembler = mlf.VectorAssembler(inputCols=cols, outputCol='corr_features')
    sample_df = corr_assembler.transform(sample_df)
    sample_df.persist(StorageLevel(False, True, False, False, 1))
    sample_rdd = sample_df.select('corr_features').rdd.map(lambda x: Vectors.dense(x[0]))

    # spearman corr for correlation grid because it's non parametric
    corr_out = mllibs.Statistics.corr(sample_rdd, method='spearman')
    # since corr is 1-distance, absolute value
    corr_out = np.abs(corr_out)
    # sometimes floating point will result in sliiiighlty above 1 corr
    corr_out[corr_out > 1] = 1
    # clustering distance metric = 1-corr
    corr_array = 1 - corr_out
    # diags must be 0 for np.squareform checks
    np.fill_diagonal(corr_array, 0)

    # filter out cols where all corr_array vals are null
    # should be redundant since cols w/ 0 var producing nulls were filtered earlier. left here for safety
    null_mask = np.logical_or(np.isnan(corr_array), corr_array == 0).all(axis=1)
    corr_array_n = corr_array[~null_mask, :][:, ~null_mask]
    colnames = np.array(cols)[~null_mask]

    # generate clusters
    condensed_matrix = squareform(corr_array_n, checks=True)
    Z = linkage(condensed_matrix, method=method)
    # databricks scipy version does not have optimal order arg, uncomment when they update
    # Z = linkage(condensed_matrix, method=method, optimal_ordering=True)
    clusters = fcluster(Z, cluster_thresh, criterion='distance')

    # extract 1 col from each group
    out_cols = []
    for cluster in np.unique(clusters):
        cluster_cols = colnames[clusters == cluster]
        # pick mid col to best represent whole cluster, needs optimal ordering in flcuster above
        cluster_col = cluster_cols[math.floor(len(cluster_cols) / 2)]
        out_cols.append(str(cluster_col))

    logging.getLogger(__name__).info("{n:,} cols/clusters chosen of {m:,} total".format(n=len(out_cols), m=len(cols)))
    feature_assembler = mlf.VectorAssembler(inputCols=out_cols, outputCol=features_col)
    # drop is no op if feature call is absent, not an error
    df = feature_assembler.transform(df.drop(features_col))

    return df, out_cols

@_time_log
def bin_features(df: DataFrame,
                 features_col: str,
                 ntiles: Union[Dict[str, int], int] = 5,
                 error_scale: Union[int, float] = 5,
                 sample_size: Optional[int] = 10 ** 4) \
        -> Tuple[DataFrame, List[str]]:
    r"""
    use quantiles to bin numeric features into interval/ordinal


    Parameters
    ----------
    df : pyspark.sql.DataFrame
        dataframe containing the relevant columns
    cols : int
        the columns to be binned
    ntiles : [dict, int]
        either col:ntile dict or single int for all columns. The number
        of equal by count divisions the column should be binned into
    features_col : str
        the column to assemble the new binned features into.
        dropped if already in `df`

    Returns
    -------
    dataframe: pyspark.sql.DataFrame
        input dataframe with new columns
    binned_cols : List[str]
        List of new column names


    Other Parameters
    ----------------
    error_scale: Union[float, int], Optional:
        1/ntile/error_scale is the errorTolerance passed to
        approxQuantile when calculating quantile bin thresholds.
        Default value is 5
    sample_size: int, Optional
        number of rows to consider when subsampling to calculate quantiles.
        max of 10,000 by default
        if None whole dataframe will be used

    Raises
    ------
    Uncaught exceptions
        Illegal input values, esp non-whole, negative,
        or otherwise unreasonable numbers for ntile
        Name conflicts for new "binned{`col`}" columns

    Notes
    -----
    If data is '0 degenerate', where the number of minimum values
    is greater than a single ntile division, make them the bottom bin and
    squeeze the rest of the divisions in the remaining space. Lose the
    interval guarantee but is useful in practice
    """

    # create dictionary of column to number of even splits

    if isinstance(ntiles, int):
        logging.getLogger(__name__).info("binning features into {n} bins".format(n=ntiles))
    else:
        logging.getLogger(__name__).info("binning features into individual bin numbers")

    cols = _get_pred_cols(df=df, features_col=features_col)

    if isinstance(ntiles, dict):
        ntile_dict = ntiles
    else:
        ntile_dict = {col: ntiles for col in cols}

    sample_df = _sample_df(df=df, sample_size=sample_size)
    _persist_if_unpersisted(sample_df)
    sample_count = sample_df.count()

    num_pinned = 0

    _persist_if_unpersisted(df)
    for col, ntile in ntile_dict.items():
        relative_error = 1 / ntile / error_scale
        min_val = sample_df.select(F.min(F.col(col))).take(1)[0][0]
        min_count = sample_df.where(F.col(col) == min_val).count()

        # if the number of minimum values is greater than the space in one division, make 1st tile the minimum values
        # and compress the rest in the remaining space
        if (min_count / sample_count) > (1 / ntile):
            probabilities = [float(x) for x in np.linspace(start=0, stop=1, num=ntile - 1, endpoint=False)][1:]
            threshs = [min_val] + sample_df.where(F.col(col) > min_val).approxQuantile(col, probabilities, relative_error)
            num_pinned += 1
        else:
            probabilities = [float(x) for x in np.linspace(start=0, stop=1, num=ntile, endpoint=False)][1:]
            threshs = sample_df.approxQuantile(col, probabilities, relative_error)

        # the number of thresholds it is greater than or equal to is its nthtile
        def make_udf(threshs):
            return F.udf(lambda x: sum([x > y for y in threshs]), T.ShortType())

        # add 1 to conform to mathematical indexing of ntiling
        df = df.withColumn("binned_{col}".format(col=col), make_udf(threshs)(F.col(col)) + 1)

    logging.getLogger(__name__).info("{n:,} of {m:,} columns have min_val pinned first ntile".format(n=num_pinned, m=len(ntile_dict)))
    binned_cols = ["binned_{col}".format(col=col) for col in cols]

    assembler = mlf.VectorAssembler(inputCols=binned_cols, outputCol=features_col)
    df = assembler.transform(df.drop(features_col))

    return df, binned_cols


def reduce_dimensionality(args: dict, method: str = 'chi') -> Tuple[DataFrame, List[str]]:
    r"""entry point for dimensionality reduction functions.

    Parameters
    ----------
    args : dict
        kwargs dicts w/ arguments for chi or log dim reduction method
    method : {'chi', log'}
        which dim reduction method to use

    See Also
    --------
    reduce_chi_dimensionality
    reduce_log_dimensionality
    """

    func_dict = {
        'chi': reduce_chi_dimensionality,
        'log': reduce_log_dimensionality
    }
    return func_dict[method](**args)

@_time_log
def reduce_chi_dimensionality(
        df: DataFrame,
        label_col: str,
        binned_features_col: str,
        ncols: Optional[int] = None,
        drop_uninformative: bool = True,
        sample_size: Optional[int] = 10 ** 4) -> Tuple[DataFrame, List[str]]:
    r""" Use chi-squared to pick the most informative features

    Find the most informative features using chi squared. Input features
    should be categorical. Uninformative features may dropped using fwe.
    Under-the-hood multiple chi squared tests are run w/ `sample_size`
    to account for data limitations

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    label_col : str
        colname of label column. should be 1 or 0 with smaller treatment
        class as 1
    binned_features_col : str
        column of predictors assembled into a vector
    ncols : int, Optional
        number of desired output columns. If unspecified, will default to
        the number of positive samples/`SAMPLES_PER_FEATURE`
    drop_uninformative : bool, Optional
        defaults to True. uses fwe w/ significance of .05 to drop
        uninformative columns
    sample_size : int, Optional
        sample size used in each run to evaluate significance of each
        predictor. defaults to 10**4. This threshold was chosen empirically
        can be passed as None to avoid sampling ( not recommended)

    Returns
    -------
    df : pyspark.sql.DataFrame
        input dataframe, but `binned_features_col` now consists of selected
        columns.
    selected_cols : List[str]
        list of chosen columns

    Raises
    ------
    AssertionError
        if `ncols` is less than 1, either through calculation or
        specification
        if no columns are informative
    UncaughtExceptions
        non-int values for ncols

    Notes
    -----
    using ChiSqSelector with many samples will make p-values of
    columns in internal calculations identical. Sorting by p-value and
    returning therefore relies on input order. This method
    circumvents that by subsampling to `sample_size` and running
    int(math.ceil(math.log(all_count/sample_size))) times
    (or 1 when df size is less than sample_size). Each iteration,
    the rank for each column is calculated and added to previous ranks.
    The top ncol columns are chosen with the smallest aggregate rank.

    """
    _persist_if_unpersisted(df)



    binned_pred_cols = _get_pred_cols(df, binned_features_col)

    # if ncols is not specified, set it based on samples specified per feature and positive sample count
    if ncols is None:
        pos_count = df.where(F.col(label_col) == 1).count()
        ncols = int(pos_count // SAMPLES_PER_FEATURE)
        logging.getLogger(__name__).info("ncols given as None, calculated to be {nc:,}".format(nc=ncols))

    if isinstance(sample_size, int):
        ss = "{:,}".format(sample_size)
    else: ss = str(sample_size)
    logging.getLogger(__name__).info("using chi method to reduce dim with params "
                                     "ncols={nc:,} drop_uninformative={du} sample_size={ss}".format(
            nc=ncols, du=str(drop_uninformative), ss=ss))

    if ncols < 1:
        logging.getLogger(__name__).critical("ncols {nc}, illegal".format(nc=ncols))
        assert ncols >= 1, "cannot return less than 1 column"
    if not isinstance(ncols, int):
        logging.getLogger(__name__).critical("ncols({nc}) type is illegal".format(nc=str(ncols)))
        raise ValueError("ncols is not int but type {type_}".format(type_=type(ncols)))

    if len(binned_pred_cols) <= ncols:
        logging.getLogger(__name__).info("{n:,} predictors is already less/equal to desired {m:,}".format(n=len(binned_pred_cols),m=ncols))
        if drop_uninformative:
            informative_selector = mlf.ChiSqSelector(selectorType='fwe',
                                                     fwe=.05,
                                                     labelCol=label_col,
                                                     featuresCol=binned_features_col)
            selected_info_cols = [str(x) for x in
                             list(np.array(binned_pred_cols)[informative_selector.fit(df).selectedFeatures])]
            logging.getLogger(__name__).info("{n:,} of {m:,} cols found to be informative".format(n=len(selected_info_cols), m=len(binned_pred_cols)))
            assert len(selected_info_cols) > 1, "no informative columns found"
            assembler = mlf.VectorAssembler(inputCols=selected_info_cols, outputCol=binned_features_col)
            df = assembler.transform(df.drop(binned_features_col))
            return df, selected_info_cols
        else:
            return df, binned_pred_cols

    all_count = df.count()

    if (sample_size is None) | (all_count <= sample_size):
        if isinstance(sample_size, int):
            logging.getLogger(__name__).info("no sampling necessary {c:,} <= {sn:,}".format(c=all_count, sn=sample_size))
        else:
            logging.getLogger(__name__).info("requested not to sample, df of size {c:,}".format(c=all_count))
        num_runs = 1
        sample_frac = float(1)
    else:  # all_count > sample_size:
        logging.getLogger(__name__).info("sampling with desired_size {sz:,} and df size {ac:,}".format(sz=sample_size, ac=all_count))
        num_runs = int(math.ceil(math.log(all_count / sample_size)))
        sample_frac = sample_size / all_count

    logging.getLogger(__name__).info("num_runs:{nr:,}   sample_frac:{sf:.2f}".format(nr=num_runs, sf=sample_frac))

    col_overall_ranks = defaultdict(lambda: 0)
    feature_ranker = mlf.ChiSqSelector(selectorType='numTopFeatures',
                                       numTopFeatures=len(binned_pred_cols),
                                       labelCol=label_col,
                                       featuresCol=binned_features_col)

    # test binning
    # for each run, add significance rank of column to previous ranks.
    for run in range(num_runs):
        # while we usually specify the seed, we rely on randomness of
        # the seeds here. perhaps introduce a prng where we specify
        # the seed, fulfilling reproducibility and randomness needs
        sampled_df = df.sample(withReplacement=False, fraction=sample_frac)
        sampled_df.persist(StorageLevel(False, True, False, False))

        cols = feature_ranker.fit(sampled_df).selectedFeatures
        col_ranks = zip(cols, list(range(len(cols))))
        for col, rank in col_ranks:
            col_overall_ranks[col] += rank

    # pick top ncol cols with lowest aggregate ranks
    col_overall_ranks = list(col_overall_ranks.items())
    col_sorted_ranks = sorted(col_overall_ranks, key=lambda x: x[1])
    selected_col_indices = [x[0] for x in col_sorted_ranks[:ncols]]
    selected_cols = [str(x) for x in list(np.array(binned_pred_cols)[selected_col_indices])]

    assembler = mlf.VectorAssembler(inputCols=selected_cols, outputCol=binned_features_col)
    df = assembler.transform(df.drop(binned_features_col))

    if drop_uninformative:
        informative_selector = mlf.ChiSqSelector(selectorType='fwe',
                                                 fwe=.05,
                                                 labelCol=label_col,
                                                 featuresCol=binned_features_col)
        selected_info_cols = [str(x) for x in list(np.array(selected_cols)[informative_selector.fit(df).selectedFeatures])]
        assert len(selected_cols) > 1, "no informative columns found"
        assembler = mlf.VectorAssembler(inputCols=selected_info_cols, outputCol=binned_features_col)
        df = assembler.transform(df.drop(binned_features_col))
        logging.getLogger(__name__).info(
            "{n:,} of {m:,} cols found to be informative".format(n=len(selected_info_cols), m=len(selected_cols)))
        return df, selected_info_cols
    return df, selected_cols


@_time_log
def reduce_log_dimensionality(df: DataFrame,
                              label_col: str,
                              binned_features_col: str,
                              ncols: Optional[int] = None,
                              sample_size: Optional[int] = 10 ** 6,
                              log_args: Optional[dict] = None) -> Tuple[DataFrame, List[str]]:
    r"""use logistic regression on normalized predictors to find top
    ncols with greatest absolute beta coefficients

    Parameters
    ----------
    df : pyspark.sql.DataFrame
    binned_pred_cols : List[str]
        list of normalized predictors assembled in `binned_features_col`
    label_col : str
        colname with class labels. Should be binary, with 1 being the
        smaller treatment class
    binned_features_col : str
        colname of categorical predictors assembled into a vector
    ncols : int, optional
        defaults to positive_samples/`SAMPLES_PER_FEATURE`
        number of columns to return
    sample_size : int, optional
        defaults to 10**6, size of sample to train logistic regression on
        can be passed as None to avoid sampling
    log_args : dict, optional
        kwargs dict for pyspark.ml.classification.LogisticRegression
        estimator initialization. Otherwise specifies label, feature col
        and uses pyspark defaults. must include featureCol and labelCol
        arguments. label_col, binned_features_col args will be ignored
        but must still be passed


    Returns
    -------
    df : pyspark.sql.DataFrame
        input df but `binned_features_cols` is now a vector of chosen columns
    selected_cols : List[str]
        list of chosen columns

    Raises
    ------
    AssertionError
        ncol less than 1 by calculation or specification
    UncaughtExceptions
        non-int vals for ncol

    Notes
    -----
    Assumes input variables have been normalized. Train log regression and
    picks `ncols` with greatest absolute coefficient. Does not guarantee
    this combination of columns will have greatest predictive power,
    especially in cases with high-impact low-presence predictors.

    """

    _persist_if_unpersisted(df)
    binned_pred_cols = _get_pred_cols(df=df, features_col=binned_features_col)

    if ncols is None:
        pos_count = df.where(F.col(label_col) == 1).count()
        ncols = int(pos_count // SAMPLES_PER_FEATURE)
        logging.getLogger(__name__).info("ncols given as None, calculated to be {nc:,}".format(nc=ncols))

    assert ncols >= 1, "cannot return less than 1 column"

    if log_args is None:
        log_estimator = mlc.LogisticRegression(featuresCol=binned_features_col, labelCol=label_col)
    else:
        log_estimator = mlc.LogisticRegression(**log_args)

    sample_df = _sample_df(df=df, sample_size=sample_size)
    _persist_if_unpersisted(sample_df)
    log_model = log_estimator.fit(sample_df)
    # sort abs value greatest to least
    coeffs = sorted(zip(binned_pred_cols, log_model.coefficients), key=lambda x: -abs(x[1]))
    selected_cols = [x[0] for x in coeffs[:ncols]]

    assembler = mlf.VectorAssembler(inputCols=selected_cols, outputCol=binned_features_col)
    df = assembler.transform(df.drop(binned_features_col))

    return df, selected_cols
