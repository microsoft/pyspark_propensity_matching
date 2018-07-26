import math
import logging
from typing import Tuple

import pyspark
import pyspark.sql.functions as F
import pyspark.ml.feature as mlf
import pyspark.ml.classification as mlc

from .config import SAMPLES_PER_FEATURE, MINIMUM_POS_COUNT
from .utils import reduce_dimensionality, _persist_if_unpersisted, _get_pred_cols, _time_log


@_time_log
def impact(df: pyspark.sql.DataFrame,
           response_col: str,
           prob_mod: mlc.Model
           ) -> Tuple[float, float, float]:
    r"""observe impact of treatment on response variable

    currently response must be binary
    if the df is small enough return naive difference in groupby label
    response mean. otherwise do additional regression on response col
    with label as predictor and use its coefficient as a measure of its
    impact. binning and dimensionality reduction will occur if necessary
    to do an effective regression

    Parameters
    ----------
    df: pyspark.sql.DataFrame
    response_col: str
    prob_mod: Tmlc.Model
        propensity model, mostly used to keep track of feature_col,
        label_col, pred_cols

    Returns
    -------
    treatment_rate : float
        treatment response rate
    control_rate : float
        control response rate
    adjusted_response : float
        impact of treatment on response, which may be
        `control_rate`-`treatment_rate` or may have further bias adjustement

    Raises
    ------
    ValueError
        when number of rows is less than `MINIMUM_POS_COUNT`*2
    UncaughtExceptions

    See Also
    --------
    bin_features
    _reduce_dimensionality

    Notes
    -----

    """

    _persist_if_unpersisted(df)

    label_col = prob_mod.getOrDefault('labelCol')
    features_col = prob_mod.getOrDefault('featuresCol')
    pred_cols = _get_pred_cols(df, features_col)

    all_count = df.count()

    # safety check
    if all_count < MINIMUM_POS_COUNT * 2:
        logging.getLogger(__name__).critical("somehow have less than 2*MINIMUM_POS_COUNT*2 rows")
        raise ValueError("Have less than MINIMUM_POS_COUNT*2 rows, this shouldnt be happening")

    # dict because 1, 0 for label col are not guaranteed to be ordered
    naive_response_dict = dict()
    response_list = df.groupby(label_col).mean(response_col).collect()
    naive_response_dict[response_list[0][label_col]] = response_list[0]["avg({col})".format(col=response_col)]
    naive_response_dict[response_list[1][label_col]] = response_list[1]["avg({col})".format(col=response_col)]
    treatment_rate, control_rate = naive_response_dict[1], naive_response_dict[0]
    logging.getLogger(__name__).info("treatment_rate:{tr:.2f}   control_rate:{cr:.2f}".format(tr=treatment_rate, cr=control_rate))

    # choose fewer features if appropriate to prevent overfit. round down
    num_preds = int(df.where(F.col(label_col) == 1).count() // SAMPLES_PER_FEATURE) - 1
    logging.getLogger(__name__).info("need max {n:,} predictors".format(n=num_preds))
    if num_preds < len(list(pred_cols)):
        logging.getLogger(__name__).info("desired predictors {np:,} is less than existing {ep:,}, reducing dimensionality".format(np=num_preds, ep=len(pred_cols)))
        kwargs = {
            'df': df,
            'label_col': label_col,
            'binned_features_col': features_col,
            'ncols': num_preds
            }
        df, pred_cols = reduce_dimensionality(args=kwargs, method='chi')

    pred_cols_r = pred_cols + [label_col]
    assembler_r = mlf.VectorAssembler(inputCols=pred_cols_r, outputCol='features_r')
    df = assembler_r.transform(df)
    _persist_if_unpersisted(df)
    lre_r = mlc.LogisticRegression(featuresCol='features_r',
                                   labelCol=response_col,
                                   predictionCol='prediction_{0}'.format(response_col),
                                   rawPredictionCol='rawPrediction_{0}'.format(response_col),
                                   probabilityCol='probability_{0}'.format(response_col),
                                   regParam=.05,
                                   elasticNetParam=.5
                                   )
    lrm_r = lre_r.fit(df)

    coeff_dict = dict(zip(pred_cols_r, lrm_r.coefficients))

    adjusted_response = control_rate * (1 - math.exp(coeff_dict[label_col]))
    logging.getLogger(__name__).info("bias adjusted response is {ar:.2f}".format(ar=adjusted_response))
    return treatment_rate, control_rate, adjusted_response
