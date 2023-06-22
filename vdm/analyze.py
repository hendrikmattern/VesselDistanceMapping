import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, product
from scipy.stats import kurtosis, skew, pearsonr, spearmanr, linregress
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering, KMeans, AffinityPropagation, MeanShift, OPTICS, estimate_bandwidth
from sklearn import preprocessing
from sklearn.decomposition import PCA
from vdm.io import load_as_np_array
import warnings


def calc_metrics_per_roi(data_filename_list, rois_filename_list, rois_dict,
                         metric_functions=None, metric_names=None, ids=None, csv_output_filename=None):
    """
        Compute multiple metrics per ROI for a list of files

        Parameters
        ----------
        data_filename_list : list of strings
            stores the data filenames (including path) to all files to be analyzed
        rois_filename_list : list of strings
            stores the ROI filenames (including path)
            has to be specified for each data set individually
        rois_dict: dictionary
            stores the name of the ROIs to be analyzed as the keys
            for each key a single value or multiple values (as a list) can be provide
            example: rois_dict={"roi1": 100, "roi1+2": [100, 200]}
        metric_functions: list of function handles
            points to all the functions / metrics to be calculated per ROI and data set
        metric_names: list of strings
            name for each metric to be computed
            if not provided, will be automatically derived from the metric_functions
        ids: list of strings
            list with the data set / subject identifier
            if not provided the data_filename_list will be used as ids
        csv_output_filename: string
            if filename provided, the final pandas data frame will be saved as a csv-file accordingly
    """

    # set function handle if not provide
    if metric_functions is None:
        metric_functions = [np.mean, np.std]
    if not isinstance(metric_functions, list):  # check input is a list
        warnings.warn("metric_functions has to be a list (use [single_metric] if only one metric used)")
        metric_functions = [metric_functions]
    # if required set the metric names
    if metric_names is None:
        metric_names = [metric.__name__ for metric in metric_functions]
    if not isinstance(metric_names, list):  # check input is a list
        warnings.warn("metric_names has to be a list (use [single_metric] if only one metric used)")
        metric_names = [metric_names]

    # set ids if required:
    if ids is None:
        ids = data_filename_list
    else:  # check that ids and data_filename_list have same length
        if not len(ids) == len(data_filename_list):
            warnings.warn("ids and data_filename_list have not the same length")

    # check that rois and data set list match
    if not len(rois_filename_list) == len(data_filename_list):
        warnings.warn("rois_filename_list and data_filename_list have not the same length\n")

    # init results as empty list
    result_list = []

    # compute metrics for all data sets / subjects
    for data_fn, rois_fn in zip(data_filename_list, rois_filename_list):
        print("processing " + data_fn)
        # load data
        data = load_as_np_array(data_fn)
        rois = load_as_np_array(rois_fn)
        # init results for current dataset as empty list
        result_per_data_set = []
        # init column labels as empty list
        columns = []
        # FYI: one could compute them ad-hoc, but we want to make sure that any changes to the loop below will be
        # automatically propagated to the columns label list to prevent mismatches
        # i.e. columns = [roi + metric.__name__ for roi, metric in product(rois_dict.keys(), metric_functions)]

        # loop over all rois specified in the dictionary
        for key, value in rois_dict.items():
            # create a mask for current roi
            if not isinstance(value, list):  # FYI: roi could have multiple values, hence the loop
                mask = rois == value
            else:
                mask = np.zeros_like(rois)
                for v in value:
                    mask = mask + np.int_(rois == v)

            # loop over all metrics
            for metric_handle, metric_name in zip(metric_functions, metric_names):
                result_per_data_set.append(metric_handle(data[mask > 0]))
                columns.append(key + "_" + metric_name)

        # store results for current data set / subject in list
        result_list.append(result_per_data_set)

    # convert to data frame
    df = pd.DataFrame(index=ids, columns=columns, data=result_list)

    # save data frame as csv file
    if csv_output_filename is not None:
        df.to_csv(csv_output_filename)

    return df


def __bonferroni_correction(p_value, n_bonferroni_correction=1):
    # FYI: correction done by multiplying N with p-value instead of dividing the significance thresholds
    # IMHO this is easier to keep track of since the returned p-values are corrected from the get-go,
    # no bookkeeping required later
    p_value = p_value * n_bonferroni_correction
    if p_value > 1.0:
        p_value = 1.0
    return p_value


def correlation_with_bonferroni(x_values, y_values, correlation_method="Pearson", n_bonferroni_correction=1):
    if correlation_method.lower() == "pearson":
        r_value, p_value = pearsonr(x_values, y_values)
    else:
        r_value, p_value = spearmanr(x_values, y_values)
    return r_value, __bonferroni_correction(p_value, n_bonferroni_correction)


def regression_with_bonferroni(x_values, y_values, correlation_method="Pearson", n_bonferroni_correction=1):
    # get correlation
    r_value, p_value = correlation_with_bonferroni(x_values, y_values, correlation_method, n_bonferroni_correction)
    # get linear function
    results = linregress(x_values, y_values)
    return r_value, p_value, results.slope, results.intercept, results.stderr, results.intercept_stderr

