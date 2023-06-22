import numpy as np
from scipy.ndimage import distance_transform_edt as edt  # euclidean distance transform
from skimage.segmentation import watershed


def vdm(vessel_segmentation, voxel_size=None):
    # check if voxel size is given
    if voxel_size is None:
        voxel_size = 1  # compute distance in voxels
    # safety mechanism: work with boolean array
    vessel_segmentation = vessel_segmentation > 0
    # safety check: if all values in segmentation are False, return zero
    # otherwise, edt(~segmentation) computes the non-zero results for all voxels being True
    if np.all(~vessel_segmentation):  # np.all returns True only if all values are True (logical AND)
        return np.zeros(vessel_segmentation.shape)
    else:
        return edt(~vessel_segmentation, sampling=voxel_size)


def vessel_specific_vdm(multi_label_vessel_segmentation, labels=None, mode='rms', voxel_size=None):
    # compute vessel distance maps for several vessels individually
    # each vessel needs to have an unique label/segmentation
    # there are several combination modes:
    #       rms - Root-Mean-Squared
    #       avg - average
    #       uncombined - returns a VDM per label, hence, increases the dimensionality (2D->3D; 3D->4D); MEMORY HUNGRY!!!

    # enforce array of integers
    multi_label_vessel_segmentation = np.int_(multi_label_vessel_segmentation)

    # work-around for 2D data when using uncombined
    if (len(multi_label_vessel_segmentation.shape) == 2) & (mode == 'uncombined'):
        multi_label_vessel_segmentation = multi_label_vessel_segmentation[:, :, np.newaxis]

    # initialize labels
    if labels is None:
        labels = np.unique(multi_label_vessel_segmentation)
        labels = labels[labels != 0]  # exclude background label

    # safety-check: only do the analysis for labels which were actually segmented!
    # if a non-segmented label is analyzed, its vs-vdm will be all 0.0; thus, falsifying the subsequent analysis
    # since, vs-vdm will be normalized by the number of labels
    labels = np.intersect1d(multi_label_vessel_segmentation, labels)

    # initialize root-mean-squared
    vs_vdm = np.zeros(multi_label_vessel_segmentation.shape)

    # if maps for each label individually are required:
    if mode == 'uncombined':
        vs_vdm = np.zeros(multi_label_vessel_segmentation.shape + (len(labels),))
        for i, value in zip(range(0, len(labels)), labels):
            vs_vdm[:, :, :, i] = vdm(multi_label_vessel_segmentation == value, voxel_size=voxel_size)
    # otherwise, label-specific maps are combined each iteration to reduce the memory load
    else:
        for value in labels:
            # compute vdm
            current_vdm = vdm((multi_label_vessel_segmentation == value), voxel_size=voxel_size)
            # switch between different combination modes
            if mode == 'rms':
                vs_vdm = vs_vdm + current_vdm**2
            elif mode == 'avg':
                vs_vdm = vs_vdm + current_vdm

        # normalize according to the used combination mode
        if mode == 'rms':
            vs_vdm = np.sqrt(vs_vdm / len(labels))
        elif mode == 'avg':
            vs_vdm = vs_vdm / len(labels)

    return vs_vdm


def supply_and_distance_ratio(multi_label_vessel_segmentation, labels=None, voxel_size=None):
    # find the vessel/label associated with the minimal distance to the voxel of interest,
    # hence, find the supplying vessel
    # further, compute the (normalized) distance ratio per voxel: ratio = (shortest distance)/(2nd shortest distance)
    # hence, if the ratio is close to 1, there is evidence for overlap of perfusion territories / mixed supply pattern
    # additionally, the label of the potential collateral vessel is returned (label of 2nd closest vessel)
    # FYI: each vessel needs to have an unique label/segmentation

    # initialize labels
    if labels is None:
        labels = np.unique(multi_label_vessel_segmentation)
        labels = labels[labels != 0]  # exclude background label

    # safety-check: only do the analysis for labels which were actually segmented!
    # if a non-segmented label is analyzed, its vs-vdm will be all 0.0; thus, falsifying the subsequent analysis
    # since, the non-segmented label will always be considered the closest vessel
    labels = np.intersect1d(multi_label_vessel_segmentation, labels)

    # get vessel-specific vdm for each label
    vs_vdm_uncombined = vessel_specific_vdm(multi_label_vessel_segmentation, labels=labels, mode='uncombined',
                                            voxel_size=voxel_size)

    # get the index of the shortest and 2nd shortest vessel/label for each voxel
    supplying_vessel = vs_vdm_uncombined.argsort(axis=3)[:, :, :, 0]
    if vs_vdm_uncombined.shape[3] > 1:  # prevent error if only single vs-vdm was compute (only one label found)
        collateral_vessel = vs_vdm_uncombined.argsort(axis=3)[:, :, :, 1]
    else:
        collateral_vessel = np.zeros_like(supplying_vessel)  # todo: add warning

    # replace index with label values  (0 == 1st label value, 1 == 2nd label value, ...)
    supplying_vessel = labels[supplying_vessel]
    collateral_vessel = labels[collateral_vessel]

    # compute ratio of shortest and 2nd shortest distance
    if vs_vdm_uncombined.shape[3] > 1:  # prevent error if only single vs-vdm was compute (only one label found)
        vs_vdm_uncombined.sort(axis=3)
        ratio = vs_vdm_uncombined[:, :, :, 0] / vs_vdm_uncombined[:, :, :, 1]
    else:
        ratio = np.zeros_like(supplying_vessel)  # todo: add warning

    return supplying_vessel, ratio, collateral_vessel


def vdm_watershed(vdm_data, multi_label_vessel_segmentation):
    # create markers based on the multi label segmentation
    markers = np.zeros(vdm_data.shape)
    for label_value in np.unique(multi_label_vessel_segmentation):
        markers[multi_label_vessel_segmentation == label_value] = label_value
    # return watershed
    return watershed(vdm_data, markers=markers, watershed_line=True)  # line included in output as zeros in array


