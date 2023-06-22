import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import threshold_multiotsu, apply_hysteresis_threshold, frangi
from skimage.filters.ridges import compute_hessian_eigenvalues
from skimage.morphology import skeletonize, remove_small_objects, black_tophat, white_tophat, disk, ball
from scipy.ndimage import distance_transform_edt as edt  # euclidean distance transform

# required to the save option in the OMELETTE pipeline function
from vdm.io import load_nii_as_np_array, save_np_array_as_nii, get_voxel_size_from_nii_header
from vdm.visualize import segmentation_overview_plot, save_figure


def vessel_segmentation(filename_nii, save_as_nii=True, save_screenshots=True,
                        filter_name="Frangi", sigmas=range(1, 5, 1), black_ridges=False,
                        tau=0.75, gamma=0.1, select_gamma_automatically=True, alpha=0.5, beta=0.5,
                        normalize_min=None, normalize_max=None,
                        mode='reflect', cval=0,
                        do_tophat=False, tophat_size=None,
                        n_threshold_iterations=1, kappa=0.1, pruning_cut_off=0, prevent_leaking=True,
                        verbose=False):
    """
    OMELETTE vessel segmentation pipeline

    Parameters todo
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data [2D or 3D].
        This image will be normalized to [0..1].
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    tau : float [0.5..1.0], optional
        Parameter used for the regularization.
        Lower values result in increased response (also for noise)
    black_ridges : boolean, optional
        When True, the filter detects black ridges;
        When False, it detects white ridges (default).
    do_tophat: boolean, optional
        When True, a top-hat transformation is performed as a pre-processing step to isolate the vasculature.
    tophat_size: integer, optional
        Determines the size of the structured element used in the top-hat transformation.
        Features larger than this structured element are filtered out
        Thus, set the size larger than the highest sigma used.
        If not specified, an tophat_size is computed automatically based on the sigmas provided (input)
    mode : {'constant', 'reflect' (default), 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
        The use of 'reflect' aka default is recommended
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    verbose : bool, optional
        If true the Hessian eigenvalues are plotted per scale.
    Returns todo
    -------
    filtered_image: (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).
    max_response_sigma: (N, M[, P]) ndarray
        Maximal response/sigma for each pixel across all scales.

    Notes todo
    -----
    Vessel enhancement filter proposed by T. Jerman et al in [1]_
    MATLAB implementation in [2]_
    Re-written for Python by Hendrik Mattern, August 2020
    The code relies heavily on the scikit-image, and also the code structure is largely following the scikit-image
    Frangi implementation (very similar up to the response function, which is re-implemented from Jerman et al.)

    References todo
    ----------
    .. [1] T. Jerman, F. Pernus, B. Likar, Z. Spiclin,
        "Enhancement of Vascular Structures in 3D and 2D Angiographic Images",
        IEEE Transactions on Medical Imaging, 35(9), p. 2107-2118 (2016), doi=10.1109/TMI.2016.2550102
    .. [2] T. Jerman: https://github.com/timjerman/JermanEnhancementFilter
    .. [3] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    """
    # Load data
    image, affine, header = load_nii_as_np_array(filename_nii)
    # Normalize intensity from 0.0 to 1.0
    if normalize_min is None:
        normalize_min = np.min(image)
    if normalize_max is None:
        normalize_max = np.max(image)
    image = (image - normalize_min) / (normalize_max - normalize_min)
    image[image > 1.0] = 1.0
    image[image < 0.0] = 0.0
    # Vessel enhancement filter switch:
    if filter_name.lower() == "frangi":
        print("Frangi filtering")
        filtered = hm_frangi(image, sigmas=sigmas,
                             gamma=gamma, select_gamma_automatically=select_gamma_automatically,
                             black_ridges=black_ridges, alpha=alpha, beta=beta, mode='reflect', cval=0,
                             do_tophat=do_tophat, tophat_size=tophat_size)
    else:
        # Jerman enhancement
        print("Jerman filtering")
        filtered, _ = jerman(image, sigmas=sigmas, tau=tau, black_ridges=black_ridges, mode=mode, cval=cval,
                             do_tophat=do_tophat, tophat_size=tophat_size, verbose=verbose)

    # To estimate the radius in mm, we need the voxel size
    voxel_size = get_voxel_size_from_nii_header(header)

    # Do segmentation
    _, segmentation, vessel_radius = filtered2skeleton(filtered, n_threshold_iterations=n_threshold_iterations,
                                                       kappa=kappa, pruning_cut_off=pruning_cut_off,
                                                       prevent_leaking=prevent_leaking, voxel_size=voxel_size,
                                                       verbose=verbose)
    # Get filename without extension (required for following options
    if save_as_nii | save_screenshots:
        if "nii.gz" in filename_nii:
            basename = filename_nii.split(".nii.gz")[0]
        else:
            basename = filename_nii.split(".nii")[0]
    # Save as nii.gz
    if save_as_nii:
        # save everything in same folder as the original data
        save_np_array_as_nii(filtered, affine, header, basename + "_filtered.nii.gz")
        save_np_array_as_nii(segmentation, affine, header, basename + "_segmentation.nii.gz")
        save_np_array_as_nii(vessel_radius, affine, header, basename + "_radius.nii.gz")
    # Create and save screenshots
    if save_screenshots:
        segmentation_overview_plot(image, filtered, segmentation, vessel_radius)
        save_figure(basename + "_segmentation.jpg")
        plt.close()

    return image, filtered, segmentation, vessel_radius


def _jerman_vesselness_response(hessian_eigenvalues, black_ridges=False, tau=0.5):
    # extract the second eigenvalues
    lambda2 = hessian_eigenvalues[1, :, :]

    # 2D vs 3D handling
    if len(hessian_eigenvalues) == 2:
        lambda_rho = np.copy(lambda2)  # if 2D use lambda2; copy require, otherwise only reference
    else:
        lambda_rho = hessian_eigenvalues[2, :, :]  # if 3D use lambda3; hence using reference is okay here

    # bright ridges require "sign switch"
    if not black_ridges:
        lambda2 = -lambda2
        lambda_rho = -lambda_rho

    # computes lambda_rho: regularized version of lambda2/3 to prevent noise enhancement for low responses
    threshold = tau * np.max(lambda_rho)  # compute threshold
    lambda_rho[(lambda_rho > 0) & (lambda_rho <= threshold)] = threshold  # "boost" pos. values smaller than threshold
    lambda_rho[lambda_rho <= 0] = 0  # set negative values to zero

    # calculate vessel response
    denominator = ((lambda2 + lambda_rho) ** 3)
    denominator[denominator == 0] = 1e-6  # prevent division by zero
    response = 27 * (lambda_rho - lambda2) * lambda2 ** 2 / denominator
    # account for structures with elliptic cross-sections with ratio of lambda2/lambda_rho from 0.5 to 1
    response[(lambda2 >= lambda_rho / 2) & (lambda_rho > 0)] = 1
    # suppress negative eigenvalues
    response[(lambda2 <= 0) | (lambda_rho <= 0)] = 0

    return response


def _plot_hessian_eigenvalues(hessian_eigenvalues):
    cmap = plt.get_cmap('gray')
    if hessian_eigenvalues.shape[0] == 2:
        _, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(hessian_eigenvalues[0, :, :], cmap=cmap)
        ax[1].imshow(hessian_eigenvalues[1, :, :], cmap=cmap)
    else:
        _, ax = plt.subplots(1, 3, figsize=(8, 4))
        ax[0].imshow(hessian_eigenvalues[0, :, :, np.int_(hessian_eigenvalues.shape[3]//2)], cmap=cmap)
        ax[1].imshow(hessian_eigenvalues[1, :, :, np.int_(hessian_eigenvalues.shape[3]//2)], cmap=cmap)
        ax[2].imshow(hessian_eigenvalues[2, :, :, np.int_(hessian_eigenvalues.shape[3]//2)], cmap=cmap)
    plt.show()


def jerman(image, sigmas=range(1, 10, 2), tau=0.5, black_ridges=False,
           do_tophat=False, tophat_size=None,
           mode='reflect', cval=0, verbose=False):
    """
    Filter an image with the Jerman vesselness filter [1]. Available for MATLAB here [2].
    Related to the popular Frangi vesselness approach [3].
    This filter is based on the analysis of the eigenvalues of the Hessian matrix of an image and
    can be used to detect continuous ridges, e.g. vessels, wrinkles, rivers.
    Defined only for 2-D and 3-D images.

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        Array with input image data [2D or 3D].
        This image will be normalized to [0..1].
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    tau : float [0.5..1.0], optional
        Parameter used for the regularization.
        Lower values result in increased response (also for noise)
    black_ridges : boolean, optional
        When True, the filter detects black ridges;
        When False, it detects white ridges (default).
    do_tophat: boolean, optional
        When True, a top-hat transformation is performed as a pre-processing step to isolate the vasculature.
    tophat_size: integer, optional
        Determines the size of the structured element used in the top-hat transformation.
        Features larger than this structured element are filtered out
        Thus, set the size larger than the highest sigma used.
        If not specified, an tophat_size is computed automatically based on the inputed sigmas
    mode : {'constant', 'reflect' (default), 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
        The use of 'reflect' aka default is recommended
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    verbose : bool, optional
        If true the Hessian eigenvalues are plotted per scale.
    Returns
    -------
    filtered_image: (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).
    max_response_sigma: (N, M[, P]) ndarray
        Maximal response/sigma for each pixel across all scales.

    Notes
    -----
    Proposed by T. Jerman et al in [1]_
    MATLAB implementation in [2]_
    Re-written for Python by Hendrik Mattern, August 2020
    The code relies heavily on the scikit-image, and also the code structure is largely following the scikit-image
    Frangi implementation (very similar up to the response function, which is re-implemented from Jerman et al.)

    References
    ----------
    .. [1] T. Jerman, F. Pernus, B. Likar, Z. Spiclin,
        "Enhancement of Vascular Structures in 3D and 2D Angiographic Images",
        IEEE Transactions on Medical Imaging, 35(9), p. 2107-2118 (2016), doi=10.1109/TMI.2016.2550102
    .. [2] T. Jerman: https://github.com/timjerman/JermanEnhancementFilter
    .. [3] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    """

    # Check image dimensions (could also be done with private method <check_nD(image, [2, 3])> )
    if not len(image.shape) in [2, 3]:
        raise ValueError('image is not 2D or 3D')

    # Check sigma scales (could also be done with private method <sigmas = _check_sigmas(sigmas)> )
    if np.any(np.asarray(sigmas).ravel() < 0.0):
        raise ValueError('Sigma values should be non-negative.')

    # Normalize input image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Top-hat transformation (optional pre-processing:
    if do_tophat:
        # set default size of top-hat transformation if required:
        if tophat_size is None:
            tophat_size = np.int_(sigmas[-1] + 5.0)
        # create structure element to perform the transformation with
        if len(image.shape) == 2:
            structure_element = disk(tophat_size)
        else:
            structure_element = ball(tophat_size)
        # do black or white top-hat transformation depending on the rigid-flag
        # FYI: after the top-hat transformation the output is hyperintense, aka white vessels
        # Hence, adapt black_ridges flag if required
        if black_ridges:
            image = black_tophat(image, footprint=structure_element)
            # image = black_tophat(image, selem=structure_element)
            black_ridges = False
        else:
            image = white_tophat(image, footprint=structure_element)
            # image = white_tophat(image, selem=structure_element)

    # Initialize filtered image
    filtered_image = np.zeros_like(image)

    # Initialize array which stores scale of the maximum response (can be used to approx. vessel diameter)
    max_response_sigma = np.zeros_like(image)

    # Filtering for all (sigma) scales
    for sigma in sigmas:
        print('current sigma: ', str(sigma))
        # Compute the eigenvalues of the Hessian (ie. lambda 1, 2 (and 3))
        #   1. Compute the Hessian of the image (use xy instead of rc order)
        #   2. Correct for scale
        #   3. Get eigenvalues
        #   4. Sort by absolute value
        hessian_eigenvalues = compute_hessian_eigenvalues(image=image, sigma=sigma, sorting='abs', mode=mode, cval=cval)

        if verbose:
            _plot_hessian_eigenvalues(hessian_eigenvalues)

        # Compute vesselness response function
        response = _jerman_vesselness_response(hessian_eigenvalues, black_ridges=black_ridges, tau=tau)

        # enhanced output is the maximum across all scales
        filtered_image = np.maximum(response, filtered_image)

        # get scale of maximum response for diameter approximation
        #   (if current scale is current maximum, replace the sigma entry in max_scale_per_voxel
        max_response_sigma[filtered_image == response] = sigma

    # Return for every pixel the maximum value over all (sigma) scales
    return filtered_image, max_response_sigma


def hm_frangi(image, sigmas=range(1, 10, 2), gamma=0.1, select_gamma_automatically=True, black_ridges=False,
              alpha=0.5, beta=0.5, mode='reflect', cval=0,
              do_tophat=False, tophat_size=None):
    # Wrapper function with optional tophat preprocessing

    # Check image dimensions (could also be done with private method <check_nD(image, [2, 3])> )
    if not len(image.shape) in [2, 3]:
        raise ValueError('image is not 2D or 3D')

    # Check sigma scales (could also be done with private method <sigmas = _check_sigmas(sigmas)> )
    if np.any(np.asarray(sigmas).ravel() < 0.0):
        raise ValueError('Sigma values should be non-negative.')

    # Normalize input image
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    # Top-hat transformation (optional pre-processing:
    if do_tophat:
        # set default size of top-hat transformation if required:
        if tophat_size is None:
            tophat_size = np.int_(sigmas[-1] + 5.0)
        # create structure element to perform the transformation with
        if len(image.shape) == 2:
            structure_element = disk(tophat_size)
        else:
            structure_element = ball(tophat_size)
        # do black or white top-hat transformation depending on the rigid-flag
        # FYI: after the top-hat transformation the output is hyperintense, aka white vessels
        # Hence, adapt black_ridges flag if required
        if black_ridges:
            image = black_tophat(image, footprint=structure_element)
            # image = black_tophat(image, selem=structure_element) deprecated
            black_ridges = False
        else:
            image = white_tophat(image, footprint=structure_element)
            # image = white_tophat(image, selem=structure_element) deprecated

    # Select gamma automatically
    if select_gamma_automatically:
        # get the Hessian eigenvalues for first scale
        hessian_eigenvalues = compute_hessian_eigenvalues(image=image, sigma=sigmas[0], sorting='abs', mode=mode,
                                                          cval=cval)
        # Gamma is half of the max Hessian eigenvalue
        gamma = 0.5 * np.max(hessian_eigenvalues)
        print(" - gamma used " + str(gamma))
    # Frangi filter
    filtered = frangi(image, sigmas=sigmas, alpha=alpha, beta=beta, gamma=gamma, black_ridges=black_ridges, mode=mode,
                      cval=cval)
    return filtered


def prune_skeleton_by_vessel_radius(skeleton, vessel_radius_per_voxel, cut_off_radius):
    # skeleton - skeleton of the segmentation
    # vessel_radius_per_voxel - response of the segmentation
    # cut_off_radius - find only vessels smaller than cut-off radius
    return (vessel_radius_per_voxel <= cut_off_radius) * (skeleton > 0)


def pruning_by_volume(data, pruning_cut_off=0):
    if not((isinstance(pruning_cut_off, (int, float))) | (type(np.array(1)) is np.array)):
        pruning_cut_off = 0
    if pruning_cut_off > 0:
        # enforce boolean data
        data = data > 0
        # pruning
        data = remove_small_objects(data, pruning_cut_off, connectivity=len(data.shape))
        # remove_small_objects(data, pruning_cut_off, connectivity=len(data.shape), in_place=True) deprecated
    return data


def filtered2skeleton(image_filtered, n_threshold_iterations=1, kappa=0.1, pruning_cut_off=0,
                      prevent_leaking=True, voxel_size=None, verbose=False):
    # FYI: non-iterative is default; selected by "n_threshold_iterations=1"

    # Create a copy of the original filtered image which will be altered
    # Hence, by creating a copy (and not using the reference), the original filter output stays unaltered
    image_filtered_temp = image_filtered.copy()
    # Check parameters
    if n_threshold_iterations < 1:
        print('n_threshold_iterations < 1\n hard reset to 1')
        n_threshold_iterations = 1

    # Display mode
    if n_threshold_iterations == 1:
        print('using NON-iterative segmentation')
    else:
        print('using iterative segmentation')

    # Init verbose plotting
    if verbose:
        n_bins = 255
        n_col = 3
        n_row = 2
        _, axs = plt.subplots(ncols=n_col, nrows=n_row)
        axs = axs.ravel()
        _plotting_processing(axs, n_col * n_row, 0, image_filtered_temp, 'unprocessed vesselness image')

    # Iterative thresholding approach
    for i in range(0, n_threshold_iterations):
        if verbose:
            print('iteration ', str(i + 1), ' of ', str(n_threshold_iterations))

        # Threshold from multi-otsu
        # below first/lower threshold is background, above second/higher threshold are (large/bright) vessels
        # small vessels lie in between thresholds
        # FYI: consider only non-zero values
        thresholds = threshold_multiotsu(image_filtered_temp[image_filtered_temp != 0].ravel(), 3)

        if verbose:
            if i == 0:
                axs[1].hist(image_filtered_temp.ravel(), bins=n_bins)
                axs[1].set_yscale('log')
                axs[1].set_title('log-intensity histrogram (first iteration)')
                for thresh in thresholds:
                    axs[1].axvline(thresh, color='r')
            elif i == n_threshold_iterations - 1:
                axs[2].hist(image_filtered_temp.ravel(), bins=n_bins)
                axs[2].set_yscale('log')
                axs[2].set_title('log-intensity histrogram (last iteration)')
                for thresh in thresholds:
                    axs[2].axvline(thresh, color='r')

        # Hysteresis thresholding based on multi-Otsu thresholds
        # this helps to find small vessels (in between thresholds)
        # which are connected to large/bright vessels (above second threshold)
        segmentation = apply_hysteresis_threshold(image_filtered_temp, thresholds[0], thresholds[1])

        # Prune-by-volume the segmentation:
        # Everything below cut-off volume will be removed; helps cleaning up the segmentation
        segmentation = pruning_by_volume(segmentation, pruning_cut_off)

        # Soft-enhancing:
        # Increase intensity of segmented voxels and repeat iterative thresholding
        # This is similar to a compressed sensing approach and should help to disentangle vessels from background
        image_filtered_temp += segmentation * kappa
        image_filtered_temp[image_filtered_temp > 1.0] = 1.0  # cut-off out-of-range values

    # Prevent leaking:
    # The iterative approach can produce a leaky segmentation (i.e. vessel diameter is over-estimated).
    # This is addressed by extracting the center-line of the iterative segmentation, applying the soft-thresholding
    # once to the initial filtered image only with the center-line and doing the multi-Otsu + hysteresis thresholding
    if prevent_leaking:
        # create new enhanced filtered image (center-line intensity == 1.0)
        image_filtered_temp = image_filtered + skeletonize(segmentation) * 1.0
        image_filtered_temp[image_filtered_temp > 1.0] = 1.0  # cut-off out-of-range values
        # apply thresholding
        thresholds = threshold_multiotsu(image_filtered_temp[image_filtered_temp != 0].ravel(), 3)
        segmentation = apply_hysteresis_threshold(image_filtered_temp, thresholds[0], thresholds[1])
        segmentation = pruning_by_volume(segmentation, pruning_cut_off)

    if verbose:
        _plotting_processing(axs, n_col * n_row, 3, segmentation, 'binary image after iterative thresholding')

    # Skeletonize
    # method='lee' will be used automatically for 3D input
    skeleton = skeletonize(segmentation) > 0  # additional condition to enforce boolean

    if verbose:
        _plotting_processing(axs, n_col * n_row, 4, skeleton, 'skeleton')
        plt.show()
        axs[-1].axis('off')

    # Vessel radius from segmentation
    vessel_radius = center_line_vessel_radius(segmentation, skeleton, voxel_size=voxel_size)

    return skeleton, segmentation, vessel_radius


def _plotting_processing(axs, num, i, img, title):
    # handle 3D data as MIP
    if len(img.shape) == 3:
        img = np.max(img, axis=2)

    # handle case if only one axis is required
    cmap = plt.get_cmap('gray')
    if num == 1:
        axs.imshow(img, cmap=cmap)
        axs.set_title(title)
    else:
        axs[i].imshow(img, cmap=cmap)
        axs[i].set_title(title)
    return i + 1


def center_line_vessel_radius(segmentation, skeleton=None, voxel_size=None):
    # enforce binary segmentation
    segmentation = segmentation > 0

    # get skeleton if necessary:
    if skeleton is None:
        skeleton = skeletonize(segmentation) > 0  # enforce boolean

    # compute vessel radius
    # use distance transform to find the shortest distance from outside the segmentation to the center-line
    radius_map = edt(segmentation, voxel_size)

    # return only the radius for the center-line
    return radius_map * np.int_(skeleton)


# metrics intended for segmentation performance
def _voxelwise_confusion_matrix(ground_truth, prediction):
    g = ground_truth.astype(dtype=bool)
    p = prediction.astype(dtype=bool)

    true_pos = np.logical_and(g, p)
    false_pos = np.logical_and(~g, p)
    false_neg = np.logical_and(g, ~p)
    true_neg = np.logical_and(~g, ~p)
    return true_pos, false_pos, false_neg, true_neg


def confusion_matrix(ground_truth, prediction):
    true_pos, false_pos, false_neg, true_neg = _voxelwise_confusion_matrix(ground_truth, prediction)
    return true_pos.sum(), false_pos.sum(), false_neg.sum(), true_neg.sum()


def dice_coefficient(ground_truth, prediction):
    true_pos, false_pos, false_neg, _ = confusion_matrix(ground_truth, prediction)
    return 2 * true_pos / (2 * true_pos + false_pos + false_neg)


def segmentation_sensitivity(ground_truth, prediction):  # aka recall
    true_pos, _, false_neg, _ = confusion_matrix(ground_truth, prediction)
    return true_pos / (true_pos + false_neg)


def segmentation_specificity(ground_truth, prediction):
    _, false_pos, _, true_neg = confusion_matrix(ground_truth, prediction)
    return true_neg / (true_neg + false_pos)


def youden_index(sensitivity, specificity):
    return sensitivity + specificity - 1