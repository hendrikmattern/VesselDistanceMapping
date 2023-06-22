import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from vdm.io import load_as_np_array
from vdm.analyze import correlation_with_bonferroni


def save_figure(filename, file_format='jpg', dpi=300):
    plt.savefig(filename,
                dpi=dpi,
                format=file_format,
                bbox_inches='tight',
                pad_inches=0)


def _crop_to_mask_find_index_1d(mask_1d, border_in_voxels=0):
    min_1d = np.min(np.argwhere(mask_1d > 0)) - border_in_voxels
    if min_1d < 0:
        min_1d = 0
    max_1d = np.max(np.argwhere(mask_1d > 0)) + border_in_voxels
    if max_1d > len(mask_1d) - 1:
        max_1d = len(mask_1d) - 1
    return min_1d, max_1d


def crop_to_mask(data, mask, border_in_voxels=0):
    # 3D cropping
    if len(data.shape) == 3:
        x_min, x_max = _crop_to_mask_find_index_1d(np.max(mask, axis=(1, 2)), border_in_voxels)
        y_min, y_max = _crop_to_mask_find_index_1d(np.max(mask, axis=(0, 2)), border_in_voxels)
        z_min, z_max = _crop_to_mask_find_index_1d(np.max(mask, axis=(0, 1)), border_in_voxels)
        crop = data[x_min:x_max, y_min:y_max, z_min:z_max] * mask[x_min:x_max, y_min:y_max, z_min:z_max]
    else:  # 2D cropping
        x_min, x_max = _crop_to_mask_find_index_1d(np.max(mask, axis=1), border_in_voxels)
        y_min, y_max = _crop_to_mask_find_index_1d(np.max(mask, axis=0), border_in_voxels)
        crop = data[x_min:x_max, y_min:y_max] * mask[x_min:x_max, y_min:y_max]
    return crop


def histogram_from_image_data(ax, data, mask=None, title=None, hist_type='step',
                              alpha=1.0, n_bins=256, hist_range=None,
                              show_y_label=True, x_label=None):
    if mask is None:
        mask = np.ones(data.shape)
    data_1d = data[mask > 0]
    ax.hist(data_1d, alpha=alpha, bins=n_bins, range=hist_range, density=True, histtype=hist_type)
    ax.set_title(title)
    if show_y_label:
        ax.set_ylabel("Normalized probability density")
    if x_label is not None:
        ax.set_xlabel(x_label)


def density_estimation_from_image_data(ax, data, mask=None, title=None, show_y_label=True, x_label=None):
    # extract data as 1d vector
    if mask is None:
        mask = np.ones(data.shape)
    data_1d = data[mask == 1]
    # estimate density
    kernel = gaussian_kde(data_1d)
    # plot
    x = np.linspace(np.floor(np.min(data_1d)), np.ceil(np.max(data_1d)), 1000)
    ax.plot(x, kernel(x), '.')
    ax.set_title(title)
    if show_y_label:
        ax.set_ylabel("Normalized probability density")
    if x_label is not None:
        ax.set_xlabel(x_label)


def compare_images_as_scatter_plot(img1, img2, mask=None, ax=None, title="", x_label="", y_label="", s=0.1, alpha=1.0):
    # init figure if required
    if ax is None:
        _, ax = plt.subplots()
    # extract values inside mask
    if mask is None:
        mask = np.ones_like(img1)
    img1_1d = img1[mask > 0]
    img2_1d = img2[mask > 0]
    # plot
    ax.scatter(img1_1d, img2_1d, marker='.', s=s, facecolor='0.0', alpha=alpha)
    # format plot
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    # return axis handle
    return ax


def _vdm_show_get_2d_slice(data, slc=None, axis=None):
    # init orientation if required
    if axis is None:
        axis = 0

    # init slc if required
    if slc is None:
        slc = np.int_(np.floor(data.shape[axis] / 2))

    # shuffle axis
    data_rearranged = np.moveaxis(data, axis, -1)

    # extract slice
    return data_rearranged[:, :, slc]


def vdm_show(ax, data, slc=None, axis=None, vmin=None, vmax=None, title=None, cmap_str=None,
             colorbar_active=True, colorbar_unit="", colorbar_label_min=None, colorbar_label_max=None):
    # set default colormap
    if cmap_str is None:
        cmap_str = 'Reds_r'
    # check if data is already 2D
    if len(data.shape) == 2:
        data_2d = data
    else:  # otherwise extract 2d slice
        data_2d = _vdm_show_get_2d_slice(data, slc, axis)
    # set vmin and vmax if not provided
    if vmax is None:
        vmax = np.max(data_2d)
    if vmin is None:
        vmin = 0

    # plot
    im = ax.imshow(np.rot90(data_2d), cmap=cmap_str, vmin=vmin, vmax=vmax)
    # formatting (i.e. title and remove ticks/borders)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # set color bar
    if colorbar_active:
        # set labels if required
        if colorbar_label_min is None:
            colorbar_label_min = f"{vmin:.1f} {colorbar_unit}"
        if colorbar_label_max is None:
            colorbar_label_max = f"{vmax:.1f} {colorbar_unit}"

        # create colorbar with adapted size and position
        fig = plt.gcf()
        cax = fig.add_axes([ax.get_position().x1 + ax.get_position().x1 * 0.005, ax.get_position().y0,
                            ax.get_position().width * 0.05, ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)

        # adjust ticks and labels
        cbar.set_ticks((vmin, vmax))
        cbar.set_ticklabels((colorbar_label_min, colorbar_label_max))
        # cbar.ax.tick_params(labelsize=18)


def create_mip(image, mip_start=None, mip_end=None, projection_axis=0, mask=None):
    if mip_start is None:
        mip_start = 0
    if mip_end is None:
        mip_end = image.shape[projection_axis]
    # additional mask support to prevent contribution from data from outside the mask
    if mask is not None:
        mask = mask > 0  # enforce binary mask
        image[mask != 1] = np.min(image[mask == 1])  # set everything outside mask to image minimum values
    # create projections depending on the axis
    if projection_axis == 0:
        mip = np.max(image[mip_start:mip_end, :, :], axis=projection_axis)
    elif projection_axis == 1:
        mip = np.max(image[:, mip_start:mip_end, :], axis=projection_axis)
    else:
        mip = np.max(image[:, :, mip_start:mip_end], axis=projection_axis)
    return np.rot90(mip)


def create_minip(image, mip_start=None, mip_end=None, projection_axis=0, mask=None):
    if mip_start is None:
        mip_start = 0
    if mip_end is None:
        mip_end = image.shape[projection_axis]
    # additional mask support to prevent all black / zero minIP for cropped data
    if mask is not None:
        mask = mask > 0  # enforce binary mask
        image[mask != 1] = np.max(image[mask == 1])  # set everything outside mask to image maximum values
    # create projections depending on the axis
    if projection_axis == 0:
        mip = np.min(image[mip_start:mip_end, :, :], axis=projection_axis)
    elif projection_axis == 1:
        mip = np.min(image[:, mip_start:mip_end, :], axis=projection_axis)
    else:
        mip = np.min(image[:, :, mip_start:mip_end], axis=projection_axis)
    return np.rot90(mip)


def collage_from_file_list(file_list, mask_list=None, mask_labels=None, title_list=None,
                           is_vdm=False, do_mip=False, do_minip=False, image_axis=2, post_processing_function=None):
    # set up number of rows and columns
    n = len(file_list)
    n_row = np.int_(np.floor(np.sqrt(n)))
    n_col = np.int_(np.ceil(n / n_row))

    # initialize figure
    plt.rcParams.update({'font.size': 6,
                         'legend.fontsize': 6,
                         'figure.figsize': [10.0, 10.0]})
    fig, axs = plt.subplots(nrows=n_row, ncols=n_col)
    axs = axs.ravel()
    # loop over all files
    for i, fn_data in enumerate(file_list):
        # load image data
        image = load_as_np_array(fn_data)
        # optional post processing (external function)
        if post_processing_function is not None:
            image = post_processing_function(image)
        # load mask and crop
        if mask_list is not None:
            mask = load_as_np_array(mask_list[i])
            # if mask has multiple labels, extract only the selected one; otherwise select all non-zero values
            if mask_labels is None:
                mask = mask > 0
            else:
                mask = np.isin(mask, mask_labels)
            image = crop_to_mask(image, mask, border_in_voxels=0)
        # plot
        if is_vdm:
            vdm_show(axs[i], image, axis=image_axis, colorbar_active=False)
        else:
            if len(image.shape) == 2:
                axs[i].imshow(image, cmap="gray")
            elif do_mip:
                axs[i].imshow(create_mip(image, mip_start=0, mip_end=image.shape[image_axis],
                                         projection_axis=image_axis), cmap="gray")
            elif do_minip:
                if mask_list is not None:  # prevent all zero projection when mask used
                    image[crop_to_mask(mask, mask, border_in_voxels=0) == 0] = np.max(image)  # need to crop mask too
                axs[i].imshow(create_minip(image, mip_start=0, mip_end=image.shape[image_axis],
                                           projection_axis=image_axis), cmap="gray")
            else:
                # central slice index
                slc_ind = np.int_(image.shape[image_axis] // 2)
                # shuffle image axis to last dimension and plot
                axs[i].imshow(np.moveaxis(image, image_axis, -1)[:, :, slc_ind], cmap="gray")
        # add title, turn off axis...
        if title_list is None:
            axs[i].set_title(str(i + 1))
        else:
            axs[i].set_title(title_list[i])
        axs[i].set_axis_off()
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=0.5)


def segmentation_overview_plot(image, filtered, segmentation, vessel_radius):
    # figure settings
    plt.rcParams.update({'font.size': 12,
                         'legend.fontsize': 14,
                         'figure.figsize': [6.0, 6.0]})

    # if 3D, plot maximum intensity projections
    if len(image.shape) == 3:
        image = np.max(image, axis=2)
        filtered = np.max(filtered, axis=2)
        segmentation = np.max(segmentation, axis=2)
        vessel_radius = np.max(vessel_radius, axis=2)

    cmap = plt.get_cmap('gray')
    fig, axs = plt.subplots(ncols=2, nrows=2)
    axs = axs.ravel()
    axs[0].imshow(image, cmap=cmap)
    axs[0].set_title('image')
    axs[1].imshow(filtered, cmap=cmap)
    axs[1].set_title('vesselness filter output')
    axs[2].imshow(segmentation, cmap=cmap)
    axs[2].set_title('binary segmentation')
    axs[3].imshow(vessel_radius, cmap="hot")
    axs[3].set_title('skeleton with \nvessel_radius')
    for i in range(0, len(axs)):
        axs[i].set_axis_off()


def bar_plot_volume_fraction(ax, volume_fraction, vessel_label_names, cmap_str="Reds"):
    # create a stacked bar plot for all volume-fractions (i.e. supply or collateral
    # vessel_label_names is a list of strings containing the names of the vessels analyzed
    # volume_fraction[subj x branch_labels]
    n_subjects, n_seg_labels = volume_fraction.shape
    # set index, width, colors and bottom offset
    ind = np.arange(1, n_subjects + 1)
    width = 0.3
    colors = cm.get_cmap(cmap_str, n_seg_labels + 1)  # we omit the first color (would be white)
    bottom = np.zeros(n_subjects)  # required as offset for stacked bar plot
    # plot stacked bar plot
    for i, label in enumerate(vessel_label_names):
        ax.bar(ind, volume_fraction[:, i], width, bottom=bottom, label=label, color=colors(i + 1))
        bottom += volume_fraction[:, i]  # cumulative offset
    # set y-label and legend
    ax.set_ylabel('Volume fraction')
    ax.legend(loc='lower left', ncol=n_seg_labels)


def vdm_hog_mag_and_phase_imshow(mag, phi, theta=None, slc_nr=None):
    # check slice number
    if (len(mag.shape) == 3) & (slc_nr is None):
        # use central slice in z-direction
        slc_nr = np.int_(mag.shape[2] // 2)  # // == floor division

    # extract image slices to be plotted
    if len(mag.shape) == 3:
        slc_mag = mag[:, :, slc_nr]
        slc_phi = phi[:, :, slc_nr]
        if theta is not None:
            slc_theta = theta[:, :, slc_nr]
    else:
        slc_mag = mag
        slc_phi = phi
        if theta is not None:
            slc_theta = theta

    # check if theta is provided
    if theta is None:
        n_cols = 2
    else:
        n_cols = 3

    # plot
    map_mag = "gray"
    map_phase = "hsv"
    fig, axs = plt.subplots(ncols=n_cols)
    axs[0].imshow(slc_mag, cmap=map_mag)
    axs[1].imshow(slc_phi, cmap=map_phase)
    if theta is not None:
        axs[2].imshow(slc_theta, cmap=map_phase)


def _get_axis_size_in_pixel(ax, fig=None):
    if fig is None:
        fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    return bbox.width * fig.dpi, bbox.height * fig.dpi


def _icon_scaling_for_dendrogram(available_space_total, icon_size, number_of_icons):
    # compute scaling as available_space_per_icon/actual_icon_size
    return 0.6 * ((available_space_total / number_of_icons) / icon_size)  # FYI: multiply with margin factor


def plot_dendrogram(model, ax=None, title=None, orientation="top",
                    icon_png_file_list=None, icon_cmap="gray", icon_loc_offset=None,
                    color_threshold=None, color_above_threshold="k",
                    **kwargs):
    # Wrapper to scikit-learn dendrogram;
    # Use to visualize hierarchical clustering results as a full tree (leave nodes are single data points/subjects)
    # icon_png_file_list needs to be order by subject index (a.k.a. input order for the clustering

    # Compute linkage matrix
    #   inspired by: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    # number of samples / subjects / datapoints
    n_samples = len(model.labels_)
    # initialize the counts i.e. samples under each node (at least 2 for having too leaf nodes, hence, original samples
    #   FYI:children_.shape[0] stores the number of performed steps; for full tree = n_samples - 1
    counts = np.zeros(model.children_.shape[0])  # childeren_ stores the "merging" process
    # create the counts of samples under each node (for each step, check what samples/nodes have been merged)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node; hence, original data point
            else:
                current_count += counts[child_idx - n_samples]  # current node is not a leaf node; add count from child
        counts[i] = current_count
    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # if threshold provided ...
    if color_threshold is not None:
        # set colors
        set_link_color_palette(plt.rcParams['axes.prop_cycle'].by_key()['color'])  # ...  standard color

    # Plot the corresponding dendrogram
    #   FYI: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html
    R = dendrogram(linkage_matrix, orientation=orientation, ax=ax,
                   above_threshold_color=color_above_threshold, color_threshold=color_threshold, no_labels=False,
                   **kwargs)

    # set axis, layout and title
    plt.tight_layout()
    if ax is None:
        ax = plt.gca()
    if title is None:
        title = "Dendrogram"
    ax.set_title(title)

    # Overlay icons of each leaf node
    if icon_png_file_list is not None:
        # get label location, icon scaling and, if required, set icon offset
        if (orientation == "left") | (orientation == "right"):
            label_loc = ax.get_yticks()
            icon_size = load_as_np_array(icon_png_file_list[0]).shape[0]
            total_available_space = _get_axis_size_in_pixel(ax, fig=None)[1]
            icon_scaling = _icon_scaling_for_dendrogram(total_available_space, icon_size, len(icon_png_file_list))
            if icon_loc_offset is None:
                icon_loc_offset = np.int_(0.1 * ax.get_xticks()[-1])
        else:  # "top" or "bottom
            label_loc = ax.get_xticks()
            icon_size = load_as_np_array(icon_png_file_list[0]).shape[1]
            total_available_space = _get_axis_size_in_pixel(ax, fig=None)[0]
            icon_scaling = _icon_scaling_for_dendrogram(total_available_space, icon_size, len(icon_png_file_list))
            if icon_loc_offset is None:
                icon_loc_offset = np.int_(0.1 * ax.get_yticks()[-1])
        # get label order after dendogram plot
        icon_order = R["leaves"]

        # loop over all labels and files
        for i, current_loc in enumerate(label_loc):
            # extract current filename (input order changed by dendrogram plot
            filename = icon_png_file_list[icon_order[i]]
            # load corresponding icon
            icon = OffsetImage(plt.imread(filename), zoom=icon_scaling, cmap=icon_cmap)
            # add annotation box to plot icon
            if (orientation == "right") | (orientation == "left"):
                ab = AnnotationBbox(icon, (icon_loc_offset, current_loc), frameon=False, pad=0.0)
            else:  # "top" or "bottom
                ab = AnnotationBbox(icon, (current_loc, icon_loc_offset), frameon=False, pad=0.0)
            ax.add_artist(ab)
    # return R from the dendogram plot
    return R


def plot_dendrogram_comparison(model1, model2, title1=None, title2=None,
                               icon_png_file_list1=None, icon_png_file_list2=None,
                               icon_cmap="gray", icon_loc_offset=None, **kwargs):
    # plot two dendrograms;
    fig, axs = plt.subplots(nrows=2)

    R1 = plot_dendrogram(model1, orientation="top", ax=axs[0], title="",
                         icon_png_file_list=icon_png_file_list1,
                         icon_cmap=icon_cmap, icon_loc_offset=icon_loc_offset,
                         **kwargs)
    axs[0].set_title(title1, y=0.99)
    R2 = plot_dendrogram(model2, orientation="bottom", ax=axs[1], title="",
                         icon_png_file_list=icon_png_file_list2,
                         icon_cmap=icon_cmap, icon_loc_offset=icon_loc_offset,
                         **kwargs)
    axs[1].set_title(title2, y=-0.01)
    # return both Rs from the dendogram plots
    return R1, R2


def add_regression_line_to_plot(ax, x_values, m_value, n_value, regression_label=None,
                                line_color="k", line_style="solid", line_width=1.5):
    ax.plot(x_values, m_value * np.asarray(x_values) + n_value, label=regression_label,
            color=line_color, linestyle=line_style, linewidth=line_width)


def create_significance_stars_from_p_value(p_value, significance_levels_list=(0.1, 0.01, 0.001)):
    significance_string = ""
    for sign_level in significance_levels_list:
        if p_value < sign_level:
            significance_string = significance_string + "*"
    return significance_string


def __helper_value_to_string(value, precision):
    return f"{value:.{precision}f}"


def correlation_plot_from_data_frame(df, variable_names, variable_labels, ax=None,
                                     correlation_method="Pearson", correct_multi_comparisons=True,
                                     significance_levels_list=(0.1, 0.01, 0.001), do_annotation=True):
    # init
    if ax is None:
        fig, ax = plt.subplots()
    corr_matrix = np.zeros((len(variable_names), len(variable_names)))
    p_matrix = np.zeros_like(corr_matrix)
    annot_list = []
    # create mask to get only lower triangle
    mask = np.zeros_like(corr_matrix, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True
    # compute number of "true" comparisons: several comparisons redundant due to mat.sym.
    if correct_multi_comparisons:
        _n_correction = np.sum(mask) - len(variable_names)  # mask ( == triangle) - diagonal ( == identity matrix)
    else:
        _n_correction = 1
    # compute correlation and p-value matrix
    for ii, metric_x in enumerate(variable_names):
        for jj, metric_y in enumerate(variable_names):
            corr_matrix[ii, jj], p_matrix[ii, jj] = correlation_with_bonferroni(df[metric_x], df[metric_y],
                                                                                correlation_method, _n_correction)
            significance_star_str = create_significance_stars_from_p_value(p_matrix[ii, jj], significance_levels_list)
            annot_list.append(f"{__helper_value_to_string(corr_matrix[ii, jj], 3)}{significance_star_str}")
    # create annotation matrix if requested, i.e. plot the R2 + significance
    if do_annotation:
        annot_matrix = np.array(annot_list).reshape(corr_matrix.shape)
    else:
        annot_matrix = None
    # plot
    _heatmap = sns.heatmap(corr_matrix, ax=ax, mask=mask, square=True, linewidths=0.5,
                           cmap="coolwarm", cbar_kws={'shrink': 0.8, "ticks": [-1, -0.5, 0, 0.5, 1]},
                           vmin=-1, vmax=1, annot=annot_matrix, fmt="", annot_kws={"size": 7})
    # add the column names as labels
    ax.set_xticks(np.arange(len(variable_labels)) + 0.5)
    ax.set_yticks(np.arange(len(variable_labels)) + 0.5)
    ax.set_yticklabels(variable_labels, rotation=0)
    ax.set_xticklabels(variable_labels, rotation=45, ha="right")
    ax.set_title("Pearson correlation matrix")
    sns.set_style({'xtick.bottom': True}, {'ytick.left': True})

    return corr_matrix, p_matrix, _heatmap
