import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line_nd
from scipy.ndimage import gaussian_filter
from vdm import io


def spherical2cartesian(radius, azimuth, inclination):
    # FYI: radius [0..inf]; azimuth/phi [-pi, pi], x-axis==0 rad; inclination/theta [0..pi], z-axis==0 rad
    x = radius * np.cos(azimuth) * np.sin(inclination)
    y = radius * np.sin(azimuth) * np.sin(inclination)
    z = radius * np.cos(inclination)
    return x, y, z


def spherical2cartesian_in_degree(radius, azimuth, inclination):
    # FYI: radius [0..inf]; azimuth/phi [-180, 180], x-axis==0 degrees; inclination/theta [0..180], z-axis==0 degrees
    x, y, z = spherical2cartesian(np.radians(radius), np.radians(azimuth), np.radians(inclination))
    return np.degrees(x), np.degrees(y), np.degrees(z)


def cartesian2spherical(x, y, z):
    # FYI: radius [0..inf]; azimuth/phi [-pi, pi], x-axis==0 rad; inclination/theta [0..pi], z-axis==0 rad
    radius = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    inclination = np.arctan2(np.sqrt(x**2 + y**2), z)
    return radius, azimuth, inclination


def cartesian2spherical_in_degree(x, y, z):
    # FYI: radius [0..inf]; azimuth/phi [-180, 180], x-axis==0 degrees; inclination/theta [0..180], z-axis==0 degrees
    radius, azimuth, inclination = cartesian2spherical(np.radians(x), np.radians(y), np.radians(z))
    return np.degrees(radius), np.degrees(azimuth), np.degrees(inclination)


def _force_voxel_in_img_vol(voxel, img_vol):
    # dimension have to agree
    if len(voxel) != len(img_vol.shape):
        print("WARNING: dimension mismatch")  # todo: proper warning
    for i, v in enumerate(voxel):
        # has to be positive
        if v < 0:
            voxel[i] = 0
        if v >= img_vol.shape[i]:
            voxel[i] = img_vol.shape[i] - 1  # account for zero-indexing
    return voxel


class VesselBranch:
    """Vessel branch class"""
    def __init__(self, start_voxel, length=1, azimuth_deg=0, inclination_deg=0, diameter=1, max_intensity=1,
                 tortuosity=None):
        self.start_voxel = np.asarray(start_voxel).astype(int)
        self.length = length
        self.azimuth = azimuth_deg
        self.inclination = inclination_deg
        self.diameter = diameter
        self.max_intensity = max_intensity
        self.tortuosity = tortuosity  # todo: implement at some point

    def get_end_point_cartesian(self, length=None):
        # if no length is provided explicitly use the branch length itself
        # FYI: length != self.length used to find position of children branches in the VesselTree class
        if length is None:
            length = self.length
        # compute the end point of the branch/line
        distance_vector = spherical2cartesian_in_degree(length, self.azimuth, self.inclination)
        distance_vector = np.asarray(distance_vector).astype(int)
        return self.start_voxel + distance_vector

    def draw_branch(self, img_vol):
        # check start voxel within image volume
        self.start_voxel = _force_voxel_in_img_vol(self.start_voxel, img_vol)
        # get Cartesian end point from spherical coordinates
        end_voxel = self.get_end_point_cartesian()
        # check end voxel within image volume
        end_voxel = _force_voxel_in_img_vol(end_voxel, img_vol)
        # get line index
        line_index = line_nd(self.start_voxel, end_voxel)
        # create an image of only this branch
        img_only_this_branch = np.zeros_like(img_vol)  # init empty volume
        img_only_this_branch[line_index] = self.max_intensity  # draw line
        img_only_this_branch = gaussian_filter(img_only_this_branch, self.diameter/2)  # Gaussian smoothing
        # add the image of this branch to input (complete) image volume
        return img_vol + img_only_this_branch


class VesselTree(VesselBranch):
    """Vessel tree class which stores all connected branches"""
    def __init__(self, start_voxel, length=1, azimuth_deg=0, inclination_deg=0, diameter=1, max_intensity=1,
                 tortuosity=None):
        # create trunk of vessel tree / main branch by calling base class constructor
        super().__init__(start_voxel, length=length, azimuth_deg=azimuth_deg,
                         inclination_deg=inclination_deg, diameter=diameter,
                         max_intensity=max_intensity, tortuosity=tortuosity)
        # add self/this tree trunk to branch list
        self.branch_list = [self]

    def add_branch(self, parent_branch, relative_branching_point,
                   length=1, azimuth_deg=0, inclination_deg=0, diameter=1, max_intensity=1, tortuosity=None):
        # compute Cartesian start voxel position of new branch
        start_voxel = super().get_end_point_cartesian(length=relative_branching_point * parent_branch.length)
        # create new branch and add to list
        self.branch_list.append(VesselBranch(start_voxel, length=length, azimuth_deg=azimuth_deg,
                                             inclination_deg=inclination_deg, diameter=diameter,
                                             max_intensity=max_intensity, tortuosity=tortuosity))

    def draw_tree(self, img_vol):
        # loop over branch list
        for branch in self.branch_list:
            # draw individual branch and update image volume
            img_vol = branch.draw_branch(img_vol)
        # make sure no branching points are too bright
        img_vol[img_vol > self.max_intensity] = self.max_intensity
        # return final image volume
        return img_vol


class VesselPhantom:
    """Vessel phantom class; stores all vessel trees inside the image volume"""
    def __init__(self, img_vol, noise_sigma=0.0):
        self.img_vol = img_vol
        self.tree_list = []
        self.img_noise = noise_sigma * np.random.randn(*img_vol.shape)  # * to unroll tuple

    def save_as_nii(self, filename, affine=None, header=None):
        # create / draw the latest image volume
        self.draw()
        # save as nii-file
        io.save_np_array_as_nii(self.img_vol, affine=affine, header=header, filename=filename)

    def add_existing_tree(self, tree):
        self.tree_list.append(tree)

    def draw(self):
        # loop over all vessel trees in phantom
        for tree in self.tree_list:
            # draw/add each tree individually into (previous) image volume
            self.img_vol = tree.draw_tree(self.img_vol)
        # add image noise
        self.img_vol = np.abs(self.img_vol + self.img_noise)  # prevent negative intensities
