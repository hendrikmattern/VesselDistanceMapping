import nibabel as nib
import numpy as np
import h5py
from PIL import Image


def _remove_ext_from_fn(fn, ext='.nii.gz'):
    return fn.split(ext)[0]


def load_nii_as_np_array(filename):
    nii_file = nib.load(filename)
    return np.squeeze(nii_file.get_fdata()), nii_file.affine, nii_file.header


def save_np_array_as_nii(np_array, affine, header, filename):
    # if bool, convert to int16 (otherwise ImageJ cannot open data)
    if np_array.dtype == bool:
        np_array = np.int16(np_array)
    # check if affine is None
    if affine is None:
        affine = np.eye(4)  # identity matrix
    # check if header is None
    if header is None:
        empty_header = nib.Nifti1Header()  # dummy to get correct header in subsequent step
        header = nib.Nifti1Image(np_array, affine, empty_header).header
    # set default min and max display values
    header["cal_min"] = np.min(np_array)
    header["cal_max"] = np.max(np_array)
    # create nii object
    nii = nib.Nifti1Image(np_array, affine, header)
    # todo check out this if problems with saving
    #  hdr_tof['datatype'] = 768 # unsigned int
    #  nib.Nifti1Image(vessel_seg.astype('int'), affine, header)
    # set data type
    nii.set_data_dtype(np_array.dtype)
    # save
    nib.save(nii, filename)


def get_voxel_size_from_nii_header(header):
    return header.get_zooms()


def load_h5_as_np_array(filename, dataset_name=None):
    f = h5py.File(filename, 'r')

    if dataset_name is None:
        dset = f[list(f.keys())[0]]
    else:
        dset = f[dataset_name]

    return dset[()]


def convert_nii_to_h5(fn_nii, ext_nii='.nii.gz'):
    # load nii
    data, affine, hdr = load_nii_as_np_array(fn_nii)

    # create empty h5-file
    f = h5py.File(_remove_ext_from_fn(fn_nii, ext_nii) + '.h5', "w")

    # write h5 file; "chunk-ing" can improve performance for e.g. ilastik
    f.create_dataset("vdm_data", data=data, chunks=True)
    f.close()


def save_np_array_as_png(np_array, filename, do_normalization=True):
    # rescale to integer array from 0..255
    if do_normalization:
        np_array = 255*(np_array - np.min(np_array)) / (np.max(np_array) - np.min(np_array))
        np_array = np.uint8(np_array)
    # convert to image and save
    im = Image.fromarray(np_array)
    im.save(filename)


def load_as_np_array(filename):
    # NifTi files
    if (".nii" in filename) | (".nii.gz" in filename):
        # if affine and header required use load_nii_as_np_array instead
        data, _, _ = load_nii_as_np_array(filename)
    # H5 files
    elif (".h5" in filename) | (".hdf5" in filename):
        # if only a specific (sub-)dataset should be loaded use load_h5_as_np_array
        data = load_h5_as_np_array(filename)
    # try to load an image data set
    else:
        try:
            data = np.array(Image.open(filename))
        except IOError:
            print("--- load_as_np_array --- Error: file type not supported or could not find file")
            return -1
    return data
