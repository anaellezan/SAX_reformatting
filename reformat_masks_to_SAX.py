# Given R transformation matrix, masks in TA view, and the image already reformatted to SAX, get the
# corresponding masks in SAX view.
# Careful: outputs from Nico's segmentation method are:
#   - masks with [0, 1] value. The resampling done during the reformatting is not accurate (especially in extremely
#   thin regions (typically apical region)) due to numerical interpolation -> change the values to [0, 255] in advance,
#   resample, and then threshold the result.
#   - LV endo and LV wall. Resampling the wall is again tricky in thin regions. Also, I need epi and endo (and not wall)
#   for computing wall thickness with Nico's method -> create epi (endo + wall) & resample endo and epi (compact shape
#   -> less errors due to resampling). I will then get LV wall as epi - endo (it is not necessary in this 'thickness'
#   pipeline but for other cases...)

# Reformat also RV epi

# conda activate sax_reformatting

import os
import sys
from aux_functions import *


def change_masks(mask):
    """Change mask labels from [0,1] to [0, 255]"""
    np_mask = sitk.GetArrayFromImage(mask)
    np_mask[np.where(np_mask == 1)] = 255
    mask_out = sitk.GetImageFromArray(np_mask)
    mask_out = sitk.Cast(mask_out, sitk.sitkUInt8)
    mask_out.SetSpacing(mask.GetSpacing())
    mask_out.SetOrigin(mask.GetOrigin())
    mask_out.SetDirection(mask.GetDirection())
    return mask_out

def np_to_im(np_im, ref_im, pixel_type):
    im_out = sitk.GetImageFromArray(np_im)
    im_out = sitk.Cast(im_out, pixel_type)
    im_out.SetSpacing(ref_im.GetSpacing())
    im_out.SetOrigin(ref_im.GetOrigin())
    im_out.SetDirection(ref_im.GetDirection())
    return im_out


prefix_path = 'example_pat0/'
name = 'ct'

# inputs
lvendo_filename = prefix_path + name + '-lvendo.mha'
lvwall_filename = prefix_path + name + '-lvwall.mha'
rvepi_filename = prefix_path + name + '-rvepi.mha'
im_sax_filename = prefix_path + name + '-sax.mha'
r_filename = prefix_path + name + '-R-matrix-sax.txt'

# outputs
lvendo_sax_filename = prefix_path + name + '-lvendo-sax.mha'
lvepi_sax_filename = prefix_path + name + '-lvepi-sax.mha'
lvwall_sax_filename = prefix_path + name + '-lvwall-sax.mha'
rvepi_sax_filename = prefix_path + name + '-rvepi-sax.mha'


if not os.path.isfile(lvendo_filename) or not os.path.isfile(lvwall_filename) or not os.path.isfile(rvepi_filename) :
    sys.exit('One or several segmentations are missing, please check filenames.')
if not os.path.isfile(r_filename):
    sys.exit('txt file with rotation matrix is missing, please check the filename.')
if not os.path.isfile(im_sax_filename):
    sys.exit('CT image in SAX view is missing, please check the filename')


# compute LV epi mask.
lvendo_TA = sitk.ReadImage(lvendo_filename)
lvwall_TA = sitk.ReadImage(lvwall_filename)
add = sitk.AddImageFilter()
lvepi_TA = add.Execute(lvendo_TA, lvwall_TA)
## fill small holes between them. This closing may also help in filling small holes in extremely thin wall regions
# This creates wall voxels in the base... avoid from now
# lvepi_TA = sitk.BinaryMorphologicalClosing(lvepi_TA, np.array([2, 2, 2], dtype='int').tolist())  # conversion to list using python3
# sitk.WriteImage(lvepi_TA, lvepi_filename)

# Read already computed transformation
R = np.loadtxt(r_filename)

ref_sax = sitk.ReadImage(im_sax_filename)
sax_size = ref_sax.GetSize()[0]   # only one, then compute_reference_images does: reference_size = [size] * dimension
reference_origin = ref_sax.GetOrigin()
reference_spacing = ref_sax.GetSpacing()

reference_image, reference_center = compute_reference_image(ref_sax, size=sax_size, spacing=reference_spacing, reference_origin=reference_origin)

lvendo255_sax = get_sax_view(change_masks(lvendo_TA), reference_image, reference_origin, reference_center, R, default_pixel_value=0)
np_lvendo255_sax = sitk.GetArrayFromImage(lvendo255_sax)
np_lvendo255_sax[np.where(np_lvendo255_sax < 128)] = 0
np_lvendo255_sax[np.where(np_lvendo255_sax >= 128)] = 1
lvendo_sax = np_to_im(np_lvendo255_sax, ref_im=lvendo255_sax, pixel_type=sitk.sitkUInt8)
sitk.WriteImage(lvendo_sax, lvendo_sax_filename, True)

lvepi255_sax = get_sax_view(change_masks(lvepi_TA), reference_image, reference_origin, reference_center, R, default_pixel_value=0)
np_lvepi255_sax = sitk.GetArrayFromImage(lvepi255_sax)
np_lvepi255_sax[np.where(np_lvepi255_sax < 128)] = 0
np_lvepi255_sax[np.where(np_lvepi255_sax >= 128)] = 1
lvepi_sax = np_to_im(np_lvepi255_sax, ref_im=lvepi255_sax, pixel_type=sitk.sitkUInt8)
sitk.WriteImage(lvepi_sax, lvepi_sax_filename, True)

rvepi_TA = sitk.ReadImage(rvepi_filename)
rvepi255_sax = get_sax_view(change_masks(rvepi_TA), reference_image, reference_origin, reference_center, R, default_pixel_value=0)
np_rvepi255_sax = sitk.GetArrayFromImage(rvepi255_sax)
np_rvepi255_sax[np.where(np_rvepi255_sax < 128)] = 0
np_rvepi255_sax[np.where(np_rvepi255_sax >= 128)] = 1
rvepi_sax = np_to_im(np_rvepi255_sax, ref_im=rvepi255_sax, pixel_type=sitk.sitkUInt8)
sitk.WriteImage(rvepi_sax, rvepi_sax_filename, True)

np_lvwall_sax = np_lvepi255_sax - np_lvendo255_sax   # only 0 and 1 hopefully... shoudn't have -1...
if len(np.unique(np_lvwall_sax)) > 2:
    print('Values in LV wall mask: ', np.unique(np_lvwall_sax))
lvwall_sax = np_to_im(np_lvwall_sax, ref_im=rvepi255_sax, pixel_type=sitk.sitkUInt8)
sitk.WriteImage(lvwall_sax, lvwall_sax_filename, True)

       
