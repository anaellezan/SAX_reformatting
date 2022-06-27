# Given R transformation matrix, masks in TA view, and the image already reformatted to SAX, get the
# corresponding masks in SAX view.
# Careful: outputs from our segmentation method are:
#   - masks with [0, 1] value. The resampling done during the reformatting is not accurate (especially in extremely
#   thin regions (typically apical region)) due to numerical interpolation -> change the values to [0, 255] in advance,
#   resample, and then threshold the result back to [0, 1].
#   - LV endo and LV wall. Resampling the wall is again tricky in thin regions. Also, I need epi and endo (and not wall)
#   for computing wall thickness with our method -> create epi (endo + wall) & resample endo and epi (compact shape
#   -> less errors due to resampling). I will then get LV wall as epi - endo.

# Reformat also RV epi

# NOTE version 2 -> first I used conversion to numpy etc, now I changed to shorter/faster/more elegant approach
# using only image masks. Results are identical.

# conda activate sax_reformatting

from aux_functions import *
import time

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
    sys.exit('CT image in SAX view is missing, please check the filename.')


# compute LV epi mask.
lvendo_TA = sitk.ReadImage(lvendo_filename)
lvwall_TA = sitk.ReadImage(lvwall_filename)
add = sitk.AddImageFilter()
lvepi_TA = add.Execute(lvendo_TA, lvwall_TA)

rvepi_TA = sitk.ReadImage(rvepi_filename)

ref_sax = sitk.ReadImage(im_sax_filename)
patient_name = get_patientname(ref_sax)

t = time.time()
lvendo_sax = reformat_mask_to_sax(lvendo_TA, ref_sax, r_filename)
lvendo_sax = add_basic_metadata(lvendo_sax, patient_name, 'sax', 'lvendo')
sitk.WriteImage(lvendo_sax, lvendo_sax_filename, True)

lvepi_sax = reformat_mask_to_sax(lvepi_TA, ref_sax, r_filename)
lvepi_sax = add_basic_metadata(lvepi_sax, patient_name, 'sax', 'lvepi')
sitk.WriteImage(lvepi_sax, lvepi_sax_filename, True)

rvepi_sax = reformat_mask_to_sax(rvepi_TA, ref_sax, r_filename)
rvepi_sax = add_basic_metadata(rvepi_sax, patient_name, 'sax', 'rvepi')
sitk.WriteImage(rvepi_sax, rvepi_sax_filename, True)

lvwall_sax = sitk.SubtractImageFilter().Execute(lvepi_sax, lvendo_sax)
lvwall_sax = sitk.BinaryThreshold(lvwall_sax, 1, 1, 1, 0)   # correct potential -1
lvwall_sax = add_basic_metadata(lvwall_sax, patient_name, 'sax', 'lvwall')
sitk.WriteImage(lvwall_sax, lvwall_sax_filename, True)

print('Elapsed time 1 = ', time.time()-t)     # 4.49 s



# #####   keep v1, using numpy etc, just in case
# t = time.time()
#
# # Read already computed transformation
# R = np.loadtxt(r_filename)
#
# sax_size = ref_sax.GetSize()[0]   # only one, then compute_reference_images does: reference_size = [size] * dimension
# reference_origin = ref_sax.GetOrigin()
# reference_spacing = ref_sax.GetSpacing()
#
# reference_image, reference_center = compute_reference_image(ref_sax, size=sax_size, spacing=reference_spacing, reference_origin=reference_origin)
# patient_name = get_patientname(ref_sax)
#
# lvendo255_sax = get_sax_view(change_masks(lvendo_TA), reference_image, reference_origin, reference_center, R, default_pixel_value=0)
# np_lvendo255_sax = sitk.GetArrayFromImage(lvendo255_sax)
# np_lvendo255_sax[np.where(np_lvendo255_sax < 128)] = 0
# np_lvendo255_sax[np.where(np_lvendo255_sax >= 128)] = 1
# # lvendo_sax = np_to_im(np_lvendo255_sax, ref_im=lvendo255_sax, pixel_type=sitk.sitkUInt8)
# lvendo_sax = np_to_image(img_arr=np_lvendo255_sax, origin=lvendo255_sax.GetOrigin(), spacing=lvendo255_sax.GetSpacing(),
#                          direction=lvendo255_sax.GetDirection(), pixel_type=sitk.sitkUInt8,
#                          name=patient_name, study_description='sax', series_description='lvendo')
# sitk.WriteImage(lvendo_sax, lvendo_sax_filename, True)
#
# lvepi255_sax = get_sax_view(change_masks(lvepi_TA), reference_image, reference_origin, reference_center, R, default_pixel_value=0)
# np_lvepi255_sax = sitk.GetArrayFromImage(lvepi255_sax)
# np_lvepi255_sax[np.where(np_lvepi255_sax < 128)] = 0
# np_lvepi255_sax[np.where(np_lvepi255_sax >= 128)] = 1
# # lvepi_sax = np_to_im(np_lvepi255_sax, ref_im=lvepi255_sax, pixel_type=sitk.sitkUInt8)
# lvepi_sax = np_to_image(img_arr=np_lvepi255_sax, origin=lvepi255_sax.GetOrigin(), spacing=lvepi255_sax.GetSpacing(),
#                          direction=lvepi255_sax.GetDirection(), pixel_type=sitk.sitkUInt8,
#                          name=patient_name, study_description='sax', series_description='lvepi')
# sitk.WriteImage(lvepi_sax, lvepi_sax_filename, True)
#
# rvepi255_sax = get_sax_view(change_masks(rvepi_TA), reference_image, reference_origin, reference_center, R, default_pixel_value=0)
# np_rvepi255_sax = sitk.GetArrayFromImage(rvepi255_sax)
# np_rvepi255_sax[np.where(np_rvepi255_sax < 128)] = 0
# np_rvepi255_sax[np.where(np_rvepi255_sax >= 128)] = 1
# # rvepi_sax = np_to_im(np_rvepi255_sax, ref_im=rvepi255_sax, pixel_type=sitk.sitkUInt8)
# rvepi_sax = np_to_image(img_arr=np_rvepi255_sax, origin=rvepi255_sax.GetOrigin(), spacing=rvepi255_sax.GetSpacing(),
#                          direction=rvepi255_sax.GetDirection(), pixel_type=sitk.sitkUInt8,
#                          name=patient_name, study_description='sax', series_description='rvepi')
# sitk.WriteImage(rvepi_sax, rvepi_sax_filename, True)
#
# np_lvwall_sax = np_lvepi255_sax - np_lvendo255_sax   # only 0 and 1 hopefully... shoudn't have -1...
# if len(np.unique(np_lvwall_sax)) > 2:
#     print('Values in LV wall mask: ', np.unique(np_lvwall_sax))
# # lvwall_sax = np_to_im(np_lvwall_sax, ref_im=rvepi255_sax, pixel_type=sitk.sitkUInt8)
# lvwall_sax = np_to_image(img_arr=np_lvwall_sax, origin=lvendo255_sax.GetOrigin(), spacing=lvendo255_sax.GetSpacing(),
#                          direction=lvendo255_sax.GetDirection(), pixel_type=sitk.sitkUInt8,
#                          name=patient_name, study_description='sax', series_description='lvwall')
# sitk.WriteImage(lvwall_sax, lvwall_sax_filename, True)
#
# print('Elapsed time 2 = ', time.time()-t)    # 20.15 s
