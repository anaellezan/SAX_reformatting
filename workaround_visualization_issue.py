# Manually set image Direction to ease data visualization
# CAREFUL, we'll lose matching world coordinates with initial TA image (not relevant in most cases...)

import SimpleITK as sitk

# im_filename = '/home/marta/Desktop/DATA/Hugo/weird_case/image_0-sax.mha'
# mask_filename = '/home/marta/Desktop/DATA/Hugo/weird_case/image_0_lvendo-sax.mha'


# im_filename = '/home/marta/Desktop/DATA/Sonny/ICD_cancel_TA/wrong-sax/0101597655/ct1-sax-iso.mha'
# mask_filename = '/home/marta/Desktop/DATA/Sonny/ICD_cancel_TA/wrong-sax/0101597655/ct1-lvendo-sax-iso.mha'

#im_filename = '/home/marta/Desktop/DATA/Sonny/ICD_cancel_TA/wrong-sax/0103193704/ct1-sax-iso.mha'
#mask_filename = '/home/marta/Desktop/DATA/Sonny/ICD_cancel_TA/wrong-sax/0103193704/ct1-lvendo-sax-iso.mha'

im_filename = '/home/marta/Desktop/DATA/Sonny/ICD_cancel_TA/wrong-sax/0126577926/ct1-sax-iso.mha'
mask_filename = '/home/marta/Desktop/DATA/Sonny/ICD_cancel_TA/wrong-sax/0126577926/ct1-lvendo-sax-iso.mha'

im_sax = sitk.ReadImage(im_filename)
mask_sax = sitk.ReadImage(mask_filename)

print('Computed direction (not easy to display)', im_sax.GetDirection())
new_direction = [1, 0, 0,
                 0, 0, -1,
                 0, 1, 0]

im_sax.SetDirection(new_direction)
mask_sax.SetDirection(new_direction)

sitk.WriteImage(im_sax, im_filename[:-4] + '-ease-vis.mha')
sitk.WriteImage(mask_sax, mask_filename[:-4] + '-ease-vis.mha')