# Automatic reformatting to LV short-axis view (SAX)
Author: Marta Nuñez-Garcia (marnugar@gmail.com)

## About
Implementation of the method described in:
[*Automatic multiplanar CT reformatting from trans-axial into left ventricle short-axis view*. Marta Nuñez-Garcia et al. STACOM (2020)](https://link.springer.com/chapter/10.1007/978-3-030-68107-4_2). Please cite this reference when using this code. PDF available here: [hal.inria.fr](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjM2O2xqaz1AhUP2BQKHR95AyIQFnoECAsQAQ&url=https%3A%2F%2Fhal.inria.fr%2Fhal-02961500%2Fdocument&usg=AOvVaw2t4ZjZm5ZgfdZa1cxhlp8w)

Given a raw trans-axial (TA) CT image and the corresponding LV endo, LV wall and RV epi segmentations (.mha or .vtk), compute LV short axis view image. It also reformats the masks.

Get meshes from masks and use them to find the transformation that aligns:
  1. MV plane
  2. Septum (RV position with regard to LV position)
  3. LV long axis
to the corresponding theoretical planes in standard short-axis view.
Manually set image parameters in the beginning: image size (nb of voxels), spacing, keep_physical_location = True/Fals

Resampling diagram:

![resampling diagram](https://github.com/martanunez/SAX_reformatting/blob/main/diagram_resampling.png)

Schematic pipeline:

![Schematic pipeline](https://github.com/martanunez/SAX_reformatting/blob/main/schematic_pipeline.png)

## Note
With respect to the method presented in the paper, this code additionally includes:
  - a 4th rotation (suggested and implemented by Nicolas Cedilnik) that improves LV septum alignment: after a  preliminary reformat to sax, use LV endo and LV epi masks (a slice midway along the long axis) to compute LV and RV centers and get the rotation matrix that will place the RV to the left of the LV
  - The option of keeping the physical location (approx). Manually modify the 'keep_physical_location' variable to 'True'. Otherwise, the default behaviour is to set the output image origin = (0,0,0).
  - The option of performing an initial automatic check of complete LV in TA view. If check_lv_cropped = True, exit if not complete (cropped) LV. 
  - Automatic check of potential appex cropping with current spacing and spacing modification if necessary.

## Extras
A couple of additional functionalities are also included:
- Basic Quality Control (QC) of the result: check final LV long axis direction (on a slightly different mesh) and compare it to the theoretical one.
- Compute LV wall parcellation: 17-AHA (according to ["oficial" definition](https://www.pmod.com/files/download/v34/doc/pcardp/3615.htm), notably, taking into account "Only slices containing myocardium in all 360° are included", i.e. part of the base is excluded); and a similar parcellation without excluding that part (we will use that one mainly).
- Compute LV mesh parcellation. Similar to previous point but computing the parcellation directly on the mesh.  


## Code
[Python](https://www.python.org/)

Required packages: NumPy, pyacvd, SimpleITK, VTK, pyvista. 

## Instructions
Clone the repository:
```
git clone https://github.com/martanunez/SAX_reformatting

cd SAX_reformatting
```

## Usage
```
python main.py  [-h] [--path PATH] [--ct_im FILENAME] 
                [--mask_lvendo FILENAME] [--mask_lvwall FILENAME] [--mask_rvepi FILENAME] 
                [--save SAVE] [--isotropic ISO]

Arguments:
  -h, --help          Show this help message and exit
  --path              Path to folder with input data
  --ct_im             Input CT image name
  --mask_lvendo       Input LV endo mask name
  --mask_lvwall       Input LV wall mask name
  --mask_rvepi        Input RV epi mask name
  --save              Save intermediate results (rotated meshes etc.)
  --isotropic         Output must be isotropic

```

## Usage example
```
python main.py --path example_pat0/ --ct_im ct.mha --mask_lvendo ct-lvendo.mha --mask_lvwall ct-lvwall.mha --mask_rvepi ct-rvepi.mha
```
