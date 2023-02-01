Dor Azriel 
Yosef Simhon


summary:
Python project that train  SVM model for identifying the writer's gender according to the manuscript.

run:
 python main.py idir_name odir_name

idir_name - input directory path
** input directory must have the next hierarchy: **
**                     path/
                     |       |
                   train    tests
                   |   |     |   |
                male female male female
          (tarin images JPG) (test images JPG)  **

**directory path ends with '\' !*

reqierments:
python==3.8
sklearn
skimage
numpy==1.8