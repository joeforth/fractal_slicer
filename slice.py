from fracprint import processor
from fracprint import writer

# Parameters to vary
filedir = './test_files/'  # Directory where the files are stored
# filename = 'test_squiggle_3d.txt'       # Name of the file you're loading
filename = 'test_squiggle_2d.csv'
x_dim, y_dim = 22., 22.       # x, y dimensions of the container you're printing into
x_trans, y_trans = 0., 0.    # Distance by which to translate the pattern
inlet_d = 0.                 # Length of the inlet / outlet ports in mm
res = 0.1  # Pattern resolution in mm

# x_all_shift, y_all_shift = processor.shape_prep(filedir, filename, x_dim, y_dim, inlet_d, x_trans, y_trans, res)
processor.shape_prep(filedir, filename, x_dim, y_dim, inlet_d, x_trans, y_trans, res)
