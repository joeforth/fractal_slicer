# Import and define useful modules
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.seterr(invalid='ignore')   # Suppress divide by zero error


def typer(string_in):
    if 'LINESTRING' in string_in:
        start = string_in.find('(')
        end = string_in.find(')')
        string_in = string_in[start+1:end]
        string_in = string_in.split(',')
    
    if 'POLYGON' in string_in:
        start = string_in.find('((')
        end = string_in.find('))')
        string_in = string_in[start+2:end]
        string_in = string_in.split(',')
        
    if 'POINT' in string_in:
        print('POINT detected in wkt file - these are currently ignored')
        string_in = ''
   
    return string_in


def fileread(name):
    # Read target wkt/svg file, sanitise filename, split into elements
    with open(name) as f:
        data = f.read()    
    name = name[:-4]   # Remove .csv suffix
    data = data.split('\n')
    data = data[1:len(data)-1]
    
    return data
      

def pre_coordinater(data_full):
    # Split wkt file into elements, return x and y coordinates 
    # Each element corresponds to a different element of the wkt
    x, y = [], []
    for j in range(0, len(data_full)):
        data = data_full[j]
        form = typer(data)
        coords = coordinater(form)
        if len(coords[0]) == 0:
            print('Ignoring point')
        if len(coords[0]) > 0:
            x.append(coords[0])
            y.append(coords[1])
    return x, y


def coordinater(string_in):
    x, y = [], []
    # Now run through a make x- and y- coordinates
    for i in range(0, len(string_in)):
        fragment = string_in[i].split()
        x.append(round(float(fragment[0]), 1))
        y.append(round(float(fragment[1]), 1))

    # Now truncate it so that you only have coords separated by 100 µm
    for i in range(0, len(x)-1)[::-1]:
        if cartesian(x[i], y[i], x[i+1], y[i+1]) < 0.1:
            del x[i]
            del y[i]
    return [x, y]


def cartesian(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def bbox_calculator(x_in, y_in):
    # Takes a list of lists of x- and y-coordinates and calculates:
    # x_com, y_com - centre of mass co-ordinates
    # x_min, x_max, y_min, y_max - extremal values of coordinates in the file
    # bbox_x, bbox_y - bounding box dimensions
    x_flat = [item for sublist in x_in for item in sublist]
    y_flat = [item for sublist in y_in for item in sublist]
    x_com_calc, y_com_calc = np.mean(x_flat), np.mean(y_flat)
    x_min_calc, x_max_calc = np.amin(x_flat), np.amax(x_flat)
    y_min_calc, y_max_calc = np.amin(y_flat), np.amax(y_flat)
    bbox_x_calc = round((x_max_calc - x_min_calc), 2)
    bbox_y_calc = round((y_max_calc - y_min_calc), 2)
    
    return bbox_x_calc, bbox_y_calc, x_com_calc, y_com_calc, x_min_calc, x_max_calc, y_min_calc, y_max_calc
    

def interpolator(x_in, y_in, res):
    # Replot pattern with point spacing = res mm
    # Interpolator that plots the pattern at 100 µm spacing
    x_out, y_out = [x_in[0]], [y_in[0]]
    diff_list = []
    # First - test if there are any points that need interpolation performing
    for i in range(1, len(x_in)):
        s = cartesian(x_in[i], y_in[i], x_in[i-1], y_in[i-1])
        if s <= res:
            x_out = np.append(x_out, x_in[i])
            y_out = np.append(y_out, y_in[i])
        if s > res:
            n_points = math.floor(s/res)
            # Don't include i-1 as that's already been added
            new_points_x = np.linspace(x_in[i-1], x_in[i], num=n_points)[1:]
            new_points_y = np.linspace(y_in[i-1], y_in[i], num=n_points)[1:]
            x_out = np.append(x_out, new_points_x)
            y_out = np.append(y_out, new_points_y)
    return x_out, y_out


def param_calculator(x_in, y_in, x_dim_in, y_dim_in, x_trans_in, y_trans_in):
    # Autoscales to fit a pre-determined bounding box (defined by x_dim and y_dim)
    bbox_x_out, bbox_y_out, x_com_out, y_com_out, x_min_out, x_max_out, y_min_out, y_max_out = bbox_calculator(x_in, y_in)
    
    if bbox_x_out > 0:
        x_scale = x_dim_in / bbox_x_out
    if bbox_x_out == 0:
        print('Zero pattern width detected - scaling by 1')
        x_scale = 1.
    
    if bbox_y_out > 0:
        y_scale = y_dim_in / bbox_y_out
    if bbox_y_out == 0:
        print('Zero pattern height detected - scaling by 1')
        y_scale = 1.
    
    # Rescales and shifts all input coordinates
    x_all_shift_scale, y_all_shift_scale = [], []
    for i in range(0, len(x_in)):
        x_all_shift_scale.append((x_in[i] - x_com_out)*x_scale)
        y_all_shift_scale.append((y_in[i] - y_com_out)*y_scale)
        
    # Finally, apply an x-y translation to manually shift the centre of the print, if desired
    x_all_shift_out, y_all_shift_out = [], []
    for i in range(0, len(x_in)):
        x_all_shift_out.append((x_all_shift_scale[i] + x_trans_in))
        y_all_shift_out.append((y_all_shift_scale[i] + y_trans_in))
        
    return x_all_shift_out, y_all_shift_out, x_com_out, y_com_out, x_min_out, x_max_out, y_min_out, y_max_out, bbox_x_out, bbox_y_out


def E_calculator(x_in, y_in, res_in, d_in, exp_in, offset_in, alpha_in):
    # Takes a set of x-coords, y-coords, a resolution (res), a fibril diamter (d_in), 
    # power law exponent (exp_in), offset (distance from end of path to start flow slowdown, 
    # and volume conversion factor (alpha_in) 
    # Returns:
    # E_fac - factor by which extrusion rate is reduced as you move along a shape
    # E_diff - Volume extruded between each set of data points in a given shape
    # s_list - Cumulative distance extruded for this shape
    # s_diff - Distance between each set of datapoints

    print('Point 1, x_in is', x_in)    
    # Cumulative extrusion, cumulative distance, and differential distance
    E_list, s_list, s_diff = [0], [0], []
    # Run through and calculate distances
    for i in range(1, len(x_in)):
        # Calculate distance travelled and extruded volume
        s = cartesian(x_in[i], y_in[i], x_in[i-1], y_in[i-1])
        s_diff.append(s)
        s_list.append(s + s_list[-1])
    print('Point 2, s_list is', s_list)
    print('Point 2, s_diff is', s_diff)

    # Calculate path length distance of a given point from the end
    s_list, s_diff = np.array(s_list), np.array(s_diff)
    # Appears to be caused by the last element in s_diff being greater than offset_in + 0.1
    dists = abs(s_list[-1] - s_list)   # Distance of each point from end of path
    dists = np.array(dists[1:])   # dists is calculated using s_list - so you have one-too-many data points
    pts_to_mod = np.where(dists < offset_in + 0.1)   # Points within a threshold distance of end of path
    # Calculate the 'x'-axis of the power law plot - factor of -1 shifts it to a "distance from end"
    s_in = -1*dists[pts_to_mod]
    E_x = power_law(s_in, exp=exp_in, shift=-1*s_in[0])   # Factor by which to slow extrusion
    E_fac = np.ones(len(dists))   # Factor by which to multiply E
    E_fac[pts_to_mod] = E_fac[pts_to_mod]*E_x   # Project E_x onto an array the same length as the extrusion list
    E_diff = E_fac*alpha_in*np.pi*(0.1*s_diff)*((d_in/20.)**2)   # E_fac*differential distance to extrude for each step
    return E_fac, E_diff, s_list, s_diff


def plot_out(x_all_in, y_all_in, x_all_shift_in, y_all_shift_in, inlet_d, x_min_shift, x_max_shift):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    for i in range(0, len(x_all_in)):
        ax1.plot(x_all_in[i], y_all_in[i], lw = 4)
    
    ax2 = fig.add_subplot(122)
    for i in range(0, len(x_all_shift_in)):
        ax2.plot(x_all_shift_in[i], y_all_shift_in[i], lw = 4)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax1.set_title('Before Scaling')
    ax2.set_title('After Scaling - with inlets')
    

def power_law(x_in, exp, shift):
    # x_in - x-coords to calculate the power law over
    # exp, array-like - decay exponent (in range 0 to 1, 1 is a linear decrease)
    # shift, float - x-intercept
    # Returns a power law decay with points at 100 µm spacing (= 0.1)
    
    y = 1 - np.power((x_in + shift)/shift, exp)
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(x_in, y)
#     ax.set_xlabel('Distance from end (mm)')
#     ax.set_ylabel('Reduction factor')
    return y


def shape_prep(filename, x_dim, y_dim, inlet_d, x_trans, y_trans, res):
    # Load pattern data
    data_in = fileread(filename)
    x_all, y_all = pre_coordinater(data_in)   # Note that pre-coordinater runs coordinater
    # Calculate centre of mass, min and max x and y values, and bounding box size, autoscales shape to fit in bounding box, applies an x-y translation to change centre of pattern
    x_all_shift, y_all_shift, x_com, y_com, x_min, x_max, y_min, y_max, bbox_x, bbox_y = param_calculator(x_all, y_all, x_dim-2*inlet_d, y_dim, x_trans, y_trans)

    # Calculate size of bounding box after scaling
    bbox_x_shift, bbox_y_shift, x_com_shift, y_com_shift, x_min_shift, x_max_shift, y_min_shift, y_max_shift = bbox_calculator(x_all_shift, y_all_shift)

    # Add inlets and outlets - note the order of this is important, as E_calculator needs finely spaced points to work
    if inlet_d > 0:
        inlet_x, inlet_y = [x_min_shift - inlet_d, x_min_shift], [0, 0]
        outlet_x, outlet_y = [x_max_shift + inlet_d, x_max_shift], [0, 0]
        x_all_shift.insert(0, inlet_x)
        x_all_shift.insert(1, outlet_x)
        y_all_shift.insert(0, inlet_y)
        y_all_shift.insert(1, outlet_y)

    # Interpolate points 
    for j in range(0, len(x_all_shift)):
        x_all_shift[j], y_all_shift[j] = interpolator(x_all_shift[j], y_all_shift[j], res=0.1)

    # Plot output
    plot_out(x_all, y_all, x_all_shift, y_all_shift, inlet_d, x_min_shift, x_max_shift)
    print('Bounding box before scaling =', bbox_x, 'mm x', bbox_y, 'mm')
    print('Bounding box after scaling =', bbox_x_shift + 2*inlet_d, 'mm x', bbox_y_shift, 'mm')

    return x_all_shift, y_all_shift
