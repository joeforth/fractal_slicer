# Import and define useful modules
import math
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier
np.seterr(invalid='ignore')   # Suppress divide by zero error


# def bbox_calculator(x_in, y_in):
#     # Takes a list of lists of x- and y-coordinates and calculates:
#     # x_com, y_com - centre of mass co-ordinates
#     # x_min, x_max, y_min, y_max - extremal values of coordinates in the file
#     # bbox_x, bbox_y - bounding box dimensions
#     x_flat = [item for sublist in x_in for item in sublist]
#     y_flat = [item for sublist in y_in for item in sublist]
#     x_com_calc, y_com_calc = np.mean(x_flat), np.mean(y_flat)
#     x_min_calc, x_max_calc = np.amin(x_flat), np.amax(x_flat)
#     y_min_calc, y_max_calc = np.amin(y_flat), np.amax(y_flat)
#     bbox_x_calc = round((x_max_calc - x_min_calc), 2)
#     bbox_y_calc = round((y_max_calc - y_min_calc), 2)
    
#     return bbox_x_calc, bbox_y_calc, x_com_calc, y_com_calc, x_min_calc, x_max_calc, y_min_calc, y_max_calc
    

def cartesian2d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def cartesian3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)


def coordinater_wkt(string_in, idx):
    # Takes a set of coordinates in string form, where each coordinate is separated by a space
    # Rounds it to 3 decimal place, and returns it as a list of floats
    # Input - string_in - list of strings, each of which is a set of coordinates
    # Output - [x, y, z, e_id] - list of floats where e_id is element ID
    form = []
    # Now run through a make x- and y- coordinates
    for i in range(0, len(string_in)):
        fragment = string_in[i].split()
        x = round(float(fragment[0]), 3)
        y = round(float(fragment[1]), 3)
        z = 0.0
        e_id = idx
        form.append([x, y, z, e_id])
    return form
 
def distance_calculator(df):
    # Drop NaN values (usually the first value)
    dx = df['x'].diff()
    dy = df['y'].diff()
    dz = df['z'].diff()

    # Calculate the Euclidean distance between consecutive rows
    distances = np.sqrt(dx**2 + dy**2 + dz**2)
    df['distance_from_last'] = distances
    return df


def eulerficator(df, terminal_points, nodes):
    terminal_points_nogroups = terminal_points.reset_index()
    clusters = terminal_points_nogroups['cluster'].unique()
    lines = terminal_points_nogroups['line_id'].unique()

    # Run through each cluster and create two dictionaries 
    # 1 - cluster numbers as key and the connecting lines as values
    # 2 - cluster numbers as key and the number of connecting nodes as values
    cluster_dict = {}
    connectivity_dict = {}
    for c in clusters:
        connecting_lines = terminal_points_nogroups[terminal_points_nogroups['cluster'] == c]
        cluster_dict[c] = list(connecting_lines['line_id'].values)

        connectivity_dict[c] = len(connecting_lines)

    line_order = []
    line_order_grouped = []
    node_order = [] 
    node_order_grouped = []
    while len(line_order) < len(lines):
        current_line_lines = []
        current_line_nodes = []

        # Sort lines by height order - bottom up
        min_z = df.groupby('line_id')['z'].min()
        heightsorted_line_ids = min_z.sort_values().index.tolist()
        unprinted_lines = [n for n in heightsorted_line_ids if n not in line_order]

        # Pick the unprinted line with the lowest z-value to start with
        next_line = unprinted_lines[0]
        line_order.append(next_line)
        current_line_lines.append(next_line)
        unprinted_lines = unprinted_lines[1:] # Remove the printed line from the list of remaining lines

        # Calculate which node you're at - it's the other node to which next_line is connected
        connected_nodes = terminal_points.loc[next_line]['cluster'].values

        # Start at node with lower z-value - record as first node of this line
        start_node = nodes.loc[connected_nodes]['z'].idxmin()
        current_line_nodes.append(start_node)
        node_order.append(start_node)

        # Find node at other end of line
        end_node = connected_nodes[connected_nodes != start_node][0]
        current_line_nodes.append(end_node)
        node_order.append(end_node)

        connected_lines = cluster_dict[end_node]   # Lines connected to end node
        connected_lines = [n for n in unprinted_lines if n in connected_lines]   # List unprinted connected lines in height-order
        while len(connected_lines) > 0:
            # Pick next line based on lowest z_min
            next_line = connected_lines[0]
            line_order.append(next_line)
            current_line_lines.append(next_line)
            unprinted_lines.remove(next_line)

            # Calculate which node you've moved to
            connected_nodes = terminal_points.loc[next_line]['cluster'].values
            start_node = end_node
            end_node = connected_nodes[connected_nodes != start_node][0]
            current_line_nodes.append(end_node)

            connected_lines = cluster_dict[end_node]

            # Remove lines that have already been printed
            connected_lines = [n for n in unprinted_lines if n in connected_lines]
        
        line_order_grouped.append(current_line_lines)
        node_order_grouped.append(current_line_nodes)

    return line_order, node_order, line_order_grouped, node_order_grouped


# def E_calculator(x_in, y_in, res_in, d_in, exp_in, offset_in, alpha_in):
#     # Takes a set of x-coords, y-coords, a resolution (res), a fibril diamter (d_in), 
#     # power law exponent (exp_in), offset (distance from end of path to start flow slowdown, 
#     # and volume conversion factor (alpha_in) 
#     # Returns:
#     # E_fac - factor by which extrusion rate is reduced as you move along a shape
#     # E_diff - Volume extruded between each set of data points in a given shape
#     # s_list - Cumulative distance extruded for this shape
#     # s_diff - Distance between each set of datapoints

#     print('Point 1, x_in is', x_in)    
#     # Cumulative extrusion, cumulative distance, and differential distance
#     E_list, s_list, s_diff = [0], [0], []
#     # Run through and calculate distances
#     for i in range(1, len(x_in)):
#         # Calculate distance travelled and extruded volume
#         s = cartesian(x_in[i], y_in[i], x_in[i-1], y_in[i-1])
#         s_diff.append(s)
#         s_list.append(s + s_list[-1])
#     print('Point 2, s_list is', s_list)
#     print('Point 2, s_diff is', s_diff)

#     # Calculate path length distance of a given point from the end
#     s_list, s_diff = np.array(s_list), np.array(s_diff)
#     # Appears to be caused by the last element in s_diff being greater than offset_in + 0.1
#     dists = abs(s_list[-1] - s_list)   # Distance of each point from end of path
#     dists = np.array(dists[1:])   # dists is calculated using s_list - so you have one-too-many data points
#     pts_to_mod = np.where(dists < offset_in + 0.1)   # Points within a threshold distance of end of path
#     # Calculate the 'x'-axis of the power law plot - factor of -1 shifts it to a "distance from end"
#     s_in = -1*dists[pts_to_mod]
#     E_x = power_law(s_in, exp=exp_in, shift=-1*s_in[0])   # Factor by which to slow extrusion
#     E_fac = np.ones(len(dists))   # Factor by which to multiply E
#     E_fac[pts_to_mod] = E_fac[pts_to_mod]*E_x   # Project E_x onto an array the same length as the extrusion list
#     E_diff = E_fac*alpha_in*np.pi*(0.1*s_diff)*((d_in/20.)**2)   # E_fac*differential distance to extrude for each step
#     return E_fac, E_diff, s_list, s_diff


def fileread(filedir, filename, filetype):
    if filetype == 'inkscape':
        print('Inkscape csv file detected')
        # Read target wkt/svg file, split into elements
        with open(os.path.join(filedir, filename)) as f:
            data = f.read() 
            data = data.split('\n')
            data = data[1:len(data)-1]

    elif filetype == 'rhino':
        print('Rhino csv file detected')
        data = pd.read_csv(os.path.join(filedir, filename), header=None)
        data.columns = ['x', 'y', 'z', 'r', 'g', 'b']

    else:
        print("File type not recognised - please use 'rhino' or 'inkscape'")

    return data

def inkscape_preprocess(data):
    # Split wkt file into LINESTRING/POLYGON elements, return x and y coordinates 
    # Each element corresponds to a different element of the wkt
    # Note the coordinates strings are sometimes so long print won't show them all
    print('inkscape file being processed')
    pattern = []
    # Each element is a LINESTRING or POLYGON
    for idx, element in enumerate(data):
        element = wkt_splitter(element)
        element = coordinater_wkt(element, idx)
        if not len(element):
            print('Ignoring point')
        if len(element):
            pattern.append(element)

    pattern = [item for sublist in pattern for item in sublist]
    pattern = pd.DataFrame(pattern, columns=['x', 'y', 'z', 'line_id'])
    return pattern

# def interpolator(x_in, y_in, res):
#     # Replot pattern with point spacing = res mm
#     # Interpolator that plots the pattern at 100 µm spacing
#     x_out, y_out = [x_in[0]], [y_in[0]]
#     diff_list = []
#     # First - test if there are any points that need interpolation performing
#     for i in range(1, len(x_in)):
#         s = cartesian(x_in[i], y_in[i], x_in[i-1], y_in[i-1])
#         if s <= res:
#             x_out = np.append(x_out, x_in[i])
#             y_out = np.append(y_out, y_in[i])
#         if s > res:
#             n_points = math.floor(s/res)
#             # Don't include i-1 as that's already been added
#             new_points_x = np.linspace(x_in[i-1], x_in[i], num=n_points)[1:]
#             new_points_y = np.linspace(y_in[i-1], y_in[i], num=n_points)[1:]
#             x_out = np.append(x_out, new_points_x)
#             y_out = np.append(y_out, new_points_y)
#     return x_out, y_out


def node_plotter(df, terminal_points):
    # Plots lines assigning a colour to each line_id
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for n in np.unique(df['line_id'].values):
        ax.plot(df[df['line_id'] == n]['x'], df[df['line_id'] == n]['y'], df[df['line_id'] == n]['z'])

    # If you need a specific line plotting
    # n = 2
    # ax.plot(df[df['line_id'] == n]['x'], df[df['line_id'] == n]['y'], df[df['line_id'] == n]['z']

    for n in np.unique(terminal_points['cluster'].values):
        ax.scatter(terminal_points[terminal_points['cluster'] == n]['x'], terminal_points[terminal_points['cluster'] == n]['y'], terminal_points[terminal_points['cluster'] == n]['z'])

    ax.view_init(elev=30, azim=60)  # Elevation of 30 degrees, azimuth of 45 degrees
    plt.show()


def midlinejumpsplitter(shape):
    # This needs generalising to lines with more than 2 jumps
    # print('Splitting line with id:', shape['line_id'].unique()[0])
    id = shape['line_id'].unique()[0]
    # Split the line at the index - at the moment uses a completely arbitrary distance of 2mm
    split_index = shape[shape['distance_from_last'] > 2.].index[0]
    shape1 = shape.iloc[:split_index]
    shape1['line_id'] = 1
    shape1['distance_from_last'].iloc[0] = np.nan
    shape2 = shape.iloc[split_index:]
    shape2['line_id'] = 2
    shape2['distance_from_last'].iloc[0] = np.nan
    return shape1, shape2


def node_finder(df):
    # Use hierarchical clustering to note common start / end points
    # Calculate node positions based on cluster centroids
    # Append centroids to start / end of each line

    start_points = df.groupby('line_id').first() # First point of each line
    end_points = df.groupby('line_id').last() # Last
    terminal_points = pd.concat([start_points, end_points])   # Combine the two
    dist_mat = dist.pdist(terminal_points[['x', 'y', 'z']].values)   
    link_mat = hier.linkage(dist_mat)
    # fcluster assigns each of the particles in positions a cluster to which it belongs
    cluster_idx = hier.fcluster(link_mat, t=1, criterion='distance')   # t defines the max cophonetic distance in a cluster
    terminal_points['cluster'] = cluster_idx

    # Calculate the mean position of each cluster
    nodes = terminal_points.groupby('cluster').mean()
    for n in terminal_points.index.unique():
        clusters = terminal_points.loc[n]['cluster']
        for c in clusters:
            new_point = nodes.loc[c:c]
            line = df[df['line_id'] == n]
            line_start = line.head(1)
            line_end = line.tail(1)
            new_point[['r', 'g', 'b', 'line_id']] = line_start[['r', 'g', 'b', 'line_id']].values

            start_sep = dist.euclidean(new_point[['x', 'y', 'z']].values[0], line_start[['x', 'y', 'z']].values[0])
            end_sep = dist.euclidean(new_point[['x', 'y', 'z']].values[0], line_end[['x', 'y', 'z']].values[0])
            if start_sep < end_sep:
                line = pd.concat([new_point, line]) 
            elif start_sep > end_sep:
                line = pd.concat([line, new_point])
            
            df = df[df['line_id'] != n]
            df = pd.concat([df, line])
            df = df.reset_index(drop=True)
            df = distance_calculator(df)
            
            # Set distance from last to NaN for the first row of each line
            df.loc[df.groupby('line_id').head(1).index, 'distance_from_last'] = np.nan
    return df, terminal_points, nodes


# def plot_out(x_all_in, y_all_in, x_all_shift_in, y_all_shift_in, inlet_d, x_min_shift, x_max_shift):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(121)
#     for i in range(0, len(x_all_in)):
#         ax1.plot(x_all_in[i], y_all_in[i], lw = 4)
    
#     ax2 = fig.add_subplot(122)
#     for i in range(0, len(x_all_shift_in)):
#         ax2.plot(x_all_shift_in[i], y_all_shift_in[i], lw = 4)
#     ax1.set_aspect('equal')
#     ax2.set_aspect('equal')
#     ax1.set_title('Before Scaling')
#     ax2.set_title('After Scaling - with inlets')


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


# def power_law(x_in, exp, shift):
#     # x_in - x-coords to calculate the power law over
#     # exp, array-like - decay exponent (in range 0 to 1, 1 is a linear decrease)
#     # shift, float - x-intercept
#     # Returns a power law decay with points at 100 µm spacing (= 0.1)
    
#     y = 1 - np.power((x_in + shift)/shift, exp)
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111)
# #     ax.plot(x_in, y)
# #     ax.set_xlabel('Distance from end (mm)')
# #     ax.set_ylabel('Reduction factor')
#     return y

def remove_overlap(shape):
    # Checks for any large jumps at the end of a line and, if found, moves the last row to the top
    # This hopefully removes the jump and makes the line continuous
    # Note this seems very fragile and I probably need add a second check for large jumps at the end of the code
    if shape.iloc[-1]['distance_from_last'] > 2.:
        # Move the last row to the top
        last_row = shape.iloc[[-1]]  # Select the last row as a DataFrame
        remaining_rows = shape.iloc[:-1]  # Select all rows except the last
        shape = pd.concat([last_row, remaining_rows]).reset_index(drop=True)

    shape = distance_calculator(shape)
    return shape


def rhino_preprocess(df):
    # Calculate the distance between consecutive points
    df = distance_calculator(df)
    # If line ID column doesn't exist, assign line IDs based on RGB values
    if 'line_id' not in df.columns:
        df['line_id'] = pd.factorize(df[['r','g','b']].apply(tuple, axis=1))[0]
    # Remove large jumps at the end of lines
    df = df.groupby('line_id', group_keys=False).apply(remove_overlap)

    # Recalculate the distance between consecutive points
    df = distance_calculator(df)

    # Set distance from last to NaN for the first row of each line
    df.loc[df.groupby('line_id').head(1).index, 'distance_from_last'] = np.nan

    # Find and split lines that have big jumps in the middle, e.g., inlet and outlet lines
    df = shapesplitter(df)

    # Set distance from last to NaN for the first row of each line
    df.loc[df.groupby('line_id').head(1).index, 'distance_from_last'] = np.nan
    df, terminal_points, nodes = node_finder(df)
    node_plotter(df, terminal_points)

    line_order, node_order, line_order_grouped, node_order_grouped = eulerficator(df, terminal_points, nodes)
    return df, line_order, node_order, line_order_grouped, node_order_grouped


def shape_prep(filedir, filename, filetype, x_dim, y_dim, inlet_d, x_trans, y_trans, res):
    # Load pattern data
    data = fileread(filedir, filename, filetype)
    # Convert input shape into a DataFrame
    if filetype == 'inkscape':
        data = inkscape_preprocess(data)

        # Space out points - currently not used as Rhino does this quite well.
        data = spacer(coords, res)

        # Calculate centre of mass, min and max x and y values, and bounding box size, autoscales shape to fit in bounding box, applies an x-y translation to change centre of pattern
        data, x_com, y_com, x_min, x_max, y_min, y_max, bbox_x, bbox_y = param_calculator(x_all, y_all, x_dim-2*inlet_d, y_dim, x_trans, y_trans)
    
        # # Calculate size of bounding box after scaling
        # bbox_x_shift, bbox_y_shift, x_com_shift, y_com_shift, x_min_shift, x_max_shift, y_min_shift, y_max_shift = bbox_calculator(x_all_shift, y_all_shift)

        # # Add inlets and outlets - note the order of this is important, as E_calculator needs finely spaced points to work
        # if inlet_d > 0:
        #     inlet_x, inlet_y = [x_min_shift - inlet_d, x_min_shift], [0, 0]
        #     outlet_x, outlet_y = [x_max_shift + inlet_d, x_max_shift], [0, 0]
        #     x_all_shift.insert(0, inlet_x)
        #     x_all_shift.insert(1, outlet_x)
        #     y_all_shift.insert(0, inlet_y)
        #     y_all_shift.insert(1, outlet_y)

        # # Interpolate points 
        # for j in range(0, len(x_all_shift)):
        #     x_all_shift[j], y_all_shift[j] = interpolator(x_all_shift[j], y_all_shift[j], res=0.1)

        # # Plot output
        # plot_out(x_all, y_all, x_all_shift, y_all_shift, inlet_d, x_min_shift, x_max_shift)
        # print('Bounding box before scaling =', bbox_x, 'mm x', bbox_y, 'mm')
        # print('Bounding box after scaling =', bbox_x_shift + 2*inlet_d, 'mm x', bbox_y_shift, 'mm')

    elif filetype == 'rhino':
        data, line_order = rhino_preprocess(data)
        return data, line_order


def shapesplitter(df):
    # Identify line IDs that have large jumps in the middle
    line_ids = np.sort(df['line_id'].unique())   # Sorting makes life easier later
    line_ids_new = line_ids.copy()   # A list of line IDs that we're going to update
    for line_id in line_ids:
        shape = df[df['line_id'] == line_id]
        if shape['distance_from_last'].max() > 2.:
            shape1, shape2 = midlinejumpsplitter(shape)
            df  = df[df['line_id'] != line_id] 
            line_ids_new = line_ids_new[line_ids_new != line_id]
            line_ids_new = np.append(line_ids_new, [line_ids_new[-1]+1, line_ids_new[-1]+2])
            shape1['line_id'], shape2['line_id'] = line_ids_new[-2], line_ids_new[-1]
            shape = pd.concat([shape1, shape2])
            df = pd.concat([df, shape])  
    return df


def spacer(coords, res):
    # Now truncate it so that you only have coords separated by res µm
    x, y, z = coords[0], coords[1], coords[2]
    for i in range(0, len(x)-1)[::-1]:
        if cartesian2d(x[i], y[i], x[i+1], y[i+1]) < 0.1:
            del x[i]
            del y[i]
            del z[i]
    return [x, y, z]


def wkt_splitter(string_in):
    # Splits the string from a .wkt file into its individual components
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
