# Import and define useful modules
import math
import os
import pandas as pd
from shapely.geometry import GeometryCollection,LineString, Point, MultiPoint, MultiLineString

pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import numpy as np
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier
from scipy.interpolate import interp1d
import copy

np.seterr(invalid='ignore')  # Suppress divide by zero error


def build_settings(
    filedir,
    filename,
    fileout,
    d,
    x_offset,
    y_offset,
    bed_temperature,
    floor,
    z_min,
    roof,
    f_print,
    E_clean,
    # --- New printer / safety limits (optional; defaults keep backwards compatibility)
    XMIN=110.0,
    XMAX=220.0,
    YMIN=0.0,
    YMAX=210.0,
    margin=5.0,
    autoscale=True,
):
    settings = {
        'filedir': filedir,
        'filename': filename,
        'fileout': fileout,
        'd': d,
        'x_offset': x_offset,
        'y_offset': y_offset,
        'bed_temperature': bed_temperature,
        'floor': floor,
        'z_min': z_min,
        'roof': roof,
        'f_print': f_print,
        'E_clean': E_clean,
        # limits
        'XMIN': float(XMIN),
        'XMAX': float(XMAX),
        'YMIN': float(YMIN),
        'YMAX': float(YMAX),
        'margin': float(margin),
        'autoscale': bool(autoscale),
    }
    return settings


def cartesian2d(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def cartesian3d(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


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
    distances = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    df['distance_from_last'] = distances
    return df
def select_min_key(d):
    # Step 1: Keys with odd-length lists
    odd_keys = [k for k in d if len(d[k]) % 2 == 1]

    if odd_keys:
        # Return the key with the shortest odd-length list
        return min(odd_keys, key=lambda k: len(d[k]))
    else:
        # Step 2: If no odd-length lists, get keys with even-length lists
        even_keys = [k for k in d if len(d[k]) % 2 == 0]
        if even_keys:
            return min(even_keys, key=lambda k: len(d[k]))
        else:
            return None  # In case all lists are empty or something went wrong

def vector_calculator(line_string, intersection):
    line_length =len( line_string.coords)
    # print("linelengtht",line_length)
    tenth_of_line = int(line_length/10)
    # print("tenth of line",tenth_of_line)
    target_coordinates = intersection
    coords_list = list(line_string.coords)
    closest_index = min(
        range(len(coords_list)),
        key=lambda i: Point(coords_list[i]).distance(intersection)
    )
    start_coord_index=closest_index
    if start_coord_index + tenth_of_line < line_length:
        end_coord_index = start_coord_index + tenth_of_line
    else:
        end_coord_index = max(0, start_coord_index - tenth_of_line)
    x1, y1 = line_string.coords[start_coord_index]


    x2, y2 = line_string.coords[end_coord_index]
    v = np.array([x2-x1,y2-y1])
    # print("x1", x1, "x2",x2,"y1",y1,"y2",y2)
    # print("vector coordinates", v)
    return v
def angle_calculator(line_string_1, line_string_2, intersection):
    v1 = vector_calculator(line_string_1, intersection)
    v2 = vector_calculator(line_string_2, intersection)

    v1_norm = v1/np.linalg.norm(v1)
    v2_norm = v2/ np.linalg.norm(v2)
    # print("v1 normalised", v1_norm)
    # print("v2 normalised", v2_norm)
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    # print(dot_product, "dot product")
    cross_product = np.cross(v1_norm, v2_norm)
    radian_angle = np.arctan2(cross_product,dot_product)
    degree_angle = np.degrees(radian_angle)
    return degree_angle
def interpolated_z(intersection_point, df):

    df["xy_distance"] = np.sqrt((df['x'] - Point(intersection_point).x)**2 + (df['y'] - Point(intersection_point).y)**2)

    closest_row = df.loc[df["xy_distance"].idxmin()]
    # print(closest_row)

    target_index = df["xy_distance"].argmin() # the row you're interested in
    window = 5  # number of rows before and after

    # Slice around the target
    start = max(0, target_index - window)
    end = min(len(df), target_index + window + 1)

    # Show the rows
    # print("this is the data frame around the minimum distance",df.iloc[start:end])
    z_value = closest_row["z"]
    return z_value


def z_tenth_from_intersection_calculator(intersection, df, line_string):
    line_length = len(line_string.coords)

    tenth_of_line = int(line_length / 10)

    target_coordinates = intersection
    coords_list = list(line_string.coords)
    closest_index = min(
        range(len(coords_list)),
        key=lambda i: Point(coords_list[i]).distance(intersection)
    )
    start_coord_index = closest_index
    if start_coord_index + tenth_of_line < line_length:
        end_coord_index = start_coord_index + tenth_of_line
    else:
        end_coord_index = max(0, start_coord_index - tenth_of_line)
    z_point = Point(line_string.coords[end_coord_index])
    # print(z_point, "zpoint")
    z_value = interpolated_z(z_point, df)
    return z_value

import numpy as np
from shapely.geometry import GeometryCollection, LineString, Point, MultiPoint, MultiLineString

def validator(unprocessed_lines, df):
    """
    MARMOT-style validator.

    For each candidate line (test_id), compare it to all other unprocessed lines:
      1) At any XY intersection, if the test line is higher than the other line
         by more than tol_z, the test line is invalid.
      2) If they meet at (almost) the same height and the in-plane angle between
         them is shallow (< angle_thresh), look 1/10 of the line length away
         from the intersection on both lines; if the test line is higher there
         by more than tol_z_node, it is invalid.
    """

    tol_z = 0.03        # main height tolerance (mm)
    tol_z_node = 0.03   # node / 1/10th height tolerance (mm)
    angle_thresh = 10.0 # degrees
    valid_lines = []

    # unprocessed_lines is assumed to be a list of line_ids (ints)
    for test_id in unprocessed_lines:
        # Extract this line's data
        test_df = df[df["line_id"] == test_id][["x", "y", "z"]].copy()
        if len(test_df) < 2:
            # skip degenerate lines
            continue

        test_ls = LineString(test_df[["x", "y"]].to_numpy())
        is_valid = True

        for comp_id in unprocessed_lines:
            if comp_id == test_id:
                continue
            if not is_valid:
                break

            comp_df = df[df["line_id"] == comp_id][["x", "y", "z"]].copy()
            if len(comp_df) < 2:
                continue

            comp_ls = LineString(comp_df[["x", "y"]].to_numpy())

            # Only care if they intersect in XY
            if not test_ls.intersects(comp_ls):
                continue

            inter = test_ls.intersection(comp_ls)

            # helper to apply MARMOT rules at a single (x,y) point
            def handle_point(pt):
                nonlocal is_valid
                # pt is a shapely Point
                coord = (pt.x, pt.y)

                z_test = interpolated_z(coord, test_df)
                z_comp = interpolated_z(coord, comp_df)
                if z_test is None or z_comp is None:
                    return

                dz = z_test - z_comp

                # Case A: different height → test line must not be above future line
                if abs(dz) > tol_z:
                    if dz > tol_z:
                        is_valid = False
                    return

                # Case B: effectively same height → treat as node-type intersection
                # apply angle rule + 1/10-length z check
                angle = angle_calculator(test_ls, comp_ls, pt)

                # only shallow angles are problematic
                if abs(angle) < angle_thresh:
                    try:
                        z_test_10 = z_tenth_from_intersection_calculator(
                            pt, test_df, test_ls
                        )
                        z_comp_10 = z_tenth_from_intersection_calculator(
                            pt, comp_df, comp_ls
                        )
                    except Exception:
                        # if we can't sample 1/10th points, just skip this point
                        return

                    if z_test_10 > z_comp_10 + tol_z_node:
                        is_valid = False

            # handle different intersection geometry types

            # Single point
            if isinstance(inter, Point):
                handle_point(inter)

            # Multiple discrete points
            elif isinstance(inter, MultiPoint):
                for pt in inter.geoms:
                    handle_point(pt)
                    if not is_valid:
                        break

            # Single overlapping segment
            elif isinstance(inter, LineString):
                coords = list(inter.coords)
                # sample a few points along the overlap
                step = max(1, len(coords) // 5)
                for x, y in coords[::step]:
                    handle_point(Point(x, y))
                    if not is_valid:
                        break

            # Multiple overlapping segments
            elif isinstance(inter, MultiLineString):
                for geom in inter.geoms:
                    coords = list(geom.coords)
                    step = max(1, len(coords) // 5)
                    for x, y in coords[::step]:
                        handle_point(Point(x, y))
                        if not is_valid:
                            break
                    if not is_valid:
                        break

            # Mixed geometry
            elif isinstance(inter, GeometryCollection):
                for geom in inter.geoms:
                    if isinstance(geom, Point):
                        handle_point(geom)
                    elif isinstance(geom, LineString):
                        coords = list(geom.coords)
                        step = max(1, len(coords) // 5)
                        for x, y in coords[::step]:
                            handle_point(Point(x, y))
                            if not is_valid:
                                break
                    if not is_valid:
                        break

        if is_valid:
            valid_lines.append(test_id)

    print("valid lines", valid_lines)
    return valid_lines


def node_connectivity_finder(dictionary):
    """
    Build a node-connectivity dictionary from a mapping:
        cluster_id -> list of line_ids.
    For each node (cluster), list which other nodes it connects to
    (with multiplicity based on the number of shared lines).
    """
    new_dictionary = {}

    for key1, values1 in dictionary.items():
        shared_keys = []

        for key2, values2 in dictionary.items():
            if key1 == key2:
                continue

            # Count how many values from values1 are also in values2
            shared_count = sum(1 for v in values1 if v in values2)

            # Repeat key2 shared_count times
            shared_keys.extend([key2] * shared_count)

        new_dictionary[key1] = shared_keys

    return new_dictionary


def line_from_nodes(start_node, end_node, valid_line_cluster_dict):
    """
    Given two nodes (clusters), return the line_id that connects them.
    Assumes exactly one line is shared.
    """
    end_node_list = valid_line_cluster_dict[end_node]
    start_node_list = valid_line_cluster_dict[start_node]
    shared_lines = [item for item in start_node_list if item in end_node_list]
    return shared_lines[0]


def remove_line(valid_line_cluster_dict, start_node, next_node, line):
    valid_line_cluster_dict[start_node].remove(line)
    valid_line_cluster_dict[next_node].remove(line)


def remove_node(valid_node_link_dict, u, v):
    valid_node_link_dict[u].remove(v)
    valid_node_link_dict[v].remove(u)


def depth_first_search(next_node, valid_node_link_dict, visited):
    visited[next_node] = True

    for neighbor in valid_node_link_dict[next_node]:
        if not visited[neighbor]:
            depth_first_search(neighbor, valid_node_link_dict, visited)


def bridge_check(next_node, start_node, valid_node_link_dict, node_number):
    """
    Check whether removing the edge (start_node, next_node) would disconnect
    the graph (i.e. whether it's a bridge).
    """
    if len(valid_node_link_dict[start_node]) == 1:
        # If there's only one connection, we have to take it
        return True, 0

    node_link_copy = copy.deepcopy(valid_node_link_dict)
    visited = {key: False for key in node_link_copy}

    depth_first_search(next_node, node_link_copy, visited)
    count1 = sum(1 for value in visited.values() if value)

    # Remove edge and see how many nodes remain reachable
    remove_node(node_link_copy, start_node, next_node)

    visited = {key: False for key in node_link_copy}
    depth_first_search(start_node, node_link_copy, visited)
    count2 = sum(1 for value in visited.values() if value)

    # Restore edge in the copy (not strictly needed but tidy)
    node_link_copy[start_node].append(next_node)
    node_link_copy[next_node].append(start_node)

    return (count1 == count2), count2


def recursive_eulerian(
    node_path,
    edges,
    start_node,
    valid_node_link_dict,
    node_number,
    valid_line_cluster_dict,
):
    """
    Recursive part of Fleury's algorithm to build an Eulerian path.
    """
    if not valid_node_link_dict.get(start_node):
        return

    any_bridge_passed = False
    count2_map = {}

    for node in valid_node_link_dict[start_node]:
        next_node = node
        passed_bridge, count2 = bridge_check(
            next_node, start_node, valid_node_link_dict, node_number
        )
        if count2 is not None:
            count2_map[next_node] = count2
        if passed_bridge:
            any_bridge_passed = True
            if start_node == node_path[-1]:
                line = line_from_nodes(start_node, next_node, valid_line_cluster_dict)
                edges.append(line)
                node_path.append(next_node)
                remove_node(valid_node_link_dict, start_node, next_node)
                remove_line(valid_line_cluster_dict, start_node, next_node, line)
                recursive_eulerian(
                    node_path,
                    edges,
                    next_node,
                    valid_node_link_dict,
                    node_number,
                    valid_line_cluster_dict,
                )
                break

    if not any_bridge_passed:
        if count2_map:
            next_node = min(count2_map, key=count2_map.get)
        else:
            next_node = valid_node_link_dict[start_node][0]

        if start_node == node_path[-1]:
            line = line_from_nodes(start_node, next_node, valid_line_cluster_dict)
            edges.append(line)
            node_path.append(next_node)
            remove_node(valid_node_link_dict, start_node, next_node)
            remove_line(valid_line_cluster_dict, start_node, next_node, line)
            recursive_eulerian(
                node_path,
                edges,
                next_node,
                valid_node_link_dict,
                node_number,
                valid_line_cluster_dict,
            )


def fleurys_algorithm(
    clusters, cluster_dictionary, connectivity_dictionary, valid_lines,
    edges_grouped, node_path_grouped
):
    """
    Apply Fleury's algorithm cluster-by-cluster to build Eulerian paths
    that respect the set of currently valid lines.
    """
    valid_line_cluster_dict = {}

    for cluster, lines in cluster_dictionary.items():
        valid_line_cluster_dict[cluster] = [item for item in lines if item in valid_lines]

    valid_node_link_dict = node_connectivity_finder(valid_line_cluster_dict)

    while valid_line_cluster_dict:
        node_number = len(valid_node_link_dict)

        filtered = {
            k: v for k, v in valid_node_link_dict.items()
            if v not in ("", None, [])
        }

        if filtered:
            start_node = select_min_key(filtered)
        else:
            # no usable nodes remain
            break

        node_path = []
        edges = []
        node_path.append(start_node)
        recursive_eulerian(
            node_path, edges, start_node,
            valid_node_link_dict, node_number, valid_line_cluster_dict
        )
        edges_grouped.append(edges)
        node_path_grouped.append(node_path)

    print(edges_grouped)
    print(node_path_grouped)


def eulerficator(df, terminal_points, nodes):
    """
    Top-level function that:
    - builds cluster+connectivity dictionaries,
    - repeatedly uses the validator + Fleury's algorithm
      to group lines into printable paths,
    - returns the final line / node ordering.

    This is the version that shape_prep(df) expects.
    """
    terminal_points_nogroups = terminal_points.reset_index()
    clusters = terminal_points_nogroups["cluster"].unique()
    lines = terminal_points_nogroups["line_id"].unique()

    # cluster_dict: cluster -> list of line_ids
    # connectivity_dict: cluster -> number of incident terminals
    cluster_dict = {}
    connectivity_dict = {}

    for c in clusters:
        connecting_lines = terminal_points_nogroups[
            terminal_points_nogroups["cluster"] == c
        ]
        cluster_dict[c] = list(connecting_lines["line_id"].values)
        connectivity_dict[c] = len(connecting_lines)

    edges_grouped = []
    node_path_grouped = []

    # keep going until we've accounted for all lines
    while sum(len(sublist) for sublist in edges_grouped) < len(lines):

        # Sort lines by min z -> bottom up
        min_z = df.groupby("line_id")["z"].min()
        heightsorted_line_ids = min_z.sort_values().index.tolist()
        edges_ungrouped = [item for sublist in edges_grouped for item in sublist]
        unprinted_lines = [
            n for n in heightsorted_line_ids if n not in edges_ungrouped
        ]

        valid_lines = validator(unprinted_lines, df)
        if not valid_lines:
            print("No valid lines found. Breaking to avoid infinite loop.")
            break

        fleurys_algorithm(
            clusters, cluster_dict, connectivity_dict,
            valid_lines, edges_grouped, node_path_grouped
        )

        # Remove printed lines from unprinted list
        while len(valid_lines) > 0:
            next_line = valid_lines[0]
            valid_lines = valid_lines[1:]
            if next_line in unprinted_lines:
                unprinted_lines.remove(next_line)

    node_order = [item for sublist in node_path_grouped for item in sublist]
    print(node_order, "node order")
    line_order = [item for sublist in edges_grouped for item in sublist]
    print(line_order, "line order")
    line_order_grouped = edges_grouped
    print("line order grouped", line_order_grouped)
    node_order_grouped = node_path_grouped
    print("node order grouped", node_order_grouped)

    return line_order, node_order, line_order_grouped, node_order_grouped



def e_calculator(df):
    alpha = 1
    diameter = 1
    df['E'] = np.pi * alpha * df['distance_from_last'] * (diameter / 2) ** 2  # Amount to extrude
    return df


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


def line_order_corrector(df, line_order, line_order_grouped, nodes, node_order_grouped):
    # Make lines run in node order.
    df = df.set_index('line_id').loc[line_order].reset_index()  # Reorder the dataframe based on the line order

    for idx_path, path in enumerate(line_order_grouped):
        for idx_line, line in enumerate(path):
            start_node = node_order_grouped[idx_path][idx_line]
            line_start = df[df['line_id'] == line].iloc[0][['x', 'y', 'z']].values
            node_loc = nodes.loc[start_node][['x', 'y', 'z']].values

            if not np.array_equal(line_start, node_loc):
                # print('Reversing line ', line)
                new_line = df[df['line_id'] == line].iloc[::-1]
                df = df[df['line_id'] != line]
                df = pd.concat([df, new_line])

    df = df.set_index('line_id').loc[line_order].reset_index()  # Reorder the dataframe based on the line order
    df = distance_calculator(df)
    return df


def midlinejumpsplitter(shape):
    # This needs generalising to lines with more than 2 jumps
    # print('Splitting line with id:', shape['line_id'].unique()[0])
    id = shape['line_id'].unique()[0]
    # Split the line at the index - at the moment uses a completely arbitrary distance of 2mm
    split_pos = shape.index.get_loc(shape[shape['distance_from_last'] > 2.].index[0])
    shape1 = shape.iloc[:split_pos]
    shape2 = shape.iloc[split_pos:]
    if not shape1.empty:
        shape1.loc[shape1.index[0], 'distance_from_last'] = np.nan
    if not shape2.empty:
        shape2.loc[shape2.index[0], 'distance_from_last'] = np.nan
    return shape1, shape2


def node_finder(df):
    # Use hierarchical clustering to note common start / end points
    # Calculate node positions based on cluster centroids
    # Append centroids to start / end of each line

    start_points = df.groupby('line_id').first()  # First point of each line
    end_points = df.groupby('line_id').last()  # Last
    terminal_points = pd.concat([start_points, end_points])  # Combine the two
    dist_mat = dist.pdist(terminal_points[['x', 'y', 'z']].values)
    link_mat = hier.linkage(dist_mat)
    # fcluster assigns each of the particles in positions a cluster to which it belongs
    cluster_idx = hier.fcluster(link_mat, t=1,
                                criterion='distance')  # t defines the max cophonetic distance in a cluster
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


def node_plotter(df, terminal_points, line_order, node_order, node_order_grouped):
    

    # Parameters
    n_interp = 50  # Interpolated points per segment

    # interpolates line segment
    def interpolate_line(line_data, n_points=50):
        t = np.linspace(0, 1, len(line_data))
        fx = interp1d(t, line_data['x'], kind='linear')
        fy = interp1d(t, line_data['y'], kind='linear')
        fz = interp1d(t, line_data['z'], kind='linear')
        t_new = np.linspace(0, 1, n_points)
        return np.vstack((fx(t_new), fy(t_new), fz(t_new))).T

    # terminal edges to line IDs
    terminal_edges = []

    for sublist in node_order_grouped:
        terminal_edges.extend(zip(sublist, sublist[1:]))
    # print(terminal_edges)
    # print(line_order)
    # if len(terminal_edges) != len(line_order):
    #     raise ValueError("Number of terminal edges must match number of lines")

    edge_to_line = list(zip(terminal_edges, line_order))
    # print("edge to line", edge_to_line)

    # line segments in correct order and direction
    # print("Edge to line mapping:")
    
    line_segments = []
    # print("terminal edges", terminal_edges)


    for (start_cluster, end_cluster), line_id in edge_to_line:
        line_data = df[df['line_id'] == line_id].reset_index(drop=True)
        if len(line_data) < 2:
            continue

        # Get terminal coordinates
        start_coords = terminal_points[terminal_points['cluster'] == start_cluster][['x', 'y', 'z']].iloc[0].values
        end_coords = terminal_points[terminal_points['cluster'] == end_cluster][['x', 'y', 'z']].iloc[0].values

        # Get current line direction
        line_start = line_data[['x', 'y', 'z']].iloc[0].values
        line_end = line_data[['x', 'y', 'z']].iloc[-1].values

        # Flip line if it starts at the wrong end
        dist_start = np.linalg.norm(line_start - start_coords)
        dist_end = np.linalg.norm(line_end - start_coords)
        if dist_end < dist_start:
            line_data = line_data.iloc[::-1].reset_index(drop=True)

        # Interpolate and store
        pts = interpolate_line(line_data, n_points=n_interp)
        line_segments.append(pts)

    # frame-to-segment map
    segment_lengths = [len(seg) for seg in line_segments]
    frame_to_segment = []
    for i, length in enumerate(segment_lengths):
        frame_to_segment += [(i, j) for j in range(length)]


    # Set up plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Calculate data ranges for aspect ratio
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min()
    z_range = df['z'].max() - df['z'].min()

    # Avoid zero-range issues for single points or perfectly flat lines
    if x_range == 0: x_range = 1
    if y_range == 0: y_range = 1
    if z_range == 0: z_range = 1

    # Set axis limits to contain all data
    ax.set_xlim(df['x'].min()-1, df['x'].max()+1)
    ax.set_ylim(df['y'].min()-1, df['y'].max()+1)
    ax.set_zlim(df['z'].min()-1, df['z'].max()+1)

    
    ax.set_box_aspect((x_range, y_range, z_range))

    ax.view_init(elev=30, azim=60)

    # Assign distinct colors based on jumps
    distinct_colors = plt.cm.get_cmap('tab10').colors  
    n_colors = len(distinct_colors)
    colors = []
    threshold = 1e-3  # jump threshold
    color_index = 0
    colors.append(distinct_colors[color_index % n_colors])

    for i in range(1, len(line_segments)):
        prev_end = line_segments[i - 1][-1]
        curr_start = line_segments[i][0]
        if np.linalg.norm(prev_end - curr_start) > threshold:
            color_index += 1  # New color on jump
        colors.append(distinct_colors[color_index % n_colors])

    # Line plots
    plot_lines = [
        ax.plot([], [], [], color=colors[i], linewidth=2)[0]
        for i in range(len(line_segments))
    ]

    # Line ID labels (initially hidden)
    line_labels = []
    for i, seg in enumerate(line_segments):
        mid_idx = len(seg) // 2
        mid_pt = seg[mid_idx]
        line_id = line_order[i]
        label = ax.text(mid_pt[0], mid_pt[1], mid_pt[2], str(line_id),
                        color=colors[i], fontsize=10, visible=False)
        line_labels.append(label)

    # Node labels (initially hidden)
    # node_labels = {}
    # for cluster in node_order:
    #     coords = terminal_points[terminal_points['cluster'] == cluster][['x', 'y', 'z']].iloc[0].values
    #     label = ax.text(coords[0], coords[1], coords[2], str(cluster),
    #                     color='black', fontsize=9, visible=False)
    #     node_labels[cluster] = label

    # Print head (moving marker)
    head, = ax.plot([], [], [], marker='o', color='red', markersize=5)

    # Animation update function
    def update(frame):
        seg_idx, pt_idx = frame_to_segment[frame]

        # Draw segments up to current
        for i in range(seg_idx + 1):
            seg = line_segments[i]
            end_idx = pt_idx + 1 if i == seg_idx else len(seg)
            x = seg[:end_idx, 0]
            y = seg[:end_idx, 1]
            z = seg[:end_idx, 2]
            plot_lines[i].set_data(x, y)
            plot_lines[i].set_3d_properties(z)

            # Show line label
            line_labels[i].set_visible(True)

        # Move print head
        curr_point = line_segments[seg_idx][pt_idx]
        head.set_data([curr_point[0]], [curr_point[1]])
        head.set_3d_properties([curr_point[2]])

        # Show node labels as they appear
        # if pt_idx == 0:
        #     start_cluster = node_order[seg_idx]
        #     node_labels[start_cluster].set_visible(True)
        # if pt_idx == len(line_segments[seg_idx]) - 1:
        #     end_cluster = node_order[seg_idx + 1] if seg_idx + 1 < len(node_order) else None
        #     if end_cluster is not None:
        #         node_labels[end_cluster].set_visible(True)

        return plot_lines + [head] + line_labels# + list(node_labels.values())

    # Animation
    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_to_segment),
        interval=20,
        blit=False,
        repeat=False
    )
    # print("Returning animation object:", ani)
    

    
    plt.show()
    return ani

def node_plotter_2(df, terminal_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot and label lines
    for n in np.unique(df['line_id'].values):
        line_data = df[df['line_id'] == n]
        ax.plot(line_data['x'], line_data['y'], line_data['z'])

        # Add label at the midpoint of the line
        mid_index = len(line_data) // 2
        x_mid = line_data['x'].values[mid_index]
        y_mid = line_data['y'].values[mid_index]
        z_mid = line_data['z'].values[mid_index]

        ax.text(x_mid, y_mid, z_mid, str(n), fontsize=9, color='black')

    # Plot and label terminal points
    for n in np.unique(terminal_points['cluster'].values.astype(int)):
        cluster_data = terminal_points[terminal_points['cluster'] == n]
        first_row = cluster_data.iloc[0]  # or use .mean() if you want the centroid

        ax.text(first_row['x'], first_row['y'], first_row['z'], str(int(first_row['cluster'])),
                fontsize=8, color='black', zorder=10,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, boxstyle='square,pad=0.2'))
    ax.view_init(elev=30, azim=60)
    plt.show()

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


def preprocess(df, settings):
    # Calculate the distance between consecutive points
    df = distance_calculator(df)
    # If line ID column doesn't exist, assign line IDs based on RGB values
    if 'line_id' not in df.columns:
        df['line_id'] = pd.factorize(df[['r', 'g', 'b']].apply(tuple, axis=1))[0]
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

    # --- Position + (optionally) scale pattern into reachable machine area ---
    XMIN = float(settings.get('XMIN', 0.0))
    XMAX = float(settings.get('XMAX', 220.0))
    YMIN = float(settings.get('YMIN', 0.0))
    YMAX = float(settings.get('YMAX', 220.0))
    margin = float(settings.get('margin', 0.0))

    # If user supplies x_offset/y_offset, treat them as the desired centre of the pattern
    x_center = float(settings.get('x_offset', 0.5 * (XMIN + XMAX)))
    y_center = float(settings.get('y_offset', 0.5 * (YMIN + YMAX)))

    # Compute bounding box of incoming pattern
    bbox_x = float(df['x'].max() - df['x'].min())
    bbox_y = float(df['y'].max() - df['y'].min())

    # Optional autoscale to fit within machine limits (with margin)
    scale = 1.0
    if bool(settings.get('autoscale', False)) and bbox_x > 0 and bbox_y > 0:
        avail_x = max(1e-9, (XMAX - XMIN) - 2.0 * margin)
        avail_y = max(1e-9, (YMAX - YMIN) - 2.0 * margin)
        scale = min(1.0, avail_x / bbox_x, avail_y / bbox_y)

    df['x'] = (df['x'] - df['x'].mean()) * scale + x_center
    df['y'] = (df['y'] - df['y'].mean()) * scale + y_center

    # Validate final coords are within limits (fail fast with a helpful message)
    x_min_out, x_max_out = float(df['x'].min()), float(df['x'].max())
    y_min_out, y_max_out = float(df['y'].min()), float(df['y'].max())

    if x_min_out < XMIN - 1e-6 or x_max_out > XMAX + 1e-6 or y_min_out < YMIN - 1e-6 or y_max_out > YMAX + 1e-6:
        raise ValueError(
            f"Pattern is outside machine limits after offset/scale. "
            f"X[{x_min_out:.3f},{x_max_out:.3f}] vs limits [{XMIN:.3f},{XMAX:.3f}], "
            f"Y[{y_min_out:.3f},{y_max_out:.3f}] vs limits [{YMIN:.3f},{YMAX:.3f}]. "
            f"Try reducing pattern size or increase autoscale/margin."
        )


    return df


def shape_prep(settings):
    # Load pattern data and convert to DataFrame
    df = pd.read_csv(os.path.join(settings['filedir'], settings['filename']), header=None)
    df.columns = ['x', 'y', 'z', 'r', 'g', 'b']

    df = preprocess(df, settings)

    # Find nodes via hierarchical clustering
    df, terminal_points, nodes = node_finder(df)

    # Build line_connectivity: node(cluster_id) -> list of line_ids touching that node
    line_connectivity = {}
    # terminal_points index is line_id, 'cluster' gives start/end node(s)
    for line_id in terminal_points.index.unique():
        clusters = terminal_points.loc[line_id]['cluster']
        # clusters can be a scalar or a Series, so normalise:
        for c in np.atleast_1d(clusters):
            c = int(c)
            line_connectivity.setdefault(c, []).append(int(line_id))

    # Node connectivity (node -> neighbouring nodes)
    node_connectivity = node_connectivity_finder(line_connectivity)

    # Unprinted lines is just the list of line_ids
    unprinted_lines = sorted(df['line_id'].unique().tolist())

    # Euler path / grouped order
    # Use the real 3-arg Euler function that matches the report
    line_order, node_order, line_order_grouped, node_order_grouped = eulerficator( df, terminal_points, nodes)


    # Quick visualisation
    node_plotter_2(df, terminal_points)

    # Make each line run in the node order and recompute distances
    df = line_order_corrector(df, line_order, line_order_grouped, nodes, node_order_grouped)

    return df, line_order_grouped


def shapesplitter(df):
    # Identify line IDs that have large jumps in the middle
    line_ids = np.sort(df['line_id'].unique())  # Sorting makes life easier later
    line_ids_new = line_ids.copy()  # A list of line IDs that we're going to update
    for line_id in line_ids:
        shape = df[df['line_id'] == line_id]
        if shape['distance_from_last'].max() > 2.:
            shape1, shape2 = midlinejumpsplitter(shape)
            df = df[df['line_id'] != line_id]
            line_ids_new = line_ids_new[line_ids_new != line_id]
            if line_ids_new.size == 0:
                # First IDs to assign
                print('===!!!Likely incorrectly created RHINO cad file - no line IDs found!!!===')
            else:
                line_ids_new = np.append(line_ids_new, [line_ids_new[-1] + 1, line_ids_new[-1] + 2])
            shape1['line_id'], shape2['line_id'] = line_ids_new[-2], line_ids_new[-1]
            shape = pd.concat([shape1, shape2])
            df = pd.concat([df, shape])
    return df

# def spacer(coords, res):
#     # Now truncate it so that you only have coords separated by res µm
#     x, y, z = coords[0], coords[1], coords[2]
#     for i in range(0, len(x)-1)[::-1]:
#         if cartesian2d(x[i], y[i], x[i+1], y[i+1]) < 0.1:
#             del x[i]
#             del y[i]
#             del z[i]
#     return [x, y, z]


# def wkt_splitter(string_in):
#     # Splits the string from a .wkt file into its individual components
#     if 'LINESTRING' in string_in:
#         start = string_in.find('(')
#         end = string_in.find(')')
#         string_in = string_in[start+1:end]
#         string_in = string_in.split(',')
    
#     if 'POLYGON' in string_in:
#         start = string_in.find('((')
#         end = string_in.find('))')
#         string_in = string_in[start+2:end]
#         string_in = string_in.split(',')
        
#     if 'POINT' in string_in:
#         print('POINT detected in wkt file - these are currently ignored')
#         string_in = ''
   
#     return string_in


### Legacy code
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


# def E_calculator_old(x_in, y_in, res_in, d_in, exp_in, offset_in, alpha_in):
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


# Leftover code from shapeprep to handle inkscape files
    # if filetype == 'inkscape':
    #     df = inkscape_preprocess(data)

    #     # Space out points - currently not used as Rhino does this quite well.
    #     df = spacer(coords, res)

    #     # Calculate centre of mass, min and max x and y values, and bounding box size, autoscales shape to fit in bounding box, applies an x-y translation to change centre of pattern
    #     df, x_com, y_com, x_min, x_max, y_min, y_max, bbox_x, bbox_y = param_calculator(x_all, y_all, x_dim-2*inlet_d, y_dim, x_trans, y_trans)
    
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


# def param_calculator(x_in, y_in, x_dim_in, y_dim_in, x_trans_in, y_trans_in):
#     # Autoscales to fit a pre-determined bounding box (defined by x_dim and y_dim)
#     bbox_x_out, bbox_y_out, x_com_out, y_com_out, x_min_out, x_max_out, y_min_out, y_max_out = bbox_calculator(x_in, y_in)
    
#     if bbox_x_out > 0:
#         x_scale = x_dim_in / bbox_x_out
#     if bbox_x_out == 0:
#         print('Zero pattern width detected - scaling by 1')
#         x_scale = 1.
    
#     if bbox_y_out > 0:
#         y_scale = y_dim_in / bbox_y_out
#     if bbox_y_out == 0:
#         print('Zero pattern height detected - scaling by 1')
#         y_scale = 1.
    
#     # Rescales and shifts all input coordinates
#     x_all_shift_scale, y_all_shift_scale = [], []
#     for i in range(0, len(x_in)):
#         x_all_shift_scale.append((x_in[i] - x_com_out)*x_scale)
#         y_all_shift_scale.append((y_in[i] - y_com_out)*y_scale)
        
#     # Finally, apply an x-y translation to manually shift the centre of the print, if desired
#     x_all_shift_out, y_all_shift_out = [], []
#     for i in range(0, len(x_in)):
#         x_all_shift_out.append((x_all_shift_scale[i] + x_trans_in))
#         y_all_shift_out.append((y_all_shift_scale[i] + y_trans_in))
        
#     return x_all_shift_out, y_all_shift_out, x_com_out, y_com_out, x_min_out, x_max_out, y_min_out, y_max_out, bbox_x_out, bbox_y_out


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
    


# def fileread(filedir, filename, filetype=rhino):
#     # Legacy code - will be removed
#     if filetype == 'inkscape':
#         print('Inkscape csv file detected')
#         # Read target wkt/svg file, split into elements
#         with open(os.path.join(filedir, filename)) as f:
#             data = f.read() 
#             data = data.split('\n')
#             data = data[1:len(data)-1]

#     elif filetype == 'rhino':
#         print('Rhino csv file detected')
#         data = pd.read_csv(os.path.join(filedir, filename), header=None)
#         data.columns = ['x', 'y', 'z', 'r', 'g', 'b']

#     else:
#         print("File type not recognised - please use 'rhino' or 'inkscape'")

#     return data
