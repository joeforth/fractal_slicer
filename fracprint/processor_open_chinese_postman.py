# Import and define useful modules
import math
import os
import pandas as pd
from shapely.geometry import GeometryCollection,LineString, Point, MultiPoint, MultiLineString
from shapely.ops import nearest_points

pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import numpy as np


def _flatten(grouped):
    return [item for sublist in grouped for item in sublist]
from scipy.spatial import distance as dist
import scipy.cluster.hierarchy as hier
from scipy.interpolate import interp1d
import copy

np.seterr(invalid='ignore')  # Suppress divide by zero error


def build_settings(filedir, filename, fileout, d, x_offset, y_offset, bed_temperature, floor, z_min, roof, f_print,
                   E_clean, node_nudge_mm=0.0, node_nudge_npts=0, node_slow_dist_mm=1.5, f_node=60, node_tol=0.25, slow_revisit_start=True, use_chinese_postman=True):
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
        'node_nudge_mm': node_nudge_mm,
        'node_nudge_npts': node_nudge_npts,
        'node_slow_dist_mm': node_slow_dist_mm,
        'f_node': f_node,
        'node_tol': node_tol,
        'slow_revisit_start': slow_revisit_start,
        'use_chinese_postman': use_chinese_postman,
        'E_clean': E_clean
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
    """Compute distance_from_last safely.

    If df contains a 'line_id' column, distances are computed *within each line_id*
    (so we don't accidentally measure distance between the last point of one line
    and the first point of the next).
    """
    def _within(g):
        dx = g['x'].diff()
        dy = g['y'].diff()
        dz = g['z'].diff()
        g['distance_from_last'] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        # First point of each line has no previous point
        g.iloc[0, g.columns.get_loc('distance_from_last')] = np.nan
        return g

    if 'line_id' in df.columns:
        return df.groupby('line_id', group_keys=False).apply(_within)

    dx = df['x'].diff()
    dy = df['y'].diff()
    dz = df['z'].diff()
    df['distance_from_last'] = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    df.iloc[0, df.columns.get_loc('distance_from_last')] = np.nan
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




def _add_edge_to_adjacency(adjacency, u, v, lid, w):
    adjacency.setdefault(u, [])
    adjacency.setdefault(v, [])
    if not any((n == v and int(elid) == int(lid)) for n, elid, _ in adjacency[u]):
        adjacency[u].append((v, int(lid), float(w)))
    if not any((n == u and int(elid) == int(lid)) for n, elid, _ in adjacency[v]):
        adjacency[v].append((u, int(lid), float(w)))


def _line_endpoint_clusters_from_terminal_points(terminal_points_nogroups):
    mapping = {}
    if terminal_points_nogroups is None or len(terminal_points_nogroups) == 0:
        return mapping
    for lid, g in terminal_points_nogroups.groupby('line_id'):
        clusters = [int(c) for c in g['cluster'].tolist()]
        mapping[int(lid)] = clusters
    return mapping


def _repair_graph_edges_from_terminal_points(df, terminal_points_nogroups, edges_by_line, adjacency):
    """
    Integrity pass:
    - every line_id in df should appear in edges_by_line
    - if not, first try the endpoint cluster labels directly
    - if that fails, infer start/end nodes from the nearest cluster centroids
      to the line's actual start and end points
    This is a normalization repair step for split-generated segments that
    exist in geometry but are not fully inserted into the routing graph.
    """
    import numpy as _np

    line_lengths = _line_length_map(df)
    line_ids = sorted(int(x) for x in df['line_id'].unique().tolist())
    cluster_map = _line_endpoint_clusters_from_terminal_points(terminal_points_nogroups)

    centroids = {}
    if terminal_points_nogroups is not None and len(terminal_points_nogroups) > 0:
        grouped = terminal_points_nogroups.groupby('cluster')[['x', 'y', 'z']].mean()
        for cid, row in grouped.iterrows():
            centroids[int(cid)] = _np.array([float(row['x']), float(row['y']), float(row['z'])], dtype=float)

    repaired_direct = []
    repaired_inferred = []
    diagnostics = []

    def _nearest_clusters(pt_xyz, k=3):
        if not centroids:
            return []
        pt = _np.array(pt_xyz, dtype=float)
        arr = []
        for cid, cxyz in centroids.items():
            arr.append((float(_np.linalg.norm(pt - cxyz)), int(cid)))
        arr.sort(key=lambda t: t[0])
        return arr[:k]

    for lid in line_ids:
        if int(lid) in edges_by_line:
            continue

        clusters = cluster_map.get(int(lid), [])
        uniq = list(dict.fromkeys(int(c) for c in clusters))

        g = df[df['line_id'] == lid].reset_index(drop=True)
        if len(g) >= 1:
            start_xyz = tuple(float(x) for x in g.iloc[0][['x','y','z']].values.tolist())
            end_xyz   = tuple(float(x) for x in g.iloc[-1][['x','y','z']].values.tolist())
        else:
            start_xyz, end_xyz = None, None

        w = float(line_lengths.get(int(lid), 0.0))

        if len(uniq) == 2:
            u, v = uniq
            edges_by_line[int(lid)] = {'u': u, 'v': v, 'weight': w}
            _add_edge_to_adjacency(adjacency, u, v, int(lid), w)
            repaired_direct.append(int(lid))
            continue

        inferred = False
        if start_xyz is not None and end_xyz is not None and centroids:
            start_cands = _nearest_clusters(start_xyz, k=4)
            end_cands   = _nearest_clusters(end_xyz,   k=4)

            best = None
            for ds, cs in start_cands:
                for de, ce in end_cands:
                    if cs == ce:
                        continue
                    cost = ds + de
                    if best is None or cost < best[0]:
                        best = (cost, cs, ce, ds, de)

            if best is not None:
                _, u, v, ds, de = best
                edges_by_line[int(lid)] = {'u': int(u), 'v': int(v), 'weight': w}
                _add_edge_to_adjacency(adjacency, int(u), int(v), int(lid), w)
                repaired_inferred.append((int(lid), int(u), int(v), float(ds), float(de)))
                inferred = True

        if inferred:
            continue

        diagnostics.append({
            'line_id': int(lid),
            'clusters': uniq,
            'n_clusters': len(uniq),
            'start': None if start_xyz is None else tuple(round(v, 6) for v in start_xyz),
            'end': None if end_xyz is None else tuple(round(v, 6) for v in end_xyz),
            'start_nearest': _nearest_clusters(start_xyz, k=3) if start_xyz is not None else [],
            'end_nearest': _nearest_clusters(end_xyz, k=3) if end_xyz is not None else [],
        })

    if repaired_direct:
        print("Graph integrity repair added missing graph edges from endpoint clusters for line_ids:", repaired_direct)

    if repaired_inferred:
        print("Graph integrity repair inferred missing graph edges from nearest node centroids:")
        for lid, u, v, ds, de in repaired_inferred:
            print(f"  line {lid}: u={u} (start_dist={ds:.4f}), v={v} (end_dist={de:.4f})")

    if diagnostics:
        print("WARNING: graph integrity unresolved for line_ids:")
        for d in diagnostics:
            print(
                f"  line {d['line_id']}: clusters={d['clusters']} (n={d['n_clusters']}), "
                f"start={d['start']}, end={d['end']}, "
                f"start_nearest={d['start_nearest']}, end_nearest={d['end_nearest']}"
            )

    return edges_by_line, adjacency

def eulerficator(df, terminal_points, nodes):
    """
    Top-level function that:
    - builds cluster+connectivity dictionaries,
    - repeatedly uses the validator + Fleury's algorithm
      to group lines into printable paths,
    - returns the final line / node ordering.

    Loopfix3/4-style behaviour:
    - Uses ALL line_ids present in df (not just terminal_points).
    - Never aborts early when valid_lines == []: it force-schedules the next
      unprinted line (bottom-up) to guarantee full geometry.
    - Detects lack of progress and forces a fallback to avoid infinite loops.
    """
    terminal_points_nogroups = terminal_points.reset_index()

    # Keep cluster/connectivity dictionaries based on detected terminals
    clusters = terminal_points_nogroups["cluster"].unique()

    # IMPORTANT: schedule over all lines in df
    lines = df["line_id"].unique()

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

    # Sort lines by min z -> bottom up (used for force-scheduling)
    min_z = df.groupby("line_id")["z"].min()
    heightsorted_line_ids = min_z.sort_values().index.tolist()

    # Safety cap (should not be hit if progress logic works)
    max_iterations = max(10, len(lines) * 5)
    it = 0

    while sum(len(sublist) for sublist in edges_grouped) < len(lines):
        it += 1
        if it > max_iterations:
            print("Safety break triggered in eulerficator().")
            break

        edges_ungrouped = [item for sublist in edges_grouped for item in sublist]
        printed_set = set([seg['line_id'] if isinstance(seg, dict) else seg for seg in edges_ungrouped])

        unprinted_lines = [n for n in heightsorted_line_ids if n not in printed_set]
        if not unprinted_lines:
            break

        # Progress snapshot BEFORE attempting Fleury
        printed_before = set(printed_set)

        valid_lines = validator(unprinted_lines, df)

        # Guarantee no remaining df lines are silently excluded from routing.
        valid_lines = list(dict.fromkeys([int(v) for v in valid_lines] + [int(v) for v in unprinted_lines]))

        # If validator says nothing is printable, force-schedule next unprinted line
        if not valid_lines:
            fallback_line = int(unprinted_lines[0])
            if fallback_line not in printed_set:
                edges_grouped.append([{'line_id': fallback_line, 'reverse': False, 'extrude': True}])
                node_path_grouped.append([])  # no guaranteed node path
                print(
                    f"No valid lines; forcing print of line {fallback_line} "
                    "to guarantee full geometry."
                )
            continue

        grouped_cpp_paths = chinese_postman_paths_for_valid_lines(cluster_dict, valid_lines, df, nodes)
        for path in grouped_cpp_paths:
            edges_grouped.append(path)
            node_path_grouped.append([])

        # Progress snapshot AFTER path optimisation
        printed_after = set([seg['line_id'] if isinstance(seg, dict) else seg
                             for sublist in edges_grouped for seg in sublist])
        newly_printed = printed_after - printed_before

        # If no progress, force-schedule a valid line to break deadlock
        if not newly_printed:
            fallback_line = int(valid_lines[0])
            if fallback_line not in printed_after:
                edges_grouped.append([{'line_id': fallback_line, 'reverse': False, 'extrude': True}])
                node_path_grouped.append([])
                print(
                    f"Fleury made no progress; forcing print of line {fallback_line} "
                    "to prevent infinite loop."
                )
            continue

    # Flatten for outputs
    node_order = [item for sublist in node_path_grouped for item in sublist]
    line_order = [seg['line_id'] if isinstance(seg, dict) else seg for sublist in edges_grouped for seg in sublist]

    # Keep repeated non-extrusion traversals, but audit printable coverage using only extruding segments.
    line_order_grouped = edges_grouped
    node_order_grouped = node_path_grouped

    printed_extrude = []
    for path in line_order_grouped:
        for seg in path:
            if isinstance(seg, dict):
                if seg.get('extrude', True):
                    printed_extrude.append(int(seg['line_id']))
            else:
                printed_extrude.append(int(seg))

    # Final full-graph extraction used both for continuity stitching and for
    # attaching any leftover lines into the same continuous route.
    try:
        full_valid_line_cluster_dict = _build_valid_line_cluster_dict(cluster_dict, lines)
        _, full_edges_by_line, full_adjacency = _extract_component_edges(full_valid_line_cluster_dict, _line_length_map(df))
        full_edges_by_line, full_adjacency = _repair_graph_edges_from_terminal_points(
            df, terminal_points_nogroups, full_edges_by_line, full_adjacency
        )
    except Exception as _full_graph_err:
        full_edges_by_line, full_adjacency = {}, {}
        print("WARNING: full-graph extraction failed:", _full_graph_err)

    # Final audit / full-geometry safeguard
    missing = sorted(set(lines) - set(printed_extrude))
    if missing:
        print("WARNING: Some lines were not scheduled for printing:", missing)
        # Append missing split segments as one-edge paths, but crucially include
        # start/end nodes from the full graph so the continuity stitch can bridge
        # them into the SAME continuous route instead of leaving them detached.
        for lid in missing:
            edge = full_edges_by_line.get(int(lid))
            if edge is not None:
                seg = {
                    'line_id': int(lid),
                    'reverse': False,
                    'extrude': True,
                    'start_node': edge['u'],
                    'end_node': edge['v'],
                }
            else:
                seg = {'line_id': int(lid), 'reverse': False, 'extrude': True}
            line_order_grouped.append([seg])
            node_order_grouped.append([])

        node_order = [item for sublist in node_path_grouped for item in sublist]
        line_order = [seg['line_id'] if isinstance(seg, dict) else seg for sublist in line_order_grouped for seg in sublist]

    # Do NOT force continuity after planning.
    # In a support bath, respecting print-history constraints is more important
    # than enforcing one continuous route. If continuity would require revisiting
    # already printed geometry, we keep separate paths instead.
    node_order_grouped = [[] for _ in line_order_grouped]
    node_order = [item for sublist in node_order_grouped for item in sublist]
    line_order = [seg['line_id'] if isinstance(seg, dict) else seg for sublist in line_order_grouped for seg in sublist]

    # Final audit: a line counts as scheduled if it is extruded at least once
    # in the final non-stitched route.
    scheduled_extrude = set()
    for path in line_order_grouped:
        for seg in path:
            if isinstance(seg, dict):
                if seg.get('extrude', True):
                    scheduled_extrude.add(int(seg['line_id']))
            else:
                scheduled_extrude.add(int(seg))

    missing = sorted(set(int(x) for x in df['line_id'].unique().tolist()) - scheduled_extrude)
    if missing:
        print("WARNING: Some lines were not scheduled for printing:", missing)

    return line_order, node_order, line_order_grouped, node_order_grouped




def _find_edge_between_nodes(edges_by_line, a, b):
    for lid, edge in edges_by_line.items():
        u = edge.get('u')
        v = edge.get('v')
        if (u == a and v == b) or (u == b and v == a):
            return int(lid), edge
    return None, None


def _ensure_segment_nodes(seg, edges_by_line):
    if 'start_node' in seg and 'end_node' in seg:
        return seg
    lid = int(seg['line_id'])
    edge = edges_by_line.get(lid)
    if edge is None:
        return seg
    out = dict(seg)
    if bool(out.get('reverse', False)):
        out['start_node'] = edge['v']
        out['end_node'] = edge['u']
    else:
        out['start_node'] = edge['u']
        out['end_node'] = edge['v']
    return out


def _force_continuous_path(grouped_paths, adjacency_full, edges_by_line_full):
    nonempty = [list(p) for p in grouped_paths if p]
    if not nonempty:
        return grouped_paths

    merged = [_ensure_segment_nodes(seg, edges_by_line_full) for seg in nonempty[0]]

    for raw_next in nonempty[1:]:
        next_path = [_ensure_segment_nodes(seg, edges_by_line_full) for seg in raw_next]
        if not merged:
            merged.extend(next_path)
            continue

        last_seg = _ensure_segment_nodes(merged[-1], edges_by_line_full)
        next_seg = _ensure_segment_nodes(next_path[0], edges_by_line_full)
        start_node = last_seg.get('end_node')
        target_node = next_seg.get('start_node')

        if start_node is not None and target_node is not None and start_node != target_node:
            dist_map, prev = _dijkstra_shortest_paths(adjacency_full, start_node)
            if target_node in dist_map:
                node_path, line_path = _reconstruct_node_line_path(prev, start_node, target_node)
                for a, b, lid in zip(node_path[:-1], node_path[1:], line_path):
                    _, edge = _find_edge_between_nodes(edges_by_line_full, a, b)
                    reverse = True
                    if edge is not None:
                        reverse = not (edge['u'] == a and edge['v'] == b)
                    merged.append({
                        'line_id': int(lid),
                        'reverse': bool(reverse),
                        'extrude': False,
                        'start_node': a,
                        'end_node': b,
                    })
        merged.extend(next_path)

    return [merged]



def _segment_xyz_endpoints(df, seg):
    """Return (start_xyz, end_xyz) for a segment using line geometry or custom_points."""
    if isinstance(seg, dict) and seg.get('custom_points') is not None:
        pts = seg.get('custom_points') or []
        if len(pts) >= 2:
            return tuple(float(v) for v in pts[0]), tuple(float(v) for v in pts[-1])
    lid = int(seg['line_id'])
    g = df[df['line_id'] == lid].copy().reset_index(drop=True)
    if bool(seg.get('reverse', False)):
        g = g.iloc[::-1].reset_index(drop=True)
    return (
        tuple(float(v) for v in g.iloc[0][['x','y','z']].values.tolist()),
        tuple(float(v) for v in g.iloc[-1][['x','y','z']].values.tolist()),
    )


def _force_continuous_path_robust(grouped_paths, adjacency_full, edges_by_line_full, df):
    """
    Merge grouped paths into one continuous route. Prefer graph travel-only bridges.
    If no graph bridge exists, insert a direct travel-only custom segment so the
    final route remains geometrically continuous for animation and G-code.
    """
    nonempty = [list(p) for p in grouped_paths if p]
    if not nonempty:
        return grouped_paths

    merged = [_ensure_segment_nodes(seg, edges_by_line_full) if isinstance(seg, dict) else seg for seg in nonempty[0]]

    for raw_next in nonempty[1:]:
        next_path = [_ensure_segment_nodes(seg, edges_by_line_full) if isinstance(seg, dict) else seg for seg in raw_next]
        if not merged:
            merged.extend(next_path)
            continue

        last_seg = merged[-1]
        next_seg = next_path[0]

        _, last_end_xyz = _segment_xyz_endpoints(df, last_seg)
        next_start_xyz, _ = _segment_xyz_endpoints(df, next_seg)

        last_seg = _ensure_segment_nodes(last_seg, edges_by_line_full) if isinstance(last_seg, dict) else last_seg
        next_seg = _ensure_segment_nodes(next_seg, edges_by_line_full) if isinstance(next_seg, dict) else next_seg

        start_node = last_seg.get('end_node') if isinstance(last_seg, dict) else None
        target_node = next_seg.get('start_node') if isinstance(next_seg, dict) else None

        current_xyz = last_end_xyz

        if start_node is not None and target_node is not None and start_node != target_node:
            dist_map, prev = _dijkstra_shortest_paths(adjacency_full, start_node)
            if target_node in dist_map:
                node_path, line_path = _reconstruct_node_line_path(prev, start_node, target_node)
                for a, b, lid in zip(node_path[:-1], node_path[1:], line_path):
                    _, edge = _find_edge_between_nodes(edges_by_line_full, a, b)
                    reverse = True
                    if edge is not None:
                        reverse = not (edge['u'] == a and edge['v'] == b)
                    travel_seg = {
                        'line_id': int(lid),
                        'reverse': bool(reverse),
                        'extrude': False,
                        'start_node': a,
                        'end_node': b,
                    }
                    merged.append(travel_seg)
                _, current_xyz = _segment_xyz_endpoints(df, merged[-1])

        if tuple(round(v, 6) for v in current_xyz) != tuple(round(v, 6) for v in next_start_xyz):
            merged.append({
                'line_id': -1,
                'reverse': False,
                'extrude': False,
                'custom_points': [current_xyz, next_start_xyz],
                'start_node': start_node,
                'end_node': target_node,
            })

        merged.extend(next_path)

    return [merged]


def _line_length_map(df):
    """Return {line_id: polyline length}."""
    out = {}
    for line_id, g in df.groupby('line_id'):
        xyz = g[['x', 'y', 'z']].astype(float).to_numpy()
        if len(xyz) < 2:
            out[int(line_id)] = 0.0
            continue
        diffs = np.diff(xyz, axis=0)
        seg = np.sqrt((diffs ** 2).sum(axis=1))
        out[int(line_id)] = float(seg.sum())
    return out


def _build_valid_line_cluster_dict(cluster_dictionary, valid_lines):
    valid_lines = set(int(v) for v in valid_lines)
    out = {}
    for cluster, lines in cluster_dictionary.items():
        out[cluster] = [int(item) for item in lines if int(item) in valid_lines]
    return out


def _extract_component_edges(valid_line_cluster_dict, line_lengths):
    """
    Build a simple undirected multigraph representation from node->line map.

    Returns
    -------
    nodes : set
    edges_by_line : dict
        line_id -> {'u','v','weight'}
    adjacency : dict
        node -> list of (neighbor, line_id, weight)
    """
    line_to_nodes = {}
    for node, lines in valid_line_cluster_dict.items():
        for lid in lines:
            line_to_nodes.setdefault(int(lid), []).append(node)

    edges_by_line = {}
    adjacency = {node: [] for node in valid_line_cluster_dict.keys()}
    for lid, nodes in line_to_nodes.items():
        uniq = list(dict.fromkeys(nodes))
        if len(uniq) != 2:
            continue
        u, v = uniq
        w = float(line_lengths.get(int(lid), 0.0))
        edges_by_line[int(lid)] = {'u': u, 'v': v, 'weight': w}
        adjacency.setdefault(u, []).append((v, int(lid), w))
        adjacency.setdefault(v, []).append((u, int(lid), w))

    nodes = set(adjacency.keys())
    return nodes, edges_by_line, adjacency


def _connected_components_from_adjacency(adjacency):
    visited = set()
    components = []
    for start in adjacency.keys():
        if start in visited:
            continue
        stack = [start]
        comp = set()
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            for neigh, _, _ in adjacency.get(node, []):
                if neigh not in visited:
                    stack.append(neigh)
        if comp:
            components.append(comp)
    return components


def _dijkstra_shortest_paths(adjacency, start):
    import heapq
    dist_map = {start: 0.0}
    prev = {start: None}
    pq = [(0.0, start)]
    while pq:
        dist_u, u = heapq.heappop(pq)
        if dist_u > dist_map.get(u, float('inf')) + 1e-12:
            continue
        for v, lid, w in adjacency.get(u, []):
            cand = dist_u + float(w)
            if cand + 1e-12 < dist_map.get(v, float('inf')):
                dist_map[v] = cand
                prev[v] = (u, int(lid))
                heapq.heappush(pq, (cand, v))
    return dist_map, prev


def _reconstruct_node_line_path(prev, start, end):
    if end not in prev:
        return [], []
    nodes_rev = [end]
    lines_rev = []
    cur = end
    while cur != start:
        item = prev.get(cur)
        if item is None:
            return [], []
        parent, lid = item
        lines_rev.append(int(lid))
        nodes_rev.append(parent)
        cur = parent
    return list(reversed(nodes_rev)), list(reversed(lines_rev))


def _all_pairs_shortest_component(adjacency, nodes):
    shortest = {}
    for s in nodes:
        dist_map, prev = _dijkstra_shortest_paths(adjacency, s)
        for t in nodes:
            if t == s or t not in dist_map:
                continue
            node_path, line_path = _reconstruct_node_line_path(prev, s, t)
            shortest[(s, t)] = {
                'cost': float(dist_map[t]),
                'nodes': node_path,
                'lines': line_path,
            }
    return shortest


def _min_weight_matching(nodes_tuple, shortest, memo=None):
    """Greedy minimum-weight matching fallback to avoid exponential recursion."""
    nodes = list(sorted(nodes_tuple))
    if not nodes:
        return 0.0, []

    pairs = []
    total_cost = 0.0

    while nodes:
        a = nodes.pop(0)
        if not nodes:
            break

        best_j = None
        best_cost = float('inf')
        for j, b in enumerate(nodes):
            key = (a, b) if (a, b) in shortest else (b, a)
            c = float(shortest[key]['cost'])
            if c < best_cost:
                best_cost = c
                best_j = j

        b = nodes.pop(best_j)
        pairs.append((a, b))
        total_cost += best_cost

    return total_cost, pairs


def _choose_open_cpp_endpoints(component_nodes, adjacency, odd_nodes):
    """Choose distinct start/end nodes that minimise open Chinese-postman augmentation cost."""
    shortest = _all_pairs_shortest_component(adjacency, component_nodes)
    node_list = sorted(component_nodes)
    if len(node_list) < 2:
        return None, None, shortest, []

    best = None
    odd_set = set(odd_nodes)
    for i, s in enumerate(node_list):
        for t in node_list[i+1:]:
            toggled = sorted(odd_set.symmetric_difference({s, t}))
            match_cost, pairs = _min_weight_matching(tuple(toggled), shortest)
            total = float(match_cost)
            candidate = (total, s, t, pairs)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        return node_list[0], node_list[-1], shortest, []
    _, s, t, pairs = best
    return s, t, shortest, pairs


def _build_augmented_multigraph(component_nodes, edges_by_line, adjacency):
    odd_nodes = [n for n in component_nodes if len(adjacency.get(n, [])) % 2 == 1]
    start_node, end_node, shortest, pairs = _choose_open_cpp_endpoints(component_nodes, adjacency, odd_nodes)

    multiedges = []
    adjacency_multi = {n: [] for n in component_nodes}

    def _add_edge(u, v, base_line_id, extrude):
        eid = len(multiedges)
        multiedges.append({
            'u': u,
            'v': v,
            'line_id': int(base_line_id),
            'extrude': bool(extrude),
        })
        adjacency_multi[u].append(eid)
        adjacency_multi[v].append(eid)

    # original printable edges
    for lid, edge in sorted(edges_by_line.items()):
        _add_edge(edge['u'], edge['v'], lid, True)

    # duplicate along shortest paths, but without extrusion in the eventual writer
    for a, b in pairs:
        for lid in shortest[(a, b)]['lines']:
            edge = edges_by_line[int(lid)]
            _add_edge(edge['u'], edge['v'], lid, False)

    return start_node, end_node, multiedges, adjacency_multi


def _hierholzer_open_trail(start_node, multiedges, adjacency_multi):
    if start_node is None:
        return []

    used = [False] * len(multiedges)
    local_adj = {n: list(eids) for n, eids in adjacency_multi.items()}
    stack = [(start_node, None)]
    trail = []

    while stack:
        node, via_eid = stack[-1]
        while local_adj.get(node):
            eid = local_adj[node].pop()
            if used[eid]:
                continue
            used[eid] = True
            edge = multiedges[eid]
            nxt = edge['v'] if edge['u'] == node else edge['u']
            stack.append((nxt, eid))
            node = nxt
        popped_node, popped_eid = stack.pop()
        if popped_eid is not None:
            prev_node = stack[-1][0]
            edge = multiedges[popped_eid]
            reverse = not (edge['u'] == prev_node and edge['v'] == popped_node)
            trail.append({
                'line_id': int(edge['line_id']),
                'reverse': bool(reverse),
                'extrude': bool(edge['extrude']),
                'start_node': prev_node,
                'end_node': popped_node,
            })

    trail.reverse()
    return trail


def _orient_trail_by_node_centroids(trail, df, nodes):
    """Set each segment's reverse flag by matching its geometric endpoints to
    the requested start_node / end_node centroids. This keeps the graph-level
    CPP order intact while making the actual line geometry follow that order."""
    if not trail:
        return trail
    if nodes is None or len(nodes) == 0:
        return trail

    line_endpoints = {}
    for line_id, g in df.groupby('line_id', sort=False):
        g = g.reset_index(drop=True)
        if g.empty:
            continue
        p0 = g.iloc[0][['x','y','z']].to_numpy(dtype=float)
        p1 = g.iloc[-1][['x','y','z']].to_numpy(dtype=float)
        line_endpoints[int(line_id)] = (p0, p1)

    for seg in trail:
        if not isinstance(seg, dict):
            continue
        lid = int(seg['line_id'])
        if lid not in line_endpoints:
            continue
        s_node = seg.get('start_node')
        e_node = seg.get('end_node')
        if s_node not in nodes.index or e_node not in nodes.index:
            continue

        p0, p1 = line_endpoints[lid]
        ns = nodes.loc[s_node][['x','y','z']].to_numpy(dtype=float)
        ne = nodes.loc[e_node][['x','y','z']].to_numpy(dtype=float)

        cost_fwd = float(np.linalg.norm(p0 - ns) + np.linalg.norm(p1 - ne))
        cost_rev = float(np.linalg.norm(p1 - ns) + np.linalg.norm(p0 - ne))
        seg['reverse'] = bool(cost_rev + 1e-12 < cost_fwd)
    return trail


def chinese_postman_paths_for_valid_lines(cluster_dictionary, valid_lines, df, nodes=None):
    """
    Return grouped traversal plans for a valid-line subset.

    Each segment in a path is a dict with:
        line_id, reverse, extrude, start_node, end_node

    Original lines are traversed with extrude=True exactly once. Repeated edges added by
    the open Chinese-postman augmentation are returned with extrude=False.
    """
    line_lengths = _line_length_map(df)
    valid_line_cluster_dict = _build_valid_line_cluster_dict(cluster_dictionary, valid_lines)
    _, edges_by_line, adjacency = _extract_component_edges(valid_line_cluster_dict, line_lengths)

    grouped_paths = []
    for comp_nodes in _connected_components_from_adjacency(adjacency):
        comp_edges = {lid: edge for lid, edge in edges_by_line.items() if edge['u'] in comp_nodes and edge['v'] in comp_nodes}
        comp_adj = {n: [(v, lid, w) for (v, lid, w) in adjacency.get(n, []) if v in comp_nodes] for n in comp_nodes}
        if not comp_edges:
            continue
        start_node, end_node, multiedges, adjacency_multi = _build_augmented_multigraph(comp_nodes, comp_edges, comp_adj)
        trail = _hierholzer_open_trail(start_node, multiedges, adjacency_multi)
        if trail:
            trail = _orient_trail_by_node_centroids(trail, df, nodes)
            grouped_paths.append(trail)

    return grouped_paths

def _reverse_segment(seg):
    new = dict(seg)
    new['reverse'] = not bool(seg.get('reverse', False))
    if 'start_node' in seg and 'end_node' in seg:
        new['start_node'], new['end_node'] = seg.get('end_node'), seg.get('start_node')
    return new


def _reverse_path_segments(path):
    return [_reverse_segment(seg) for seg in reversed(path)]


def _travel_segments_from_node_line_path(node_path, line_path, edges_by_line):
    segs = []
    for a, b, lid in zip(node_path[:-1], node_path[1:], line_path):
        edge = edges_by_line.get(int(lid))
        if edge is None:
            continue
        reverse = not (edge['u'] == a and edge['v'] == b)
        segs.append({
            'line_id': int(lid),
            'reverse': bool(reverse),
            'extrude': False,
            'start_node': a,
            'end_node': b,
        })
    return segs


def _stitch_grouped_paths_via_full_graph(grouped_paths, adjacency_full, edges_by_line_full):
    """
    Keep the existing per-path planning intact, but connect successive paths
    with travel-only shortest paths over the full processed graph.
    """
    if not grouped_paths:
        return grouped_paths

    nonempty = [p for p in grouped_paths if p]
    if not nonempty:
        return grouped_paths

    stitched = list(nonempty[0])
    current_end = stitched[-1].get('end_node')

    for raw_next in nonempty[1:]:
        path_fwd = raw_next
        path_rev = _reverse_path_segments(raw_next)

        candidates = []
        if current_end is not None:
            # forward orientation
            next_start = path_fwd[0].get('start_node')
            if next_start is not None:
                dist_map, prev = _dijkstra_shortest_paths(adjacency_full, current_end)
                if next_start in dist_map:
                    node_path, line_path = _reconstruct_node_line_path(prev, current_end, next_start)
                    bridge = _travel_segments_from_node_line_path(node_path, line_path, edges_by_line_full)
                    candidates.append((float(dist_map[next_start]), bridge, path_fwd))

            # reversed orientation
            next_start = path_rev[0].get('start_node')
            if next_start is not None:
                dist_map, prev = _dijkstra_shortest_paths(adjacency_full, current_end)
                if next_start in dist_map:
                    node_path, line_path = _reconstruct_node_line_path(prev, current_end, next_start)
                    bridge = _travel_segments_from_node_line_path(node_path, line_path, edges_by_line_full)
                    candidates.append((float(dist_map[next_start]), bridge, path_rev))

        if candidates:
            _, bridge, chosen_path = min(candidates, key=lambda x: x[0])
            stitched.extend(bridge)
            stitched.extend(chosen_path)
            current_end = stitched[-1].get('end_node')
        else:
            # Graph disconnected: preserve as a separate path by inserting nothing.
            # We append directly so geometry is still printed.
            stitched.extend(raw_next)
            current_end = stitched[-1].get('end_node')

    return [stitched]


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
    """
    Make each line run in the node order without altering the line geometry.

    Important change:
    - We no longer rely on node_finder() snapping endpoints onto node centroids.
    - Each line is flipped only if its *end* is closer to the scheduled start node
      than its current start. The coordinates themselves are otherwise preserved.
    """
    valid_line_ids = set(df['line_id'].unique().tolist())
    line_order = [int(l) for l in line_order if l is not None and int(l) in valid_line_ids]

    if not line_order:
        return df

    df = df.set_index('line_id').loc[line_order].reset_index()

    for idx_path, path in enumerate(line_order_grouped):
        for idx_line, line in enumerate(path):
            if node_order_grouped is None or idx_path >= len(node_order_grouped):
                continue
            if node_order_grouped[idx_path] is None or idx_line >= len(node_order_grouped[idx_path]):
                continue

            start_node = node_order_grouped[idx_path][idx_line]
            if start_node is None or start_node not in nodes.index:
                continue

            line_df = df[df['line_id'] == line]
            if line_df.empty:
                continue

            node_loc = nodes.loc[start_node][['x', 'y', 'z']].values.astype(float)
            line_start = line_df.iloc[0][['x', 'y', 'z']].values.astype(float)
            line_end = line_df.iloc[-1][['x', 'y', 'z']].values.astype(float)

            d_start = np.linalg.norm(line_start - node_loc)
            d_end = np.linalg.norm(line_end - node_loc)

            if d_end < d_start:
                new_line = line_df.iloc[::-1]
                df = df[df['line_id'] != line]
                df = pd.concat([df, new_line])

    df = df.set_index('line_id').loc[line_order].reset_index()
    df = distance_calculator(df)
    return df

def apply_node_nudge(df, nodes, nudge_mm=0.0, nudge_npts=0):
    """Move the last/first few *internal* points of each line slightly towards its
    terminal node coordinates to encourage fusion at junctions.

    This implements a geometry-correct version of 'move (a,b,c) towards (x,y,z) by X mm':
        p_new = p + u * delta
    where u is the unit vector from p to node and delta is capped so we never overshoot.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: x,y,z,line_id. Expected to already include the node
        centroid point appended at the start/end of each line (from node_finder()).
    nodes : pd.DataFrame
        Cluster centroids with columns x,y,z indexed by cluster id (from node_finder()).
    nudge_mm : float
        Maximum distance (mm) to move points towards the node.
    nudge_npts : int
        Number of points nearest each end (excluding the endpoint itself) to nudge.

    Returns
    -------
    pd.DataFrame
        Updated df with nudged coordinates.
    """
    if nudge_mm is None or nudge_mm <= 0 or nudge_npts is None or nudge_npts <= 0:
        return df

    df = df.copy().reset_index(drop=True)

    # Work line-by-line
    for line_id, g in df.groupby('line_id', sort=False):
        if len(g) < 3:
            continue  # need at least one internal point

        idxs = g.index.to_list()

        # endpoints (these should already be node centroids due to node_finder)
        start_idx = idxs[0]
        end_idx   = idxs[-1]

        start_node = df.loc[start_idx, ['x','y','z']].to_numpy(dtype=float)
        end_node   = df.loc[end_idx,   ['x','y','z']].to_numpy(dtype=float)

        # n internal points nearest each end (exclude the endpoint itself)
        n = int(min(nudge_npts, max(0, len(idxs) - 2)))
        if n <= 0:
            continue

        start_internal = idxs[1:1+n]
        end_internal   = idxs[-1-n:-1]

        def _nudge_row(row_idx, node_xyz, frac):
            p = df.loc[row_idx, ['x','y','z']].to_numpy(dtype=float)
            v = node_xyz - p
            dist = float(np.linalg.norm(v))
            if dist <= 1e-9:
                return
            # Move by up to (nudge_mm * frac) but never past the node
            delta = min(dist, float(nudge_mm) * float(frac))
            u = v / dist
            p2 = p + u * delta
            df.loc[row_idx, ['x','y','z']] = p2

        # Apply a ramp so points closer to the node move less, farther move more
        # (k runs 1..n, so the farthest internal point gets ~nudge_mm)
        for k, ridx in enumerate(start_internal, start=1):
            _nudge_row(ridx, start_node, frac=k/n)

        for k, ridx in enumerate(reversed(end_internal), start=1):
            _nudge_row(ridx, end_node, frac=k/n)

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


def node_finder(df, node_tol=0.25, snap_to_nodes=False):
    # Use hierarchical clustering to note common start / end points
    # Calculate node positions based on cluster centroids
    # IMPORTANT: by default this does NOT modify the line geometry.

    start_points = df.groupby('line_id').first()
    end_points = df.groupby('line_id').last()
    terminal_points = pd.concat([start_points, end_points])

    if len(terminal_points) == 0:
        nodes = pd.DataFrame(columns=df.columns)
        terminal_points['cluster'] = []
        return df, terminal_points, nodes

    if len(terminal_points) == 1:
        terminal_points = terminal_points.copy()
        terminal_points['cluster'] = 1
        nodes = terminal_points.groupby('cluster').mean()
        return df, terminal_points, nodes

    dist_mat = dist.pdist(terminal_points[['x', 'y', 'z']].values)
    link_mat = hier.linkage(dist_mat)
    cluster_idx = hier.fcluster(link_mat, t=node_tol, criterion='distance')
    terminal_points = terminal_points.copy()
    terminal_points['cluster'] = cluster_idx

    nodes = terminal_points.groupby('cluster').mean()

    if not snap_to_nodes:
        return df, terminal_points, nodes

    # Optional legacy behaviour: append node centroid to line start/end.
    df = df.copy()
    for n in terminal_points.index.unique():
        clusters = terminal_points.loc[n]['cluster']
        for c in np.atleast_1d(clusters):
            new_point = nodes.loc[c:c].copy()
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


def animate_postman_path(df, line_order_grouped, n_interp=50, show_ids=True):
    """Animate Chinese-postman traversal from line_order_grouped.

    Each segment may be either a plain line_id (legacy) or a dict with keys:
        {"line_id": int, "reverse": bool, "extrude": bool}

    Extruding segments are drawn solid; repeated non-extruding traversals are dashed.
    """

    def interpolate_line(line_data, n_points=50):
        if len(line_data) < 2:
            return line_data[['x', 'y', 'z']].to_numpy(dtype=float)

        t = np.linspace(0, 1, len(line_data))
        fx = interp1d(t, line_data['x'].astype(float), kind='linear')
        fy = interp1d(t, line_data['y'].astype(float), kind='linear')
        fz = interp1d(t, line_data['z'].astype(float), kind='linear')
        t_new = np.linspace(0, 1, n_points)
        return np.vstack((fx(t_new), fy(t_new), fz(t_new))).T

    ordered_segments = []
    for path in line_order_grouped:
        for seg in path:
            if isinstance(seg, dict):
                line_id = int(seg['line_id'])
                reverse = bool(seg.get('reverse', False))
                extrude = bool(seg.get('extrude', True))
            else:
                line_id = int(seg)
                reverse = False
                extrude = True

            line_data = df[df['line_id'] == line_id].copy().reset_index(drop=True)
            if line_data.empty:
                continue

            if reverse:
                line_data = line_data.iloc[::-1].reset_index(drop=True)

            pts = interpolate_line(line_data, n_points=n_interp)
            ordered_segments.append({
                'line_id': line_id,
                'reverse': reverse,
                'extrude': extrude,
                'pts': pts,
            })

    if not ordered_segments:
        raise ValueError('No segments found to animate.')

    frame_to_segment = []
    for i, seg in enumerate(ordered_segments):
        for j in range(len(seg['pts'])):
            frame_to_segment.append((i, j))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    z_min, z_max = df['z'].min(), df['z'].max()

    x_range = max(x_max - x_min, 1)
    y_range = max(y_max - y_min, 1)
    z_range = max(z_max - z_min, 1)

    ax.set_xlim(x_min - 1, x_max + 1)
    ax.set_ylim(y_min - 1, y_max + 1)
    ax.set_zlim(z_min - 1, z_max + 1)
    ax.set_box_aspect((x_range, y_range, z_range))
    ax.view_init(elev=30, azim=60)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Chinese Postman Print Path')

    plot_lines = []
    line_labels = []
    for seg in ordered_segments:
        linestyle = '-' if seg['extrude'] else '--'
        linewidth = 2 if seg['extrude'] else 1.5
        line_artist, = ax.plot([], [], [], linestyle=linestyle, linewidth=linewidth)
        plot_lines.append(line_artist)

        if show_ids:
            mid_idx = len(seg['pts']) // 2
            mid_pt = seg['pts'][mid_idx]
            suffix = '' if seg['extrude'] else ''
            label = ax.text(mid_pt[0], mid_pt[1], mid_pt[2], f"{seg['line_id']}{suffix}", fontsize=9, visible=False)
            line_labels.append(label)
        else:
            line_labels.append(None)

    head, = ax.plot([], [], [], marker='o', markersize=5)

    def update(frame):
        seg_idx, pt_idx = frame_to_segment[frame]

        for i in range(seg_idx + 1):
            pts = ordered_segments[i]['pts']
            end_idx = pt_idx + 1 if i == seg_idx else len(pts)

            x = pts[:end_idx, 0]
            y = pts[:end_idx, 1]
            z = pts[:end_idx, 2]

            plot_lines[i].set_data(x, y)
            plot_lines[i].set_3d_properties(z)

            if line_labels[i] is not None:
                line_labels[i].set_visible(True)

        curr_point = ordered_segments[seg_idx]['pts'][pt_idx]
        head.set_data([curr_point[0]], [curr_point[1]])
        head.set_3d_properties([curr_point[2]])

        artists = plot_lines + [head]
        artists += [lbl for lbl in line_labels if lbl is not None]
        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_to_segment),
        interval=20,
        blit=False,
        repeat=False
    )

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


def reorder_points_nearest_neighbour(shape: pd.DataFrame) -> pd.DataFrame:
    """Reorder points in a line_id group into a continuous path using a greedy
    nearest-neighbour walk. This prevents long 'teleport' segments when the export
    order is scrambled or when multiple segments share a line_id.

    Works well for smooth vessel paths and is fast enough for typical point counts.
    """
    if len(shape) <= 2:
        return shape

    coords = shape[['x','y','z']].to_numpy(dtype=float)

    # Choose a start point: one end of the diameter (farthest pair approx)
    # Approx by picking point farthest from centroid
    centroid = coords.mean(axis=0)
    d2 = np.sum((coords - centroid)**2, axis=1)
    start_idx = int(np.argmax(d2))

    visited = np.zeros(len(shape), dtype=bool)
    order = []
    current = start_idx

    for _ in range(len(shape)):
        order.append(current)
        visited[current] = True
        # find nearest unvisited
        unvis = np.where(~visited)[0]
        if unvis.size == 0:
            break
        diffs = coords[unvis] - coords[current]
        dist2 = np.einsum('ij,ij->i', diffs, diffs)
        current = int(unvis[int(np.argmin(dist2))])

    # Reindex shape in that order
    return shape.iloc[order].reset_index(drop=True)



def _xy_linestring_from_df(line_df):
    coords = line_df[['x','y']].astype(float).to_numpy()
    if len(coords) < 2:
        return None
    return LineString(coords)


def _project_xy_onto_segment(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return 0.0, math.hypot(px - ax, py - ay)
    t = ((px - ax) * vx + (py - ay) * vy) / denom
    t_clamped = min(1.0, max(0.0, t))
    qx = ax + t_clamped * vx
    qy = ay + t_clamped * vy
    d = math.hypot(px - qx, py - qy)
    return t_clamped, d


def _collect_graph_intersections(df, z_tol=0.05, xy_tol=1e-6, near_xy_tol=None):
    """Find graph intersections between sampled polylines and return per-line split points.

    Handles both exact XY intersections and *near* intersections between discretely sampled
    segments by inserting a shared split point when the closest XY approach is within
    near_xy_tol and the local Z values agree within z_tol.
    """
    line_ids = sorted(int(v) for v in df['line_id'].unique())
    line_dfs = {lid: df[df['line_id'] == lid].copy().reset_index(drop=True) for lid in line_ids}
    line_xy = {lid: _xy_linestring_from_df(g) for lid, g in line_dfs.items()}
    split_points = {lid: [] for lid in line_ids}
    near_xy_tol = float(xy_tol if near_xy_tol is None else near_xy_tol)

    def _append_shared_point(lid1, lid2, x, y):
        z1 = interpolated_z((x, y), line_dfs[lid1][['x', 'y', 'z']].copy())
        z2 = interpolated_z((x, y), line_dfs[lid2][['x', 'y', 'z']].copy())
        if z1 is None or z2 is None:
            return
        if abs(float(z1) - float(z2)) > float(z_tol):
            return
        split_points[lid1].append((float(x), float(y), float(z1)))
        split_points[lid2].append((float(x), float(y), float(z2)))

    def _handle_geom_intersection(lid1, lid2, inter):
        if isinstance(inter, Point):
            _append_shared_point(lid1, lid2, float(inter.x), float(inter.y))
        elif isinstance(inter, MultiPoint):
            for pt in inter.geoms:
                _append_shared_point(lid1, lid2, float(pt.x), float(pt.y))
        elif isinstance(inter, GeometryCollection):
            for geom in inter.geoms:
                if isinstance(geom, Point):
                    _append_shared_point(lid1, lid2, float(geom.x), float(geom.y))

    for i, lid1 in enumerate(line_ids):
        ls1 = line_xy.get(lid1)
        if ls1 is None:
            continue
        for lid2 in line_ids[i+1:]:
            ls2 = line_xy.get(lid2)
            if ls2 is None:
                continue

            if ls1.intersects(ls2):
                _handle_geom_intersection(lid1, lid2, ls1.intersection(ls2))
                continue

            # Near-miss support for sampled curves: if the closest XY approach is within tolerance,
            # create a shared split point at the midpoint of the closest approach.
            try:
                dxy = float(ls1.distance(ls2))
            except Exception:
                dxy = np.inf
            if dxy <= near_xy_tol:
                try:
                    p1, p2 = nearest_points(ls1, ls2)
                    x = 0.5 * (float(p1.x) + float(p2.x))
                    y = 0.5 * (float(p1.y) + float(p2.y))
                    _append_shared_point(lid1, lid2, x, y)
                except Exception:
                    pass
    return split_points


def _insert_split_points_into_line(line_df, points, xy_tol=1e-6):
    if not points or len(line_df) < 2:
        return [line_df]

    work = line_df.copy().reset_index(drop=True)
    xy_line = _xy_linestring_from_df(work)
    if xy_line is None:
        return [work]

    # sort points by arclength along the line and dedupe near-identical XY points
    pts_sorted = []
    seen = []
    for x, y, z in sorted(points, key=lambda p: xy_line.project(Point(float(p[0]), float(p[1])))):
        if any(math.hypot(x - sx, y - sy) <= xy_tol for sx, sy in seen):
            continue
        seen.append((x, y))
        pts_sorted.append((float(x), float(y), float(z)))

    rows = work.to_dict('records')
    split_row_indices = []

    for x, y, z in pts_sorted:
        # skip if already an endpoint / existing point
        existing_idx = None
        for idx, r in enumerate(rows):
            if math.hypot(float(r['x']) - x, float(r['y']) - y) <= xy_tol:
                existing_idx = idx
                break
        if existing_idx is not None:
            if 0 < existing_idx < len(rows) - 1:
                split_row_indices.append(existing_idx)
            continue

        best = None
        for idx in range(len(rows) - 1):
            a, b = rows[idx], rows[idx + 1]
            t, d = _project_xy_onto_segment(x, y, float(a['x']), float(a['y']), float(b['x']), float(b['y']))
            if d <= xy_tol and 1e-6 < t < 1 - 1e-6:
                best = (idx, t)
                break
        if best is None:
            continue
        idx, t = best
        a, b = rows[idx], rows[idx + 1]
        new_row = dict(a)
        new_row['x'] = x
        new_row['y'] = y
        # interpolate z from the local segment if supplied z is poor
        new_row['z'] = float(a['z']) + t * (float(b['z']) - float(a['z'])) if np.isfinite(z) else z
        rows.insert(idx + 1, new_row)
        split_row_indices.append(idx + 1)

    split_row_indices = sorted(set(i for i in split_row_indices if 0 < i < len(rows)-1))
    if not split_row_indices:
        return [pd.DataFrame(rows)]

    segments = []
    start = 0
    for idx in split_row_indices:
        seg_rows = rows[start:idx+1]
        if len(seg_rows) >= 2:
            segments.append(pd.DataFrame(seg_rows))
        start = idx
    seg_rows = rows[start:]
    if len(seg_rows) >= 2:
        segments.append(pd.DataFrame(seg_rows))
    return segments if segments else [pd.DataFrame(rows)]


def split_lines_at_graph_intersections(df, z_tol=0.05, xy_tol=1e-6, near_xy_tol=None):
    """Split polylines at true graph intersections so routing sees a connected graph."""
    df = df.copy().reset_index(drop=True)
    split_points = _collect_graph_intersections(df, z_tol=z_tol, xy_tol=xy_tol, near_xy_tol=near_xy_tol)
    if not any(split_points.values()):
        return df

    new_segments = []
    next_line_id = 0
    sample_cols = [c for c in df.columns if c in ['r', 'g', 'b']]
    for old_lid in sorted(int(v) for v in df['line_id'].unique()):
        line_df = df[df['line_id'] == old_lid].copy().reset_index(drop=True)
        pieces = _insert_split_points_into_line(line_df, split_points.get(old_lid, []), xy_tol=xy_tol)
        for piece in pieces:
            piece = piece.copy().reset_index(drop=True)
            piece['line_id'] = next_line_id
            if sample_cols:
                for c in sample_cols:
                    piece[c] = line_df.iloc[0][c]
            new_segments.append(piece)
            next_line_id += 1
    out = pd.concat(new_segments, ignore_index=True)
    out = distance_calculator(out)
    return out

def preprocess(df, settings):
    # If line ID column doesn't exist, assign line IDs based on RGB values
    if 'line_id' not in df.columns:
        df['line_id'] = pd.factorize(df[['r', 'g', 'b']].apply(tuple, axis=1))[0]

    # Ensure a stable ordering: group by line_id and reorder points into a continuous path.
    df = df.groupby('line_id', group_keys=False).apply(reorder_points_nearest_neighbour)

    # Compute distances within each line
    df = distance_calculator(df)

    # Remove large jumps at the end of lines (if export wraps end->start)
    df = df.groupby('line_id', group_keys=False).apply(remove_overlap)

    # Recompute distances after any trimming
    df = distance_calculator(df)

    # Split lines that still have big jumps in the middle (multiple disconnected segments sharing a line_id)
    df = shapesplitter(df)

    # Recompute distances after splitting
    df = distance_calculator(df)

    # Split lines at true graph intersections so routing sees shared nodes
    if settings.get('split_graph_intersections', True):
        df = split_lines_at_graph_intersections(
            df,
            z_tol=float(settings.get('graph_intersection_z_tol', 0.05)),
            xy_tol=float(settings.get('graph_intersection_xy_tol', 1e-3)),
            near_xy_tol=float(settings.get('graph_near_intersection_xy_tol', settings.get('graph_intersection_xy_tol', 1e-3))),
        )
        df = distance_calculator(df)

    # Apply XY offsets/centering
    df['x'] = df['x'] - df['x'].mean() + settings['x_offset']
    df['y'] = df['y'] - df['y'].mean() + settings['y_offset']

    return df


def shape_prep(settings):
    # Load pattern data and convert to DataFrame
    df = pd.read_csv(os.path.join(settings['filedir'], settings['filename']), header=None)
    df.columns = ['x', 'y', 'z', 'r', 'g', 'b']

    df = preprocess(df, settings)

    # Find nodes via hierarchical clustering
    # Use a slightly more permissive tolerance for graph connectivity than the writer's
    # endpoint matching tolerance. This preserves your existing node_tol setting for
    # writer-side revisit matching, while avoiding route graphs that fragment into
    # isolated single edges when CAD endpoints are only approximately coincident.
    graph_node_tol = settings.get('graph_node_tol', max(float(settings.get('node_tol', 0.25)), 0.25))
    df, terminal_points, nodes = node_finder(df, graph_node_tol, snap_to_nodes=False)
    # Shared nodes are clusters that contain endpoints from 2+ line endpoints (i.e., real junctions).
    cluster_counts = terminal_points.groupby('cluster').size()
    shared_clusters = cluster_counts[cluster_counts > 1].index.tolist()
    if shared_clusters:
        shared_nodes_df = nodes.loc[shared_clusters][['x','y','z']].copy()
        settings['shared_nodes'] = shared_nodes_df[['x','y','z']].values.tolist()
    else:
        settings['shared_nodes'] = []

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

    # Line orientation is now handled per-traversal in the writer so we keep the original
    # per-line geometry here instead of imposing one global direction.

    # Optional node nudge: move the last/first few points of each line slightly towards node coords
    df = apply_node_nudge(df, nodes, settings.get('node_nudge_mm', 0.0), settings.get('node_nudge_npts', 0))
    df = distance_calculator(df)

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