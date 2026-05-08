import numpy as np
import math


def _match_shared_node(pt, shared_nodes, tol):
    """Return a canonical shared-node tuple (x,y,z) if pt is within tol of a shared node, else None."""
    if not shared_nodes:
        return None
    x, y, z = pt
    tol2 = tol * tol
    best = None
    best_d2 = None
    for n in shared_nodes:
        nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
        dx = x - nx
        dy = y - ny
        dz = z - nz
        d2 = dx * dx + dy * dy + dz * dz
        if d2 <= tol2 and (best_d2 is None or d2 < best_d2):
            best = (round(nx, 3), round(ny, 3), round(nz, 3))
            best_d2 = d2
    return best


def _line_arclengths(line_xyz):
    """Given Nx3 array, return s_from_start and s_to_end arrays."""
    diffs = np.diff(line_xyz, axis=0)
    seg = np.sqrt((diffs ** 2).sum(axis=1))
    s_from_start = np.concatenate([[0.0], np.cumsum(seg)])
    s_to_end = s_from_start[-1] - s_from_start
    return s_from_start, s_to_end


def _segment_line_id(segment):
    return int(segment['line_id']) if isinstance(segment, dict) else int(segment)


def _segment_extrudes(segment):
    return bool(segment.get('extrude', True)) if isinstance(segment, dict) else True


def _segment_line_view(df, segment):
    """Return a line dataframe for either legacy int segments, dict segments, or custom travel segments."""
    import pandas as pd
    if isinstance(segment, dict) and segment.get('custom_points') is not None:
        pts = np.array(segment.get('custom_points') or [], dtype=float)
        if pts.size == 0:
            return pd.DataFrame(columns=['x','y','z','E_cumulative'])
        out = pd.DataFrame(pts, columns=['x','y','z'])
        out['E_cumulative'] = 0.0
        return out

    line_id = _segment_line_id(segment)
    reverse = bool(segment.get('reverse', False)) if isinstance(segment, dict) else False
    line = df[df['line_id'] == line_id].copy()
    if reverse:
        line = line.iloc[::-1].copy()
    return line


def gcode_writer(df, settings, line_order_grouped):
    # Keep e_calculator for compatibility / inspection, but actual G-code extrusion
    # is generated from an explicit monotonic accumulator so reversed segments never
    # cause absolute-E to go backwards.
    df_print = e_calculator(df, settings, line_order_grouped)

    # Offset path so head doesn't crash into print bed
    df_print['z'] = df_print['z'] + settings['z_min']
    df_print = df_print.round(3)

    shared_nodes = settings.get('shared_nodes', [])
    node_tol = float(settings.get('node_tol', 0.25))
    visited_nodes = set()

    # Monotonic absolute extruder accumulator
    settings['_current_E'] = 0.0

    preamble(settings)
    cleaning(settings)

    for path in line_order_grouped:
        if not path:
            continue

        position_printhead(df_print, path[0], settings)

        for segment in path:
            extrude_this_segment = _segment_extrudes(segment)
            line = _segment_line_view(df_print, segment)
            if line.empty:
                continue

            p_start = (float(line.iloc[0]['x']), float(line.iloc[0]['y']), float(line.iloc[0]['z']))
            p_end = (float(line.iloc[-1]['x']), float(line.iloc[-1]['y']), float(line.iloc[-1]['z']))

            start_node = _match_shared_node(p_start, shared_nodes, node_tol)
            end_node = _match_shared_node(p_end, shared_nodes, node_tol)

            start_revisit = (
                (start_node is not None)
                and (start_node in visited_nodes)
                and bool(settings.get('slow_revisit_start', True))
            )
            end_revisit = (end_node is not None) and (end_node in visited_nodes)

            print_line(
                df_print,
                segment,
                settings,
                slow_near_start=start_revisit,
                slow_near_end=end_revisit,
                extrude=extrude_this_segment,
            )

            # Only real printed edges should mark a node as visited.
            if extrude_this_segment:
                if start_node is not None:
                    visited_nodes.add(start_node)
                if end_node is not None:
                    visited_nodes.add(end_node)

        raise_printhead(df_print, settings)

    postamble(settings)
    return df_print


def e_calculator(df, settings, line_order_grouped):
    d = settings['d']
    alpha = 0.7034

    df = df.copy()

    if 'distance_from_last' not in df.columns:
        df['distance_from_last'] = np.sqrt(
            (df['x'].diff().fillna(0)) ** 2 +
            (df['y'].diff().fillna(0)) ** 2 +
            (df['z'].diff().fillna(0)) ** 2
        )
    df['distance_from_last'] = df['distance_from_last'].fillna(0)

    df['V_mL'] = np.pi * ((d / 2) ** 2) * df['distance_from_last']
    df['E'] = np.pi * (1 / alpha) * df['V_mL']
    df.loc[df['distance_from_last'] == 0, 'E'] = 0.0

    for path in line_order_grouped:
        if not path:
            continue
        first_id = _segment_line_id(path[0])
        idxs = df.index[df['line_id'] == first_id].tolist()
        if idxs:
            df.loc[idxs[0], 'E'] = 0.0

    df['E_cumulative'] = df['E'].cumsum()
    return df


def preamble(settings):
    preamble_out = """; Setup section
M82 ; absolute extrusion mode
G90 ; use absolute positioning
M104 S0.0 ; Set Hotend Temperature to zero
M140 S{} ; set bed temp
M190 S{} ; wait for bed temp
G28 ; home all
G92 E0.0 ; Set zero extrusion
M107 ; Fan off""".format(settings['bed_temperature'], settings['bed_temperature'])
    with open(settings['fileout'], "w") as file:
        file.write(preamble_out)


def cleaning(settings):
    cleaning_out = """\n
; Cleaning section
G1 F800 ; Set speed for cleaning
G1 X220 Y5 ; Move to front right corner ##CALIBRATE
G1 F500 ; Slow down to remove vibration
G1 Z{} ; Lower printhead to floor
G1 X180 Y5 E{} ; Move towards front left corner ##CALIBRATE
G1 Z{} ; Raise printhead
G1 X165 Y105 F2000 ; Move printhead to centre of printbed  ##CALIBRATE
G92 X0 Y0 E0 ; Set zero extrusion""".format(settings['floor'], settings['E_clean'], settings['roof'])
    with open(settings['fileout'], "a") as file:
        file.write(cleaning_out)


def postamble(settings):
    postamble_out = """\n
; End of print
M140 S0 ; Set Bed Temperature to zero
M107 ; Fan off
M140 S0 ; turn off heatbed
M107 ; turn off fan
G1 Z{} ; Raise printhead
G1 X178 Y180 F4200 ; park print head   ##CALIBRATE
G28 ; Home all
M84 ; disable motors
M82 ; absolute extrusion mode
M104 S0 ; Set Hotend Temperature to zero
; End of Gcode""".format(settings['roof'])
    with open(settings['fileout'], "a") as file:
        file.write(postamble_out)


def position_printhead(df, segment, settings):
    first_line = _segment_line_view(df, segment).iloc[0]
    positioning = """\n
; Initial positioning for new print path
G1 F800          ; Printhead speed for initial positioning
G1 X{} Y{}       ; XY-coords of first point of path
G1 Z{}           ; Z-coord of first point of path
G4 S2            ; Dwell for 2 seconds for karma / aligment
G1 F{}           ; Set printhead speed""".format(first_line['x'], first_line['y'], first_line['z'], settings['f_print'])
    with open(settings['fileout'], "a") as file:
        file.write(positioning)


def print_line(df, segment, settings, slow_near_start=False, slow_near_end=False, extrude=True):
    """Write one traversal to G-code with a monotonic absolute-E accumulator.

    Key behaviour:
    - Reversed segments are allowed geometrically, but never cause reverse extrusion.
    - Travel-only segments keep E constant.
    - Extruding segments increase E based on the actual oriented path length.
    """
    line = _segment_line_view(df, segment)
    if line.empty:
        return

    node_slow_dist = float(settings.get('node_slow_dist_mm', 0.0))
    f_print = float(settings.get('f_print', 300))
    f_node = float(settings.get('f_node', f_print))
    d = float(settings.get('d', 0.8))
    alpha = 0.7034

    xyz = line[['x', 'y', 'z']].astype(float).to_numpy()
    s_from_start, s_to_end = _line_arclengths(xyz)

    # Local oriented step lengths
    if len(xyz) >= 2:
        diffs = np.diff(xyz, axis=0)
        step_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        step_lengths = np.concatenate([[0.0], step_lengths])
    else:
        step_lengths = np.array([0.0])

    line_id = _segment_line_id(segment) if not (isinstance(segment, dict) and segment.get('custom_points') is not None) else -1
    start_line = f"\n\n; Start of line number: {line_id} | extrude={int(bool(extrude))}\n"
    gcode_lines = [start_line]

    current_F = None
    current_E = float(settings.get('_current_E', 0.0))

    rows = list(line.itertuples(index=False))
    for i, row in enumerate(rows):
        slow_here = False
        if node_slow_dist > 0:
            if slow_near_start and s_from_start[i] <= node_slow_dist:
                slow_here = True
            if slow_near_end and s_to_end[i] <= node_slow_dist:
                slow_here = True

        target_F = f_node if slow_here else f_print

        if extrude:
            v_ml = np.pi * ((d / 2.0) ** 2) * float(step_lengths[i])
            dE = np.pi * (1.0 / alpha) * v_ml
            if i == 0:
                dE = 0.0
            current_E += float(dE)

        e_value = current_E

        if current_F is None or abs(target_F - current_F) > 1e-9:
            gcode_lines.append(
                f"G1 X{row.x} Y{row.y} Z{row.z} E{e_value:.5f} F{int(target_F)}"
            )
            current_F = target_F
        else:
            gcode_lines.append(
                f"G1 X{row.x} Y{row.y} Z{row.z} E{e_value:.5f}"
            )

    settings['_current_E'] = current_E

    with open(settings['fileout'], "a") as file:
        file.write("\n".join(gcode_lines))


def raise_printhead(df, settings):
    raise_printhead_out = """\n
; Raise printhead
G1 Z{} F200""".format(5 + df['z'].max())
    with open(settings['fileout'], "a") as file:
        file.write(raise_printhead_out)
