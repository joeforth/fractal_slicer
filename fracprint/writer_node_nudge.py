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
        d2 = dx*dx + dy*dy + dz*dz
        if d2 <= tol2 and (best_d2 is None or d2 < best_d2):
            best = (round(nx, 3), round(ny, 3), round(nz, 3))  # canonical
            best_d2 = d2
    return best

def _line_arclengths(line_xyz):
    """Given Nx3 array, return s_from_start and s_to_end arrays."""
    diffs = np.diff(line_xyz, axis=0)
    seg = np.sqrt((diffs**2).sum(axis=1))
    s_from_start = np.concatenate([[0.0], np.cumsum(seg)])
    s_to_end = s_from_start[-1] - s_from_start
    return s_from_start, s_to_end


def gcode_writer(df, settings, line_order_grouped):
    # Calculate extrusion amount between points
    df_print = e_calculator(df, settings, line_order_grouped)

    # Offset path so head doesn't crash into print bed
    df_print['z'] = df_print['z'] + settings['z_min']
    df_print = df_print.round(3)   # 3dp max

    # Shared node coordinates (junctions) passed from the processor
    shared_nodes = settings.get('shared_nodes', [])
    node_tol = float(settings.get('node_tol', 0.25))
    visited_nodes = set()  # canonical node tuples already printed

    # Write sections to file
    preamble(settings)
    cleaning(settings)

    printed_line_ids = set()
    for path in line_order_grouped:
        if not path:
            continue

        # Move to the start of this path
        position_printhead(df_print, path[0], settings)

        for line_id in path:
            if line_id in printed_line_ids:
                continue

            # Decide whether we are REVISITING a shared node at either endpoint
            line = df_print[df_print['line_id'] == line_id]
            if line.empty:
                continue

            p_start = (float(line.iloc[0]['x']), float(line.iloc[0]['y']), float(line.iloc[0]['z']))
            p_end   = (float(line.iloc[-1]['x']), float(line.iloc[-1]['y']), float(line.iloc[-1]['z']))

            start_node = _match_shared_node(p_start, shared_nodes, node_tol)
            end_node   = _match_shared_node(p_end, shared_nodes, node_tol)

            start_revisit = (start_node is not None) and (start_node in visited_nodes) and bool(settings.get('slow_revisit_start', True))
            end_revisit   = (end_node is not None) and (end_node in visited_nodes)

            print_line(df_print, line_id, settings,
                       slow_near_start=start_revisit,
                       slow_near_end=end_revisit)

            # After printing, mark any shared-node endpoints as visited (so future lines "revisit")
            if start_node is not None:
                visited_nodes.add(start_node)
            if end_node is not None:
                visited_nodes.add(end_node)

            printed_line_ids.add(line_id)

        raise_printhead(df_print, settings)

    postamble(settings)
    return df_print




def e_calculator(df, settings, line_order_grouped):
    d = settings['d']  # Nozzle diameter
    alpha = 0.7034   # Extrusion multiplier

    df = df.copy()

    # Ensure distance_from_last exists and NaNs become 0
    if 'distance_from_last' not in df.columns:
        df['distance_from_last'] = np.sqrt(
            (df['x'].diff().fillna(0))**2 +
            (df['y'].diff().fillna(0))**2 +
            (df['z'].diff().fillna(0))**2
        )
    df['distance_from_last'] = df['distance_from_last'].fillna(0)

    # DO NOT drop rows; just set E=0 for zero-distance moves
    df['V_mL'] = np.pi * ((d / 2) ** 2) * df['distance_from_last']
    df['E'] = np.pi * (1 / alpha) * df['V_mL']
    df.loc[df['distance_from_last'] == 0, 'E'] = 0.0

    # Set extrusion for the first printed point of each path to zero (if it exists)
    for path in line_order_grouped:
        if not path:
            continue
        first_id = path[0]
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
G92 X0 Y0 E0 ; Set zero extrusion""".format(settings['floor'], settings['E_clean'], settings['roof'])  ##CALIBRATE?
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


def position_printhead(df, line_id, settings):
    first_line = df[df['line_id'] == line_id].iloc[0]
    positioning = """\n
; Initial positioning for new print path
G1 F800          ; Printhead speed for initial positioning
G1 X{} Y{}       ; XY-coords of first point of path
G1 Z{}           ; Z-coord of first point of path
G4 S2            ; Dwell for 2 seconds for karma / aligment
G1 F{}           ; Set printhead speed""".format(first_line['x'], first_line['y'], first_line['z'], settings['f_print'])
    with open(settings['fileout'], "a") as file:
        file.write(positioning)


def print_line(df, line_id, settings, slow_near_start=False, slow_near_end=False):
    """Write one line_id to G-code, optionally slowing near endpoints (used for revisiting shared nodes)."""
    line = df[df['line_id'] == line_id].copy()
    if line.empty:
        return

    node_slow_dist = float(settings.get('node_slow_dist_mm', 0.0))
    f_print = float(settings.get('f_print', 300))
    f_node = float(settings.get('f_node', f_print))

    # Compute distances along the polyline for slowdown gating
    xyz = line[['x','y','z']].astype(float).to_numpy()
    s_from_start, s_to_end = _line_arclengths(xyz)

    start_line = f"\n\n; Start of line number: {line_id}\n"
    gcode_lines = [start_line]

    current_F = None
    for i, row in enumerate(line.itertuples(index=False)):
        slow_here = False
        if node_slow_dist > 0:
            if slow_near_start and s_from_start[i] <= node_slow_dist:
                slow_here = True
            if slow_near_end and s_to_end[i] <= node_slow_dist:
                slow_here = True

        target_F = f_node if slow_here else f_print
        # Emit F only when it changes (or for the first move in the line)
        if current_F is None or abs(target_F - current_F) > 1e-9:
            gcode_lines.append(
                f"G1 X{row.x} Y{row.y} Z{row.z} E{row.E_cumulative} F{int(target_F)}"
            )
            current_F = target_F
        else:
            gcode_lines.append(
                f"G1 X{row.x} Y{row.y} Z{row.z} E{row.E_cumulative}"
            )

    with open(settings['fileout'], "a") as file:
        file.write("\n".join(gcode_lines))



def raise_printhead(df, settings):
    # Give 5 mm clearance
    raise_printhead_out = """\n
; Raise printhead
G1 Z{} F200""".format(5+df['z'].max())
    with open(settings['fileout'], "a") as file:
        file.write(raise_printhead_out)

