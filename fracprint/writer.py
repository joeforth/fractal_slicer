import numpy as np
from fracprint import processor

def e_calculator(df, line_order_grouped, d):
    alpha = 0.7034   # Extrusion multiplier
    # Start of print code
    df = df[df['distance_from_last'] != 0.]
    df = df.fillna(0)
    df['V_mL'] = np.pi*((d/2)**2)*df['distance_from_last']
    df['E'] = np.pi*(1/alpha)*df['V_mL']  # Amount to extrude

    # Set the extrusion amount for the first line of each path to zero
    for path in line_order_grouped:
        index_to_update = df[df['line_id'] == path[0]].index[0]  # Find the index of the first match
        df.loc[index_to_update, 'E'] = 0  # Update the value at that index

    df['E_cumulative'] = df['E'].cumsum()
    return df


def preamble():
    preamble_out = """; Setup section
M82 ; absolute extrusion mode
G90 ; use absolute positioning
M104 S0.0 ; Set Hotend Temperature to zero
M140 S0.0 ; set bed temp
M190 S0.0 ; wait for bed temp
G28 ; home all
G92 E0.0 ; Set zero extrusion
M107 ; Fan off

G1 X97.5 Y147 F2000 ; Move printhead to centre of printbed
G92 X0 Y0 E0 ; Set zero extrusion"""
    with open("gcode.txt", "w") as file:
        file.write(preamble_out)


def postamble():
    postamble_out = """\n
; End of print
M140 S0 ; Set Bed Temperature to zero
M107 ; Fan off
M140 S0 ; turn off heatbed
M107 ; turn off fan
G1 X178 Y180 F4200 ; park print head
G28 ; Home all
M84 ; disable motors
M82 ; absolute extrusion mode
M104 S0 ; Set Hotend Temperature to zero
; End of Gcode"""
    with open("gcode.txt", "a") as file:
        file.write(postamble_out)


def position_printhead(df, line_id, f_print):
    first_line = df[df['line_id'] == line_id].iloc[0]
    positioning = """\n
; Initial positioning for new print path
G1 F800          ; Printhead speed for initial positioning
G1 X{} Y{}       ; XY-coords of first point of path
G1 Z{}           ; Z-coord of first point of path
G4 S2            ; Dwell for 2 seconds for karma / aligment
G1 F{}           ; Set printhead speed""".format(first_line['x'], first_line['y'], first_line['z'], f_print)
    with open("gcode.txt", "a") as file:
        file.write(positioning)


def print_line(df, line_id):
    line = df[df['line_id'] == line_id]
    start_line = "\n\n; Start of line number: " + str(line_id) + "\n"
    gcode_output = "\n".join(
        "G1 X" + line['x'].astype(str) + " Y" + line['y'].astype(str) + " Z" + line['z'].astype(str) + " E" + line['E_cumulative'].astype(str))
    with open("gcode.txt", "a") as file:
        file.write(start_line)
        file.write(gcode_output)


def raise_printhead(df):
    # Give 5 mm clearance
    raise_printhead_out = """\n
; Raise printhead
G1 Z{} F200""".format(5+df['z'].max())
    with open("gcode.txt", "a") as file:
        file.write(raise_printhead_out)


def gcode_writer(df, line_order_grouped, f_print, d, floor):
    # Calculate extrusion amount between points
    df_print = e_calculator(df, line_order_grouped, d)
    df_print['z'] = df_print['z'] + floor
    df_print = df_print.round(3)
    # Write sections to file
    preamble()
    for path in line_order_grouped:
        position_printhead(df_print, path[0], f_print)
        for line_id in path:
            print_line(df_print, line_id)
        raise_printhead(df_print)
    postamble()

    return df_print