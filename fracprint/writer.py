import numpy as np
from fracprint import processor


def gcode_writer(df, settings, line_order_grouped):
    # Calculate extrusion amount between points
    df_print = e_calculator(df, settings, line_order_grouped)
    # Offset path so head doesn't crash into print bed
    df_print['z'] = df_print['z'] + settings['z_min']
    df_print = df_print.round(3)   # 3dp max
    # Write sections to file
    preamble(settings)
    cleaning(settings)
    for path in line_order_grouped:
        position_printhead(df_print, path[0], settings)
        for line_id in path:
            print_line(df_print, line_id, settings)
        raise_printhead(df_print, settings)
    postamble(settings)

    return df_print


def e_calculator(df, settings, line_order_grouped):
    d = settings['d']  # Nozzle diameter
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
G1 X-50 Y50 ; Move to front left corner
G1 F500 ; Slow down to remove vibration
G1 Z{} ; Lower printhead to floor
G1 X50 Y50 E{} ; Move to front right corner
G1 Z{} ; Raise printhead
G1 X97.5 Y147 F2000 ; Move printhead to centre of printbed
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
G1 X178 Y180 F4200 ; park print head
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


def print_line(df, line_id, settings):
    line = df[df['line_id'] == line_id]
    start_line = "\n\n; Start of line number: " + str(line_id) + "\n"
    gcode_output = "\n".join(
        "G1 X" + line['x'].astype(str) + " Y" + line['y'].astype(str) + " Z" + line['z'].astype(str) + " E" + line['E_cumulative'].astype(str))
    with open(settings['fileout'], "a") as file:
        file.write(start_line)
        file.write(gcode_output)


def raise_printhead(df, settings):
    # Give 5 mm clearance
    raise_printhead_out = """\n
; Raise printhead
G1 Z{} F200""".format(5+df['z'].max())
    with open(settings['fileout'], "a") as file:
        file.write(raise_printhead_out)

