
def wkt_preprocess_old(data, filetype):
    # Split wkt file into elements, return x and y coordinates 
    # Each element corresponds to a different element of the wkt
    x, y, z = [], [], []
    for j in range(0, len(data)):
        data_element = data[j]
        form = wkt_splitter(data_element)
        print(form)
        coords = coordinater_wkt(form)
        if len(coords[0]) == 0:
            print('Ignoring point')
        if len(coords[0]) > 0:
            x.append(coords[0])
            y.append(coords[1])
    return x, y, z

def coordinater_wkt_old(string_in, idx):
    # Takes a set of coordinates in string form, where each coordinate is separated by a space
    # Rounds it to 3 decimal place, and returns it as a list of floats
    # Input - string_in - list of strings, each of which is a set of coordinates
    # Output - [x, y, z, e_id] - list of floats where e_id is element ID
    x, y, z, e_id = [], [], [], []
    # Now run through a make x- and y- coordinates
    for i in range(0, len(string_in)):
        fragment = string_in[i].split()
        x.append(round(float(fragment[0]), 3))
        y.append(round(float(fragment[1]), 3))
        z.append(0.0)
        e_id.append(idx)

    return [x, y, z, e_id]


    # leftover from when we picked first node based on connectivity
# sorted_connectivity = dict(sorted(connectivity_dict.items(), key=lambda x: x[1]))
# next_node = next(iter(sorted_connectivity))
# next_line = cluster_dict[next_node][0]

# next_line = df[df['z'] == df['z'].min()]['line_id'].values[0]

# lowest = connected_lines[0]
# lowest_z = df[df['line_id'] == lowest]['z'].min()
# for line in connected_lines:
#     z_min = df[df['line_id'] == line]['z'].min()
#     if z_min < lowest_z:
#         lowest = line
#         lowest_z = z_min

# connected_lines = [x for x in connected_lines if x in unprinted_lines]
# If there are connected lines remaining, now print the line with the lowest z value


## Leftover code from Eulerficator
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
while len(line_order) < len(lines):
    print('line_order:', line_order)
    print('lines:', lines)
    # Sort lines by height order - bottom up
    min_z = df.groupby('line_id')['z'].min()
    heightsorted_line_ids = min_z.sort_values().index.tolist()
    unprinted_lines = [n for n in heightsorted_line_ids if n not in line_order]
    print('unprinted_lines:', unprinted_lines)

    # Pick the unprinted line with the lowest z-value to start with
    next_line = unprinted_lines[0]
    line_order.append(next_line)
    unprinted_lines = unprinted_lines[1:] # Remove the printed line from the list of remaining lines

    # Calculate which node you're at - it's the other node to which next_line is connected
    connected_nodes = terminal_points.loc[next_line]['cluster'].values

    # Pick the node with the lower z-value
    start_node = nodes.loc[connected_nodes]['z'].idxmin()
    end_node = connected_nodes[connected_nodes != start_node][0]

    connected_lines = cluster_dict[end_node]
    # Calculate unprinted connected lines in height-order
    connected_lines = [n for n in unprinted_lines if n in connected_lines]
    while len(connected_lines) > 0:
        next_line = connected_lines[0]
        line_order.append(next_line)
        unprinted_lines.remove(next_line)
        # Calculate which node you've moved to
        connected_nodes = terminal_points.loc[next_line]['cluster'].values
        start_node = end_node
        end_node = connected_nodes[connected_nodes != start_node][0]
        connected_lines = cluster_dict[end_node]

        # Remove lines that have already been printed
        connected_lines = [n for n in unprinted_lines if n in connected_lines]

#
line1 = line_order_grouped[0][0]

start1 = df_write[df_write['line_id'] == line1].iloc[0][['x', 'y', 'z']].values 
node1 = nodes.loc[line1][['x', 'y', 'z']].values

if not np.array_equal(start1, node1):
    print('Reversing line ')
    df_write[df_write['line_id'] == line1] = df_write[df_write['line_id'] == line1].iloc[::-1]

df_write
# line2 = line_order_grouped[0][1]
# start2 = df_write[df_write['line_id'] == line2].iloc[0][['x', 'y', 'z']].values
# node2 = nodes.loc[1][['x', 'y', 'z']].values

# if not np.array_equal(start2, node2):
#     df_write[df_write['line_id'] == line1] = df_write[df_write['line_id'] == line1].iloc[::-1]

# end1 = df[df['line_id'] == line1].tail(1)[['x', 'y', 'z']].values
# end2 = df[df['line_id'] == line2].tail(1)[['x', 'y', 'z']].values
# start1 = df[df['line_id'] == line1].head(1)[['x', 'y', 'z']].values
# start2 = df[df['line_id'] == line2].head(1)[['x', 'y', 'z']].values


# # Something weird happening with how we rearrange the start/end of lines to match the nodes

# Make lines run in node order.
df = df.set_index('line_id').loc[line_order].reset_index()   # Reorder the dataframe based on the line order

for idx_path, path in enumerate(line_order_grouped):
    for idx_line, line in enumerate(path):
        start_node = node_order_grouped[idx_path][idx_line]
        line_start = df[df['line_id'] == line].iloc[0][['x', 'y', 'z']].values 
        node_loc = nodes.loc[start_node][['x', 'y', 'z']].values

        if not np.array_equal(line_start, node_loc):
            print('Reversing line ', line)
            # df[df['line_id'] == line] = df[df['line_id'] == line].iloc[::-1]
            new_line = df[df['line_id'] == line].iloc[::-1]
            df = df[df['line_id'] != line]
            df = pd.concat([df, new_line])

df = df.set_index('line_id').loc[line_order].reset_index()   # Reorder the dataframe based on the line order
df = distance_calculator(df)


def gcode(data, line_order):


    # Write file preamble:
    target.write("""
    M82 ; absolute extrusion mode
    G90 ; use absolute positioning
    M104 S0.0 ; Set Hotend Temperature to zero
    M140 S0.0 ; set bed temp
    M190 S0.0 ; wait for bed temp
    G28 ; home all
    G92 E0.0 ; Set zero extrusion
    M107 ; Fan off

    G1 X97.5 Y147 F2000 ; Move printhead to centre of printbed
    G92 X0 Y0 E0 ; Set zero extrusion

    """)


def gcode_old(x_all_shift, y_all_shift, res, inlet_d, filename, v, d, exp, dist, floor, roof, preex, init, term, retract, alpha, extrusion_on):
    # Varying flow rate as a function of distance from start / end points
    # Calculate F, make some lists
    # F = (alpha / 1000.)*math.pi*(600*v)*(d/2.)**2
    F = v * 200
    # Cumulative extrusion, cumulative distance, and differential distance
    E_list, s_list, s_diff = [0], [0], []
    E_list_sep, E_diff_sep, s_list_sep, s_diff_sep = [], [], [], []

    name_out = filename[:-4]    # Remove .csv suffix
    full_out = name_out + '_dist=' + str(dist) + '_exp=' + str(exp) + '_v=' + str(v) + '_d=' + str(d) + '_preex=' + str(preex)
    if extrusion_on == True:
        target = open(full_out + '_xon.gcode', 'w')   
    else: 
        target = open(full_out + '_xoff.gcode', 'w')  

    if dist <= 0:
        print('dist should not be set to zero or a negative value!')
        print('setting dist = 0.1')
        dist = 0.1

    # Write file preamble:
    target.write("""M82 ; absolute extrusion mode
    G90 ; use absolute positioning
    M104 S0.0 ; Set Hotend Temperature to zero
    M140 S0.0 ; set bed temp
    M190 S0.0 ; wait for bed temp
    G28 ; home all
    G92 E0.0 ; Set zero extrusion
    M107 ; Fan off

    G1 X97.5 Y147 F2000 ; Move printhead to centre of printbed
    G92 X0 Y0 E0 ; Set zero extrusion

    """)

    for j in range(0, len(x_all_shift)):
        # If there's nothing to print, don't print anything
        if not len(x_all_shift[j]) == 0:
            if j == 0:
                s_list_sep.append([0])
            if j > 0:
                s_list_sep.append([s_list_sep[j-1][-1]])
                
            target.write('; Starting shape ' + str(j+1) + ' \n')

            # Initial positioning - print head speed
            target.write('G1 F800 \n')

            # Initial positioning
            target.write('G1 ' + 'X' + str(round(x_all_shift[j][0], 2)) + ' Y' + str(round(y_all_shift[j][0], 2)) + ' \n')

            # Lower print head
            target.write('G1 Z' + str(floor) + ' ; Lower printhead into geometry - floor of chip with 0.5 inch needle is -55.5 \n')

            # Pause for alignment
            target.write('G4 S2    ; Pause for alignment \n')
            
            # Print speed
            target.write('G1 F' + str(round(F, 2)) + ' \n')
            
            # Pre-extrude - inlet
            if inlet_d > 0 and j == 0 and extrusion_on == True:
                target.write('G1 E' + str(preex) + '      ; Pre-extrude of ' + str(preex) + ' - for inlet \n')
                E_list.append(E_list[-1] + float(preex))

            # Pre-extrude - outlet     
            if inlet_d > 0 and j == 1 and extrusion_on == True:
                target.write('G1 E' + str(preex) + '      ; Pre-extrude of ' + str(preex) + ' - for outlet \n')
                E_list.append(E_list[-1] + float(preex))

            # Startup extrusion
            if extrusion_on == True and j > 1:
                target.write('G1 E' + str(round(E_list[-1] + init, 4)) + ' F' + str(round(F, 2)) + '; Startup extrusion of ' + str(init) + ' - should not appear on inlet or outlet \n')
                E_list.append(E_list[-1] + init)

            # Calculate volume to extrude with a scaling to
            # E_calculator(x_in, y_in, res_in, d_in, exp_in, offset_in, alpha_in)
            # E_fac, E_diff, s_list, s_diff
            E_fac, E_diff_shape, s_list_shape, s_diff_shape = processor.E_calculator(x_all_shift[j], y_all_shift[j], res, d, exp, dist, alpha)
            E_diff_sep.append(E_diff_shape)
            s_diff_sep.append(s_diff_shape)
            # Just need to do s_list for each shape
            
            # Run through each element in the input coordinates
            for i in range(0, len(s_diff_shape)):
                # Calculate distance travelled and extruded volume
                s_list.append(s_diff_shape[i] + s_list[-1])
                s_list_sep[j].append(s_diff_shape[i] + s_list[-1])
                E_list.append(E_diff_shape[i] + E_list[-1])
                print(E_list)

                # Write coordinates - note the +1 on i as we've already written coordinate 0 at start of shape
                target.write('G1 ' + 'X' + str(round(x_all_shift[j][i+1], 2)) + ' Y' + str(round(y_all_shift[j][i+1], 2)))

                # Extrude - if extruding
                if extrusion_on == True:
                    target.write(' E' + str(round(E_list[-1], 4)))
                target.write(' \n')

            # Retract / end point extrusion
            if extrusion_on == True:
                target.write('G1 E' + str(round(E_list[-1] + term, 4)) + ' F' + str(round(F, 2)) + ' ; End-point extrusion of ' + str(term) + ' \n')
                E_list.append(E_list[-1] + term)   # Record extra amount extruded
                
            # Raise printhead between parts
            target.write('G1 F500 \n')
            target.write('G1 Z' + str(roof) + ' ; Remove printhead from cuvette \n')
            target.write('\n')
     
    # Raise printhead clear of chip
    target.write('G1 Z0 ; Remove printhead clear of chip \n')

    # End print
    target.write("""M140 S0 ; Set Bed Temperature to zero
    M107 ; Fan off
    M140 S0 ; turn off heatbed
    M107 ; turn off fan
    G1 X178 Y180 F4200 ; park print head
    G28 ; Home all
    M84 ; disable motors
    M82 ; absolute extrusion mode
    M104 S0 ; Set Hotend Temperature to zero
    ; End of Gcode
    """)

    # Calculate volume extruded and write to file
    V = 2.8*E_list[-1]   # I currently have no idea where the factor of 2.8 comes from
    target.write('; Total volume extruded = ' + str(round(V, 2)) + ' uL \n')
    target.write('; Distance moved = ' + str(round(s_list[-1], 2)) + ' mm')
    print('Total volume extruded = ' + str(round(V, 2)) + ' uL')
    print('Distance moved = ' + str(round(s_list[-1], 2)) + ' mm')

    # Visualisation of plot and extrusion rate
    # Run through each element in the input coordinates
    ls, ts = 16, 12
    fig = plt.figure(figsize=(7,10))
    ax1 = fig.add_subplot(211)
    for k in range(0, len(x_all_shift)):
        ax1.plot(s_list_sep[k][1:], E_diff_sep[k] / s_diff_sep[k], label = 'Shape ' + str(k + 1), lw=3)
    ax1.set_xlabel('Distance (mm)', size=ls)
    ax1.set_ylabel('Extrusion Rate', size=ls)
    ax1.tick_params(axis='both', labelsize=ts)

    ax2 = fig.add_subplot(212)
    for i in range(0, len(x_all_shift)):
        ax2.plot(x_all_shift[i], y_all_shift[i], lw = 4)
        
    ax2.set_aspect('equal')
    ax2.set_xlabel('x (mm)', size=ls)
    ax2.set_ylabel('y (mm)', size=ls)
    ax2.tick_params(axis='both', labelsize=ts)

    plt.savefig(name_out + '_dist=' + str(dist) + '_exp=' + str(exp) + '_v=' + str(v) + '_d=' + str(d) + '_preex=' + str(preex) + '.png', dpi=300, bbox_inches='tight')
    # Compile all the data down somewhere useful
    # data = pd.DataFrame(np.stack((x_list, y_list, z_list, s_list, E_list), axis=1), columns=['x', 'y', 'z', 's', 'E'])

