
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