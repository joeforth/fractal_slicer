
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