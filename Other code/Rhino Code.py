#Code for Rhino 
#This code acts to convert lines in Rhino into 0.1 mm spaced dots (future coordinates) and assigns lines unique colour IDs to lines so they can be individually identified by the algorithm. These changes are then output as a text file.
 
import rhinoscriptsyntax as rs
import random
import os
 
spacing = 0.1  # mm
output_path = r'C:\path\to\your\output_folder\filename.txt'  # <-- update this to your own file location 
curves = rs.ObjectsByType(4)  # 4 = curve objects
 
if not curves:
    print("No curves found in the document.")
else:
    f = open(output_path, 'w')
    for curve_id in curves:
        length = rs.CurveLength(curve_id)
        if not length or length <= 0:
            continue
 
        n_segments = max(1, int(round(length / spacing)))
        params = rs.DivideCurve(curve_id, n_segments, create_points=False, return_points=True)
 
        if not params:
            continue
 
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

#The following code was utilised to efficiently unlock all layers of the Rhino doc:
import rhinoscriptsyntax as rs
 
def unlock_all_layers():
    rs.EnableRedraw(False)
    layers = rs.LayerNames()
    if layers:
        for layer in layers:
            rs.LayerLocked(layer, False)
    rs.EnableRedraw(True)
 
unlock_all_layers()
