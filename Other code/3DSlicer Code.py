#Code for 3DSlicer
 #This code extracts the branching centreline structure from the vessel mesh in 3D Slicer, walking through the generated curve tree to collect every branch as a separate, non-overlapping segment. Each branch is resampled into points spaced 0.1 mm apart along its length. These points are then output as an OBJ file, with each branch written as a distinct named object so it can be individually identified in Rhino. 
import slicer, vtk, numpy as np, os
 
def get_unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    counter = 1
    while True:
        candidate = f"{base}_{counter}{ext}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1
 
def resample_polyline(pts, spacing):
    pts = np.array(pts)
    if len(pts) < 2:
        return pts
    diffs = np.diff(pts, axis=0)
    seglens = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate(([0], np.cumsum(seglens)))
    total = cumlen[-1]
    if total == 0:
        return pts[:1]
    sample_dists = list(np.arange(0, total, spacing))
    if sample_dists[-1] != total:
        sample_dists.append(total)
    resampled = []
    for d in sample_dists:
        idx = max(0, min(np.searchsorted(cumlen, d) - 1, len(pts) - 2))
        seglen = seglens[idx] if seglens[idx] > 0 else 1
        t = (d - cumlen[idx]) / seglen
        resampled.append(pts[idx] + t * (pts[idx + 1] - pts[idx]))
    return np.array(resampled)
 
centerlineCurveNode = slicer.util.getNode('Centerline curve_2 (0)')  # <-- root node name, not "(0)"
 
shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
curveItem = shNode.GetItemByDataNode(centerlineCurveNode)
childIds = vtk.vtkIdList()
shNode.GetItemChildren(curveItem, childIds, True)
 
branchNodes = []
for i in range(childIds.GetNumberOfIds()):
    node = shNode.GetItemDataNode(childIds.GetId(i))
    if node and node.IsA("vtkMRMLMarkupsCurveNode"):
        branchNodes.append(node)
 
if centerlineCurveNode.GetNumberOfControlPoints() > 0:
    branchNodes.insert(0, centerlineCurveNode)
 
spacing = 0.1  # mm
output_path = r'C:\path\to\your\output_folder\filename.obj'  # <-- update this to your own file 

locationoutput_path = get_unique_path(output_path)
 
with open(output_path, 'w') as f:
    vertex_offset = 0
    branch_id = 0
    for node in branchNodes:
        n = node.GetNumberOfControlPoints()
        if n < 2:
            continue
        raw_pts = []
        for i in range(n):
            pos = [0.0, 0.0, 0.0]
            node.GetNthControlPointPositionWorld(i, pos)
            raw_pts.append(pos)
 
        resampled = resample_polyline(raw_pts, spacing)
 
        safe_name = node.GetName().replace(' ', '_')
        f.write(f"o {safe_name}\n")
        for p in resampled:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        idx = " ".join(str(vertex_offset + k + 1) for k in range(len(resampled)))
        f.write(f"l {idx}\n")
        vertex_offset += len(resampled)
 
        branch_id += 1
 
print(f"Wrote {branch_id} branches, spaced {spacing}mm apart, to {output_path}")
