# Fractal Slicer  
### Path planning for embedded 3D bioprinting of vascular networks

Fractal Slicer is a Python-based tool for generating **continuous, collision aware print paths** for complex, branching vascular geometries.

It is designed for **embedded 3D bioprinting**, where soft inks are printed inside a support bath and print reliability depends heavily on the **order in which paths are printed**.

This project reproduces and extends the **MARMOT algorithm** developed by Ben Woodland, with the aim of making print-path generation **more accessible, more reliable, and more physics-aware**.



## Motivation

In embedded bioprinting, failure often has less to do with the shape itself and more to do with **the order in which the shape is printed**.

Common problems include:
- Smearing at intersections  
- Pooling when neighbouring lines are printed too soon  
- Stringing during long travel moves  
- Poor junction quality at shallow-angle nodes  

These issues are not always visible from the CAD model alone. They emerge from the interaction between **geometry**, **printing order**, and **material behaviour**.

Fractal Slicer focuses specifically on **print ordering** as a way to improve reliability.



## What Fractal Slicer does

When printing vascular networks, lines intersect, overlap, and sit above one another. Printing in the wrong order can cause smearing, pooling, or collapse.

Fractal Slicer addresses this by analysing the geometry and determining **which lines are safe to print at each stage**.

The algorithm works as follows:

1. Import a 3D vascular network (`.txt` file) exported from Rhino  
2. Identify junctions (“nodes”) by clustering line endpoints  
3. Decide which lines are safe to print at the current stage  
4. Group those lines into continuous paths using an Euler-style traversal  
5. Output G-code and visualise the resulting print order  

The key idea is that not all lines are safe to print at the same time, even if they exist in the same geometry.


## Algorithm overview

The slicer follows the loop below:

### ➤ Preprocess geometry  
Lines are split where necessary, large jumps are removed, and each curve is labelled with a `line_id`.

### ➤ Find nodes  
Line endpoints are clustered to identify junctions (nodes) within the network.

### ➤ Validate lines  
The algorithm decides which lines are safe to print next by checking:
- XY intersections  
- Relative Z-height  
- Shallow-angle junctions  

### ➤ Optimise the path  
A modified **Euler / Fleury** algorithm is used to group valid lines into continuous print paths.

### ➤ Write G-code  
The ordered paths are converted into printer instructions.

## Addressing Different Processor Versions

The repository contains several variations of processor/writer developed during experimentation with different path planning strategies.

These versions were intentionally kept separate rather than merged into a single processor, as each explores different approaches to improving print fidelity. The implementations were independently applied and evaluated. Keeping the versions separate also allows future users to access all implementations directly, choose what is most suitable for their application, and combine or discard features if they wish through further experimentation.

`version_debugged` should be treated as the main development version, as it was developed as a consequence of encountered errors and supports processing of a wider range of geometry complexity.

| Version | Purpose |
|---|---|
| `debugged` | Implementation of the MARMOT-style workflow using graph validation and modified Euler-style traversal for continuous print-path grouping |
| `node_nudge` | Implements local node overshoot corrections by displacing points near junctions towards shared node coordinates to improve physical filament intersection and node connectivity |
| `retract` | Implements reverse extrusion to reduce stringing and unintended material deposition during nozzle lifts |
| `chinese_postman` | Implements an open Chinese Postman formulation with shortest-path edge augmentation and non-extruding traversal along existing vessel geometry to reduce nozzle retractions and maintain continuous traversal of non-Eulerian vascular networks |

Some versions also use different internal path representations.

For example:
- older versions store paths as lists of `line_id`s
- newer versions such as version_retract may store paths as segment dictionaries containing:

```python
{"line_id": 4, "reverse": True, "extrude": False}
```

As a result, processors and writers from different versions may not always work together without modification.

## Visualisation of print paths

The image below shows a typical example of the type of geometry Fractal Slicer operates on.

<img width="418" height="407" alt="Screenshot 2025-12-20 172407" src="https://github.com/user-attachments/assets/69d28d14-23f9-46dc-b894-24f31510be52" />


Each coloured curve represents a printable line, and the numbered labels show the **order in which the algorithm chooses to print them**.

You can already see several geometric limitations being addressed:
- Some lines sit above others in the Z plane  
- Multiple lines meet at shallow angles  
- Printing in any order would cause the nozzle to pass through freshly deposited material  

**This is where ordering matters**.



## Validation: deciding what is safe to print

Before any path optimisation happens, the slicer applies a **validator** function.

Ultimately, the validator determines **which lines should be delayed or prioritised** in the print order.

A line may be rejected (temporarily) if:
- Its XY projection intersects another line that sits lower in Z  
- Two lines meet at a shallow angle and printing the higher one first would disrupt the junction  
- Printing it now would force the nozzle to collide with or smear existing material  

Only lines that pass these checks are allowed into the current print group.

This closely replicates the logic described in **Ben Woodland’s MARMOT algorithm**.


## Path Grouping and Ordering

Once a set of valid lines has been identified, the slicer attempts to print them **as continuously as possible**, equating to minimal nozzle retractions.

This is done using a modified Euler / Fleury path:
- Lines are treated as edges in a graph  
- Nodes represent clustered endpoints  
- The algorithm avoids breaking connectivity unless necessary  

The resulting grouped paths aim to:
- Minimise unnecessary nozzle lifts  
- Reduce long travel moves  
- Maintain continuity through junctions  

At this stage, the goal is reliability and interpretability, not the shortest possible travel distance.



## Animation Assist

The slicer includes an animation tool that shows the **print head moving through the generated path in 3D**:

![Print Path Video](https://github.com/user-attachments/assets/a89baacb-44d9-4d45-923a-9bcfa21936e0)


It is encouraged to run the notebook and watch the animation. This makes it easier to connect the theory to the actual behaviour of the algorithm.

Running the animation is often the fastest way to understand why a particular print order was chosen. Animation behaviour may differ between processor versions. For example, the Chinese Postman implementation includes non-extruding traversal, repeated edges, and directional segment traversal, which changes how nozzle motion is represented during animation.

The animation highlights:
- When the nozzle revisits a node  
- Where jumps occur between paths  
- How ordering changes with geometry  



## How to run the slicer

### 1. Clone the repository and install the required python modules

clone https://github.com/joeforth/fractal_slicer.git

**Required modules:**

numpy

pandas

scipy

matplotlib

shapely

### 2. Prepare your vascular network geometry
Example vascular network created in Rhino

<img width="900" height="846" alt="Screenshot 2025-12-20 183119" src="https://github.com/user-attachments/assets/cacd1105-f14a-45fb-b733-3da9ef85f9e8" />


- Export your vascular network from Rhino as a .txt file
- Place the file inside the Rhino/ folder

There is already an example network included for a preliminary run: newvesselbranched.txt

### 3. Run the notebook

Open the Jupyter notebook: fractal_slicer_v0-2.ipynb

The notebook acts as a simple front-end to the slicer and automatically imports the required modules:
```python
from fracprint import processor, writer
```
### 4. Set print parameters

The slicer relies on correct manual input of printer-specific parameters, particularly `floor` and `z_min`.
These must be determined for the printer in use to ensure safe operation and avoid nozzle crashes.

```python
settings = processor.build_settings(
    filedir='./Rhino/',
    filename='newvesselbranched.txt',
    fileout='output.gcode',
    d=0.8,
    x_offset=0,
    y_offset=0,
    bed_temperature=0,
    floor=-27.5,
    z_min=-27.4,
    roof=0,
    f_print=200,
    E_clean=10
)
```

### 5. Run the slicer

This will:

- preprocess the geometry
- determine a safe print order
- group paths into continuous print segments
- generate a G-code file
- produce visualisations (including an animation)

## About _pycache_
The _pycache_ folder is automatically created by Python to store compiled versions of files for faster loading. It is not part of the algorithm and can be ignored.
