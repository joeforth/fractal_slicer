## How to run the slicer

### 1. Clone the repository and install the required python modules

clone https://github.com/joeforth/fractal_slicer.git

All required pachages and versions are noted in requirements.txt and can be installed via %pip install -r requirements.txt (included in fractal_slicer_debugged) 

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
