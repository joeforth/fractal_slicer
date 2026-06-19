## Motivation

In embedded bioprinting, failure often has less to do with the shape itself and more to do with **the order in which the shape is printed**.

Common problems include:
- Smearing at intersections  
- Pooling when neighbouring lines are printed too soon  
- Stringing during long travel moves  
- Poor junction quality at shallow-angle nodes  

These issues are not always visible from the CAD model alone. They emerge from the interaction between **geometry**, **printing order**, and **material behaviour**.

Fractal Slicer focuses specifically on **print ordering** as a way to improve reliability.



