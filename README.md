# Heat Transfer Visualizer #

<h3>This is a visualization of heat equation based on a heat equation differential equation:</h3>

```math

 \frac{\partial T}{\partial t} = \alpha (\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2})

```

This formula can be rewritten to calculate T(x,y,t) based on the temperature at the previous interval. The formula can be visualized as a stencil.

![image](https://github.com/KarolGutkowski/HeatTransferVisualizer/assets/90787864/572484ae-d390-4b36-9f42-dd332054dff1)

The blue dot signifies T(x,y,t), whereas the red dots signify values of T at the previous time stamp on which T at the current time step depends.

Stencil being one of the very popular parallel patterns, was a topic of Chapter 8 in "Programming Massively Parallel Processors" by David Kirk and Wen-mei Hwu 
which inspired me to use this visualization and apply the suggested optimization to parallel stencil.


https://github.com/KarolGutkowski/HeatTransferVisualizer/assets/90787864/4b650d5d-8ecb-48b2-8f41-3db8b6b1d786



