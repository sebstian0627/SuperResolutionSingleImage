# Super Resolution using Single Image

This is my attempt at implementing "Super Resolution from a single Image" by Glasner et. al. 

Image super-resolution is the task of obtaining a high-resolution (HR) image of a scene given low-resolution (LR) image(s) of the scene. An image may have a “lower resolution” due to a smaller spatial resolution (i.e. size) or due to a result of degradation (such as blurring). The basic idea behind SR is to combine the non-redundant information contained in multiple low-resolution frames to generate a high-resolution image. Single image super-resolution is heavily ill-posed since multiple HR patches could correspond to the same LR image patch.

Looking at the limitations of both these family of methods, Glasner et. al. proposed a single unified approach which combines the classical SR constraints with the example-based constraints, while exploiting (for each pixel) patch redundancies across all image scales and at varying scale gaps, thus obtaining adaptive SR with as little as a single low-resolution image. The approach is based on an observation that patches in the same image tend to recur many times inside the image redundantly, both within same scale as well as across different scales.
