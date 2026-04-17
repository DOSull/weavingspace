# Geographical tiling and weaving
This is ongoing work developing tiled geospatial data layers for symbolisation of complex multi-attribute choropleths.  

A working version is available in python using `geopandas` and `shapely`. The python source is in the `weavingspace` folder. It can be installed from PyPI using 

`python3 -m pip install weavingspace`

The API is documented [here](https://dosull.github.io/weavingspace/doc/), and there are several example notebooks that will give a good idea of how to use the code in the `examples` folder. In particular, have a look at the [Using the library](examples/using-the-library.ipynb) notebook.

There is a (always under development) web app which you can use to get a sense of the capabilities of `weavingspace` without writing any code here: [dosull.github.io/mapweaver/app/](https://dosull.github.io/mapweaver/app/).

## What does it do?
The kind of things we can make are:

![a tiled map](/presentations/NZCS-Aug-2022/slides/images/imd-escher.png)
![a weave map](/presentations/NZCS-Aug-2022/slides/images/imd-weave.png)

## Presentations
An overview of the concepts assembled from early proof-of-concept _R_ code is on [this webpage](https://dosull.github.io/weavingspace/presentations/NZCS-Nov-2021/make-weave-map.html). A similar follow-up talk is [available here](https://dosull.github.io/weavingspace/presentations/Palmerston-North-Nov-2022/slides/index.html). Slides from a more recent talk explaining the work, extended to tiled maps (of which woven maps are a special case) is available [here](https://dosull.github.io/weavingspace/presentations/Palmerston-North-Nov-2022/slides/).

## Other explanatory stuff
Any of the notebooks in the `examples` folder may be of interest. Some background material and thinking about tiling is in [these notes](https://dosull.github.io/weavingspace/notes/notes-on-tiling.html) and these reflections on the [state of the code](https://dosull.github.io/weavingspace/notes/state-of-the-code.html).
