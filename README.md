Detecting Viewer-Perceived Intended Vector Sketch Connectivity
==============================================================

<strong>Jerry Yin<sup>\*,1</sup>, Chenxi Liu<sup>\*,1</sup>, Rebecca Lin<sup>1</sup>, Nicholas Vining<sup>1,2</sup>, Helge Rhodin<sup>1</sup>, Alla Sheffer<sup>1</sup></strong>

<small><sup>\*</sup>joint first authors, <sup>1</sup>University of British Columbia, <sup>2</sup>NVIDIA</small>

<img src="https://www.cs.ubc.ca/labs/imager/tr/2022/SketchConnectivity/teaser.svg" />

This repository contains the source code for the paper
[Detecting Viewer-Perceived Intended Vector Sketch Connectivity](https://www.cs.ubc.ca/labs/imager/tr/2022/SketchConnectivity/).

 - `src` contains the source code for *libsketching*, a C++ library containing core routines for working with 2D line drawings.
 - `python` contains Python bindings for libsketching, which produces a `_sketching` module which can be imported from Python.
 - `tools` contains Python scripts to run our method, visualize its results, and train our classifier (a pre-trained model is also provided).

The data can be downloaded from
[SketchConnectivityData](https://www.cs.ubc.ca/labs/imager/tr/2022/SketchConnectivity/SketchConnectivityData.zip).
The `data` folder contains the pre-processed line drawings and annotations used to train the model, as well as the drawings used to generate the results in the paper.
Put `data` under the repository root for the following instructions.


Building
--------

The `3rdparty` folder contains some required submodules, so make sure this repository is cloned with `--recursive` specified.
If you have already cloned this repo non-recursively, you can run

    git submodule update --init --recursive

to pull all of the third-party submodules.

In addition, this project depends on Gurobi (which needs to be licensed), as well as Eigen, fmt, spdlog, zlib, Cairo, and GLM (not bundled, but these should be available from most package managers such as Conda, Homebrew, or vcpkg).

To build libsketching and its Python module with debug symbols and optimizations on, run

    mkdir build && cd build
    cmake ..
    cmake --build . --config RelWithDebInfo

This will produce a static library and a Python module `_sketching` with extension `pyd` on Windows or `so` on Linux.


Running our method
------------------

After building, make sure to add the directory containing the `_sketching` Python module to your `PYTHONPATH` environment variable, or else Python will not be able to `import _sketching`.

The results seen in our [supplementary materials](https://www.cs.ubc.ca/labs/imager/tr/2022/SketchConnectivity/supplementary.zip) can be generated with

    python3 tools/run_batch.py data/inputs_train+validate.yml

(Consult the output of `python3 tools/run_batch.py --help` for details on optional flags.)

This will create a file `compilation-YYYY-MM-DD.pdf` summarizing the results, and a `snapshot` directory containing intermediate outputs.


Training the classifiers
------------------------

We provide pre-trained models in two forms: [scikit-learn models provided as a Python pickle](python/sketching/resources/classify_junction_models-current.pickle), and an auto-generated [C++ version of the same model](src/sketching/forest.cpp).
The rest of this section is for people who wish to retrain the classifiers themselves.

Build the project and make sure to add the directory containing the `_sketching` Python module to your `PYTHONPATH` environment variable.
Then run

    python3 tools/train_junction_classifier.py data/METADATA-train.yml

to train the classifiers.
The scikit-learn models will be saved to `classify_junction_models.pickle` and a corresponding C++ file `forest.cpp` will be generated in the working directory.
(The output location may be changed using optional flags; consult the output of `train_junction_classifier.py --help` for details.)
To use this newly-trained model, replace the provided `src/sketching/forest.cpp` with your newly generated one and recompile the project.


Data
----

The `data` directory contains pre-processed inputs (line drawings) in the form of [VPaint][] VEC files, grouped based on their respective sources.
The VEC files can be visualized using `tools/vec_viewer.py`.
`data/human-annotations` contains junction annotations for drawings in the training set.

[VPaint]: https://www.vpaint.org/

Some of our VEC files were collected with time information.
We extend the format of typical strokes
```xml
<edge curve="xywdense(SAMPLING X0,Y0,WIDTH0 X1,Y1,WIDTH1 ..." />
```
to include time information as
```xml
<edge curve="xywtdense(SAMPLING X0,Y0,WIDTH0,TIME0 X1,Y1,WIDTH1,TIME1 ..." />
```
where `TIME` is a double-precision floating point value measuring the number of seconds since the Epoch when that sample was drawn.
Drawings without time information are ordinary VPaint files.


### Example of creating your own VEC file

```python
import _sketching as _s

drawing = _s.Drawing()

stroke = _s.Stroke(npoints=4, has_time=False)
stroke.x[:] = [1.0, 2.0, 3.0, 4.0]
stroke.y[:] = [5.0, 6.0, 7.0, 8.0]
stroke.width[:] = [0.5, 0.75, 1.0, 0.5]

drawing.add(stroke)

drawing.save('mydrawing.vec')
```


### Pre-processing your data

Most drawings will need to be pre-processed before they will work optimally with our method.
`tools/preprocess.py` contains an example of how one might pre-process their data—the exact stages needed depend on the characteristics of your data.


License
-------

The source code (everything under `src`, `python`, and `tools`) is licensed under [Version 2.0 of the Apache License](LICENSE).
The drawings (which need to be downloaded separately from [SketchConnectivityData](https://www.cs.ubc.ca/labs/imager/tr/2022/SketchConnectivity/SketchConnectivityData.zip)) are licensed under separate licenses.
Please refer to `data/inputs_train+validate.yml` for license information for each drawing.


BibTeX
------

```
@article{sketchconnectivity,
      title = {Detecting Viewer-Perceived Intended Vector Sketch Connectivity},
      author = {Yin, Jerry and Liu, Chenxi and Lin, Rebecca and Vining, Nicholas and Rhodin, Helge and Sheffer, Alla},
      year = 2022,
      journal = {ACM Transactions on Graphics},
      publisher = {ACM},
      address = {New York, NY, USA},
      volume = 41,
      number = 4,
      doi = {10.1145/3528223.3530097}
}
```
