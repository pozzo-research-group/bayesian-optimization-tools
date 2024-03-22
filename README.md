# bayesian-optimization-tools
Collection of tools to run off-the-shelf Bayesian optimization with the current experimental setup in the Pozzo group

THIS PACKAGE IS CURRENTLY UNDER CONSTRUCTION

This package is created to run a Bayesian optimization experimental campaign where the data is being collected from experimental workflows that need separate data processing with significant delays in processing and collecting data.

This package contains a `models` folder that currently has Gaussian Process models implemented based on the `botorch` packages as well as a `datatools` folder that has some helper functions to run the experimental campaigns with different characterization data.

Example scripts can be found in `examples` folder that can be run in the Python command line using `python example_file.py ITERATION`.

To install the package, simply clone this repository and install it using `pip install .`

## Notes on the package
This package name bayesian-optimization-tools (BOT) is a nod to the chapter box-of-tools from "Surely You're Joking, Mr. Feynman!" that describes the advantage of having a different box of tools than others. While there have been several Bayesian optimization packages aimed at experimental campaign runs, none of them give us control and flexibility to work with the experimental workflows we typically work with. This package is an attempt to make it easier to run optimization campaigns for the equipment we use in the Pozzo lab.