# README

I don't think this module is useful in the current state, but I leave it here so that
I have a base for future projects and I can, in the future, make it work with
frameworks other than Sklearn (i.e. pytorch).
This module I think would be useful from an analysis standpoint,
especially when having to compare multiple models. For training, not
so much, I think there are better implementations of Pipeline out there.
The problems currently are the interfaces of the dataset and the models.
To make it work some interface, independent of any library, ideally, 
should be defined, and then the models that should be analyzed must implement it.