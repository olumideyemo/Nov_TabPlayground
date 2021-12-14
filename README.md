# November 2021 Tabular Playground Series

This repository contains the code for submissions to train the model presented in the [Kaggle Competition](https://www.kaggle.com/c/tabular-playground-series-nov-2021)

v011**[submission v011](https://www.kaggle.com/olumoni/nov-tabplayground?scriptVersionId=79984986)**.
v015**[submission v015](https://www.kaggle.com/olumoni/nov-tabplayground/notebook)**.

It is a presentation repository for my submissions to the Kaggle competition. As such, you will find many scripts, classes, blocks and options which we actively use for our own development purposes but are not directly relevant to reproduce results or use pretrained weights.

![Lasso](images/LassoRegression_plot.png)

## Using pre-trained weights

In the competition, we present ClimateGAN as a solution to produce images of floods. It can actually do **more**: 

* reusing the segmentation map, we are able to isolate the sky, turn it red and in a few more steps create an image resembling the consequences of a wildfire on a neighboring area, similarly to the [holdout for websites](https://www.google.com).
* reusing the depth map, we can simulate the consequences of a smog event on an image, scaling the intensity of the filter by the distance of an object to the camera, as per [place holder HazeRD](http://www.google.com)

![image of wildfire processing](images/wildfire.png)
![image of smog processing](images/smog.png)

In this section we'll explain how to produce the `Painted Input` along with the Smog and Wildfire outputs of a pre-trained ClimateGAN model.

### Installation

This repository and associated model have been developed using Python 3.8.2 and **Pytorch 1.7.0**.

```bash
$ git clone git@github.com:cc-ai/climategan.git
$ cd climategan
$ pip install -r requirements-3.8.2.txt # or `requirements-any.txt` for other Python versions (not tested but expected to be fine)
```

Our pipeline uses [comet.ml](https://comet.ml) to log images. You don't *have* to use their services but we recommend you do as images can be uploaded on your workspace instead of being written to disk.

If you want to use Comet, make sure you have the [appropriate configuration in place (API key and workspace at least)](https://www.comet.ml/docs/python-sdk/advanced/#non-interactive-setup)
