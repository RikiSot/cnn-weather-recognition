# CNN Recognition Weather APP

Image recognition app for weather phenomenon images. The model is trained on this [Kaggle's dataset](https://www.kaggle.com/datasets/fceb22ab5e1d5288200c0f3016ccd626276983ca1fe8705ae2c32f7064d719de).

I tested different models and found that the best model for this application is the [ResNet50](https://keras.io/applications/#resnet50).
Model training and selection can be found in the [JupyterNotebook](cnn_transfer_learning.ipynb).

## Installation

Install the dependencies with the following command:

```
pip install -r requirements.txt
```
or build Docker image from Dockerfile with the following command:

```
docker build -t cnn_recognition_weather .
```

## Usage

First, you need to start the API service.

### From the command line

Run the API service with the following command:
```
python api.py
```
and open the browser at http://localhost:5000/. You should see a message that confirms the service is running.

### From docker image
Run the recently built image with the following command:

```
docker run -p 5000:5000 --rm cnn_recognition_weather
```
and open the browser at http://localhost:5000/. Port 5000 is the port that the service is listening on, but you can change it to any port you want.

### API endpoints
To see documentation for the API endpoints, open the browser at http://localhost:5000/docs

At this moment, there is only one endpoint at /predict. It takes an image url as a parameter
and returns the weather phenomenon label and prediction score. For example,

```
localhost:5000/predict?image_url=https://media.istockphoto.com/photos/water-is-life-picture-id165981483?k=20&m=165981483&s=612x612&w=0&h=4IXRF_i9xnCwpVeqZuqYYANZ5TfO6alaVQvZ1hZFIMA=
```
Should return the following JSON:

```
{
  "prediction": "dew",
  "probability": 100
}
```

## Model
A bunch of convolutional neural networks pretrained on imagenet were trained on this dataset. The process is described in the [JupyterNotebook](cnn_transfer_learning.ipynb).
At the end, the best model was selected and saved as a Keras model. Only this model was uploaded to this repository and Heroku due to
space limitations. 

You can train and generate new models on your own using the [JupyterNotebook](cnn_transfer_learning.ipynb).

## Test the app
The whole app is deployed on Heroku. You can test it on the following link: https://weather-recognition.herokuapp.com/




