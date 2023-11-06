# Problem description

Heart failure is a common event caused by cardiovascular diseases. This project aims to predict mortality by heart failure using a supervised machine learning model.

The project offers a solution by providing a user-friendly endpoint. Users can submit a set of essential parameters, and in response, the system will provide a binary outcome, indicating the likelihood of a death event. This outcome can be either "true" or "false," signifying the presence or absence of mortality risk. The supervised machine learning model supports healthcare professionals and researchers to identify high-risk patients swiftly and accurately.

# Getting Started

You can find all the information to run/build the application on your local or Docker in this document.

## Prerequisites

- Python 3.10
- Docker
- [Heart failure dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/data) from Kaggle
- Heroku account (optional for cloud deployment)

## Preparation

In order to train the model, you need to download the heart failure clinical records dataset from [Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data/data) and locate to the root directory of the project with name `heart_failure_clinical_records_dataset.csv`. The dataset contains 299 records with 12 features and a binary label.

## Exploratory Data Analysis

The jupyter notebook file `notebook.ipynb` contains exploratory data analysis. In addition, four machine learning models namely Logistic Regression, Decision Tree, Random Forest and XGBoost models have been trained with hyper-parameter tuning. The best model with the highest AUC value has been selected for deployment.

## Model Training

- To train the model, run `python train.py` command. This will read the `heart_failure_clinical_records_dataset.csv` file and train the model with the dataset which it contains.
- This step creates `model_RF.bin` file which will be used in prediction.

## Build and run locally

- Run `pipenv shell` to create and activate a virtual environment
- `requirements.txt` will be automatically converted to `Pipfile`.
- Run `pipenv install` to install dependencies. It will also create `Pipfile.lock` file.
- Run `python predict.py` command to spin up an API endpoint to return the prediction of the given request.
- To test the endpoint, you can run below cURL command:

```shell
curl  -X POST \
  'http://localhost:5000/predict' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "age": 75,
  "anaemia": 1,
  "creatinine_phosphokinase": 81,
  "diabetes": 0,
  "ejection_fraction": 38,
  "high_blood_pressure": 1,
  "platelets": 368000,
  "serum_creatinine": 4,
  "serum_sodium": 131,
  "sex": 1,
  "smoking": 1,
  "time": 10
}'
```

- You will get the result below after running the previous command:

```shell
{
  "death_event": true
}
```

## Build and run via Docker

- Run `docker build -t heart-failure-prediction .` command in the root directory of the project to create a docker image.
- Run `docker run -e PORT=5000 -p 9696:5000 --name=predictionApp heart-failure-prediction` to run a container which serves an API endpoint which exposes container default port `5000` to local `9696` port. Now, the app which is running inside the docker container is accessible with the address `http://localhost:9696`.
- To test the endpoint, you can run below cURL command:

```shell
curl  -X POST \
  'http://localhost:9696/predict' \
  --header 'Content-Type: application/json' \
  --data-raw '{
  "age": 75,
  "anaemia": 1,
  "creatinine_phosphokinase": 81,
  "diabetes": 0,
  "ejection_fraction": 38,
  "high_blood_pressure": 1,
  "platelets": 368000,
  "serum_creatinine": 4,
  "serum_sodium": 131,
  "sex": 1,
  "smoking": 1,
  "time": 10
}'
```

- You will get the result below after running the previous command:

```shell
{
  "death_event": true
}
```

## Release & Deploy

GitHub action is used to deploy this project to Heroku. You can find the details in [actions file](.github/workflows/main.yaml).

![trigger deployment](/images/trigger_deployment.png "trigger doployment")

![deploy to heroku github actions](/images/deploy_to_heroku_github_actions.png "deploy to heroku github actions")

![send post request to heroku](/images/send_post_request_to_heroku.png "send post request to heroku")

# Contact

If you have any question please [create an issue](https://github.com/ozgeozge/heart-failure-prediction/issues/new).
