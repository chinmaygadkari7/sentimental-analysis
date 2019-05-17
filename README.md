# sentimental-analysis
Sentimental analysis of reviews

## Prerequisites

Create a virtual python environment and install all required packages
```
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Operations

```
python model.py -h

usage: Model Trainer Routine [-h] {train,predict,prepare} ...

positional arguments:
  {train,predict,prepare}
                        commands
    train               train model
    predict             predict review type
    prepare             prepare data

optional arguments:
  -h, --help            show this help message and exit
```
### Prepare data
Create train set and test set for future training and evaluation

```
python model.py prepare -h

usage: Model Trainer Routine prepare [-h] filename

positional arguments:
  filename    review text to predict

optional arguments:
  -h, --help  show this help message and exit
```
##### example:
```
python model.py prepare data.csv
```

### Train model

```
python model.py train -h

usage: Model Trainer Routine train [-h] {svm,log}

positional arguments:
  {svm,log}   Model name

optional arguments:
  -h, --help  show this help message and exit
```
##### example:
```
python model.py train svm
```

### Predict
```
python model.py predict -h

usage: Model Trainer Routine predict [-h] text

positional arguments:
  text        review text to predict

optional arguments:
  -h, --help  show this help message and exit
```
##### example:
```
python model.py predict "This is a good movie!"
```

## Running REST API
Start a web service at URL http://localhost:5000/

```
python app.py
```

#### Predict data
```
curl -X POST http://localhost:5000/predict \
-H 'cache-control: no-cache' \
-H 'Content-Type: text/plain' \
-d 'This is a good movie!'

```

## Results:

| Metrics    |   SVM  | Logistic Regression |
| :---       | :---   | :---                |
| Accuracy   | 81.07% | 81.60%              |
| Recall     | 79.09% | 79.09%              |
| Precision  | 82.17% | 83.10%              |
| F-Measure  | 80.60% | 81.04%              |
