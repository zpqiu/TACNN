# Description

Unofficial implementation of paper "Question Difﬁculty Prediction for READING Problems in Standard Tests".

# Prepare Data
The original paper didn't publish their dataset.
The data format supported by this code is json based file.
Each json line should be 
```json
{
    "difficulty": 0.XXX,
    "q": "TEXT OF QUESTION",
    "doc": "TEXT OF READING MATERIAL",
    "options": ["OPTION A", "OPTION B", "OPTION C", "OPTION D", "OPTION E"]
}
```

# Training and Testing
```shell
# Training
python main.py -cf conf.json --mode 0 
```

```shell
# Testing
python main.py -cf conf.json --mode 1 --epoch_for_test 1
```
