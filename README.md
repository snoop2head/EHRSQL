
### Install
```shell
source /home/ubuntu/miniconda3/bin/activate base
conda create -n ehr python=3.11
conda activate ehr
pip install -r requirements.txt
```

### Train
```shell
source /home/ubuntu/miniconda3/bin/activate base
conda activate ehr
python train.py config/flan-t5-base.yaml
```

Or

```shell
bash run.sh
```

### Predict
```shell
python predict.py config/flan-t5-base.yaml
```