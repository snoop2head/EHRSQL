cd /home/yjahn/joint-learning
source /home/yjahn/miniconda3/bin/activate
conda activate ehr
python train.py ./config/flan-t5-large.yaml data.kfold_split=0
python train.py ./config/flan-t5-large.yaml data.kfold_split=1
python train.py ./config/flan-t5-large.yaml data.kfold_split=2
python train.py ./config/flan-t5-large.yaml data.kfold_split=3
python train.py ./config/flan-t5-large.yaml data.kfold_split=4
python train.py ./config/flan-t5-large.yaml data.kfold_split=5
python train.py ./config/flan-t5-large.yaml data.kfold_split=6
python train.py ./config/flan-t5-large.yaml data.kfold_split=7
python train.py ./config/flan-t5-large.yaml data.kfold_split=8
python train.py ./config/flan-t5-large.yaml data.kfold_split=9
python train.py ./config/flan-t5-base.yaml data.kfold_split=0
python train.py ./config/flan-t5-base.yaml data.kfold_split=1
python train.py ./config/flan-t5-base.yaml data.kfold_split=2
python train.py ./config/flan-t5-base.yaml data.kfold_split=3
python train.py ./config/flan-t5-base.yaml data.kfold_split=4
python train.py ./config/flan-t5-base.yaml data.kfold_split=5
python train.py ./config/flan-t5-base.yaml data.kfold_split=6
python train.py ./config/flan-t5-base.yaml data.kfold_split=7
python train.py ./config/flan-t5-base.yaml data.kfold_split=8
python train.py ./config/flan-t5-base.yaml data.kfold_split=9
