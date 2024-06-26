source /home/yjahn/miniconda3/bin/activate
conda activate ehr
python predict.py ./config/flan-t5-large.yaml data.kfold_split=0 predict.ckpt_path=./text-to-sql/ctgw31o5/checkpoints/baselinev3-flan-t5-large_fold0epoch=6-step=2023.ckpt predict.inference_valid=True train.valid_batch_size=16
python predict.py ./config/flan-t5-large.yaml data.kfold_split=1 predict.ckpt_path=./text-to-sql/attj8f9k/checkpoints/baselinev3-flan-t5-large_fold1epoch=6-step=2023.ckpt predict.inference_valid=True train.valid_batch_size=16
python predict.py ./config/flan-t5-large.yaml data.kfold_split=2 predict.ckpt_path=./text-to-sql/2b39zjk4/checkpoints/baselinev3-flan-t5-large_fold2epoch=7-step=2312.ckpt predict.inference_valid=True train.valid_batch_size=16
python predict.py ./config/flan-t5-large.yaml data.kfold_split=3 predict.ckpt_path=./text-to-sql/ck3uyebq/checkpoints/baselinev3-flan-t5-large_fold3epoch=7-step=2312.ckpt predict.inference_valid=True train.valid_batch_size=16
python predict.py ./config/flan-t5-large.yaml data.kfold_split=4 predict.ckpt_path=./text-to-sql/m9xe77nu/checkpoints/baselinev3-flan-t5-large_fold4epoch=5-step=1734.ckpt predict.inference_valid=True train.valid_batch_size=16
python predict.py ./config/flan-t5-large.yaml data.kfold_split=5 predict.ckpt_path=./text-to-sql/d0jjw560/checkpoints/baselinev3-flan-t5-large_fold5epoch=5-step=1734.ckpt predict.inference_valid=True train.valid_batch_size=16
python predict.py ./config/flan-t5-large.yaml data.kfold_split=6 predict.ckpt_path=./text-to-sql/8d9yxbzf/checkpoints/baselinev3-flan-t5-large_fold6epoch=6-step=2023.ckpt predict.inference_valid=True train.valid_batch_size=16
python predict.py ./config/flan-t5-large.yaml data.kfold_split=7 predict.ckpt_path=./text-to-sql/prehkxva/checkpoints/baselinev3-flan-t5-large_fold7epoch=6-step=2023.ckpt predict.inference_valid=True train.valid_batch_size=16
python predict.py ./config/flan-t5-large.yaml data.kfold_split=8 predict.ckpt_path=./text-to-sql/d6rhbnpl/checkpoints/baselinev3-flan-t5-large_fold8epoch=5-step=1734.ckpt predict.inference_valid=True train.valid_batch_size=16
python predict.py ./config/flan-t5-large.yaml data.kfold_split=9 predict.ckpt_path=./text-to-sql/qyuoqc1r/checkpoints/baselinev3-flan-t5-large_fold9epoch=5-step=1734.ckpt predict.inference_valid=True train.valid_batch_size=16
python train.py ./config/flan-t5-base.yaml data.kfold_split=3
python train.py ./config/flan-t5-base.yaml data.kfold_split=4
python train.py ./config/flan-t5-base.yaml data.kfold_split=5
python train.py ./config/flan-t5-base.yaml data.kfold_split=6
python train.py ./config/flan-t5-base.yaml data.kfold_split=7
python train.py ./config/flan-t5-base.yaml data.kfold_split=8
python train.py ./config/flan-t5-base.yaml data.kfold_split=9