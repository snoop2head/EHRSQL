cd /home/work/inter_speech/joint-learning
source /home/work/.local/anaconda3/bin/activate
conda activate ehr
python train.py config/flan-t5-base.yaml logging.run_name=baseline
python train.py config/flan-t5-base.yaml lambda_null_classification=10 logging.run_name=lambda10-baseline
python train.py config/flan-t5-base.yaml optimizer.lr=0.0001 logging.run_name=lambda10-baseline-1e-4-lr
python train.py config/flan-t5-base.yaml optimizer.lr=0.00005 logging.run_name=lambda10-baseline-5e-5-lr
python train.py config/flan-t5-base.yaml optimizer.lr=0.00001 logging.run_name=lambda10-baseline-1e-5-lr
# gsutil cp -r /home/work/inter_speech/joint-learning/text-to-sql/gsgzd44p/checkpoints/epoch=9-step=730.ckpt gs://lrw-dataset/CKPT-NAACL
gsutil cp -r /home/work/inter_speech/joint-learning/text-to-sql/wvu5e63l/checkpoints/*.ckpt gs://lrw-dataset/CKPT-NAACL