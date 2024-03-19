cd /home/work/inter_speech/joint-learning
source /home/work/.local/anaconda3/bin/activate
conda activate ehr
python train.py config/flan-t5-base.yaml logging.run_name=baseline-shorter
python train.py config/flan-t5-base.yaml logging.run_name=baseline-shorter-all data.split_ratio=1.0 scheduler.num_training_steps=900

# gsutil cp -r /home/work/inter_speech/joint-learning/text-to-sql/gsgzd44p/checkpoints/epoch=9-step=730.ckpt gs://lrw-dataset/CKPT-NAACL
# gsutil cp -r /home/work/inter_speech/joint-learning/text-to-sql/wvu5e63l/checkpoints/*.ckpt gs://lrw-dataset/CKPT-NAACL
conda activate lip
cd /home/work/inter_speech/sync_auto_avsr
python main.py conf/vox2+lrs2+lrs3_reorganized_8gpu.yaml