cd /home/work/inter_speech/joint-learning
source /home/work/.local/anaconda3/bin/activate
conda activate ehr

python train.py config/flan-t5-base.yaml logging.run_name=baselinev2
python train.py config/flan-t5-base.yaml inference.is_impossible_threshold=0.1 logging.run_name=baselinev2-impo0.1
python train.py config/flan-t5-base.yaml optimizer.lr=0.002 logging.run_name=baselinev2-lr0.002
python train.py config/flan-t5-base.yaml optimizer.lr=0.003 logging.run_name=baselinev2-lr0.003
python train.py config/flan-t5-base.yaml optimizer.lr=0.0005 logging.run_name=baselinev2-lr0.0005
python train.py config/flan-t5-base.yaml optimizer.lr=0.0001 logging.run_name=baselinev2-lr0.0001
python train.py config/flan-t5-base.yaml optimizer.lr=0.002 is_impossible_threshold=0.1 logging.run_name=baselinev2-lr0.002-impo0.1
python train.py config/flan-t5-base.yaml optimizer.lr=0.003 is_impossible_threshold=0.1 logging.run_name=baselinev2-lr0.003-impo0.1
python train.py config/flan-t5-base.yaml optimizer.lr=0.0005 is_impossible_threshold=0.1 logging.run_name=baselinev2-lr0.0005-impo0.1
python train.py config/flan-t5-base.yaml optimizer.lr=0.0001 is_impossible_threshold=0.1 logging.run_name=baselinev2-lr0.0001-impo0.1
python train.py config/flan-t5-base.yaml logging.run_name=baselinev2-6epoch scheduler.num_training_steps=500
python train.py config/flan-t5-base.yaml inference.is_impossible_threshold=0.1 logging.run_name=baselinev2-6epoch-impo0.1 scheduler.num_training_steps=500
python train.py config/flan-t5-base.yaml optimizer.lr=0.002 logging.run_name=baselinev2-6epoch-lr0.002 scheduler.num_training_steps=500
python train.py config/flan-t5-base.yaml optimizer.lr=0.003 logging.run_name=baselinev2-6epoch-lr0.003 scheduler.num_training_steps=500
python train.py config/flan-t5-base.yaml optimizer.lr=0.0005 logging.run_name=baselinev2-6epoch-lr0.0005 scheduler.num_training_steps=500
python train.py config/flan-t5-base.yaml optimizer.lr=0.0001 logging.run_name=baselinev2-6epoch-lr0.0001 scheduler.num_training_steps=500
python train.py config/flan-t5-base.yaml optimizer.lr=0.002 is_impossible_threshold=0.1 logging.run_name=baselinev2-6epoch-lr0.002-impo0.1 scheduler.num_training_steps=500
python train.py config/flan-t5-base.yaml optimizer.lr=0.003 is_impossible_threshold=0.1 logging.run_name=baselinev2-6epoch-lr0.003-impo0.1 scheduler.num_training_steps=500
python train.py config/flan-t5-base.yaml optimizer.lr=0.0005 is_impossible_threshold=0.1 logging.run_name=baselinev2-6epoch-lr0.0005-impo0.1 scheduler.num_training_steps=500
python train.py config/flan-t5-base.yaml optimizer.lr=0.0001 is_impossible_threshold=0.1 logging.run_name=baselinev2-6epoch-lr0.0001-impo0.1 scheduler.num_training_steps=500

# gsutil cp -r /home/work/inter_speech/joint-learning/text-to-sql/gsgzd44p/checkpoints/epoch=9-step=730.ckpt gs://lrw-dataset/CKPT-NAACL
gsutil cp -r /home/work/inter_speech/joint-learning/*.ckpt gs://lrw-dataset/CKPT-NAACL
conda activate lip
cd /home/work/inter_speech/sync_auto_avsr
python main.py conf/vox2+lrs2+lrs3_reorganized_8gpu.yaml