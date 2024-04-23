### Description
 We prioritize the privacy of patient information by exclusively using white-box models like flan-t5, avoiding potential data leakage associated with black-box models via APIs. To handle the computational constraints posed by lengthy SQL queries, we employed a T5 encoder-decoder architecture, allowing separate attention mechanisms for different parts of the input. We also exploit data efficiently by implementing an end-to-end architecture with a linear projection layer, enabling us to retain and utilize every part of the dataset, including traditionally discarded unanswerable questions. To optimize validation, we adopted a stratified KFold approach, ensuring no data split remains unused. Our monitoring strategy involves rigorous metric tracking during training, providing insights into model performance and areas needing adjustment. This approach helped us identify significant discrepancies in performance metrics and pinpoint areas for post-processing enhancement. Overall, our project demonstrates effective strategies for managing data privacy, optimizing dataset usage, and refining model performance through continuous evaluation.


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

### References
- Pytorch Lightning Template by [affjljoo3581/Inverse-DALL-E-for-Optical-Character-Recognition](https://github.com/affjljoo3581/Inverse-DALL-E-for-Optical-Character-Recognition)
- [Reliable Text-to-SQL on Electronic Health Records - Clinical NLP Workshop @ NAACL 2024](https://github.com/glee4810/ehrsql-2024)
