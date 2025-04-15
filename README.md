# CausalCode

Code for paper "A Causal Learning Framework for Enhancing Robustness of Source Code Models"

The code is able to reproduce the experimental results in the paper. In order to be anonymous, the paths are eliminated in bulk, so changing the appropriate path can make the code run normally.

## Create environment

```
conda env create -f CausalCode-environment.yml
```

## Preparing the Dataset

Classification task: we use the same dataset as carrot : https://github.com/SEKE-Adversary/CARROT

Generation task: The data can be obtained from [CodeXGLUE](https://github.com/microsoft/CodeBERT).

**Use the pre-processed datasets**

(Provided by Carrot)

1. Download the already pre-processed datasets -- [OJ](https://drive.google.com/drive/folders/1__SjuEKH8Sa_OYWhegiGE6Brbr1ObZrM?usp=sharing) and [CodeChef](https://drive.google.com/drive/folders/1ZEIb35PzfD2ojWr53Qa_myFRMVK7QI7f?usp=sharing).
2. Put the contents of OJ and CodeChef in the corresponding directories of `data` and `data_defect` respectively.
3. The pre-processed datasets for LSTM and BERT are all included in the directories now.

**Pre-process the datasets by yourself**

1. Download the raw datasets, *i.e.*, `oj.tar.gz` from [OJ](https://drive.google.com/drive/folders/1__SjuEKH8Sa_OYWhegiGE6Brbr1ObZrM?usp=sharing) and `codechef.zip` from [CodeChef](https://drive.google.com/drive/folders/1ZEIb35PzfD2ojWr53Qa_myFRMVK7QI7f?usp=sharing)
2. Put `oj.tar.gz` in the directory of `data` and `codechef.zip` in `data_defect`.
3. Run the following commands to build the OJ dataset for the DL models. The dataset format of CodeChef is almost identical to OJ, and the code can be reused.

```sh
> cd preprocess-lstm
> python3 main.py 
```

4. Copy `oj.pkl.gz`, `oj_uid.pkl.gz`, and `oj_inspos.pkl.gz` in the directory of `data`.

## Train

### Classification Task

graphcodebert in train_bert

Training the CausalCode model is done in 'train_CausalCode' function

```sh
python run.py \
--do_train --do_renew --do_eval --do_attack  \
--enhance_method CausalCode --attack_type token --task code_defect \
--epochs 28  --early_stop 3   \
--domain_list origin,LSTM-CausalCode-token
```

### Generation Task

```sh
task=Translate
base_model_name=CodeBERT
root_dir="../code_attack/" # Use absolute paths
batch_size=8


if [ ${task} == 'Translate' ];then
   lang=java_cs
elif [ ${task} == 'Summarize' ];then
    lang=java
elif [ ${task} == 'Refine' ];then
    lang=java
fi

dataset_name=codebert_augment_CodeBERT_output_iter_10
CausalCode_data_dir=${root_dir}data/${task}/dataset/$dataset_name/
train_filename=${CausalCode_data_dir}/merge_train.jsonl

CausalCode_model_name="${base_model_name}-CausalCode-${dataset_name}-0317"
load_model_path="${root_dir}data/${task}/model/${lang}/${base_model_name}/ckpt/model.bin"

data_dir=${root_dir}data/${task}/dataset/${lang}
wandb_name=$base_model_name-$dataset_name-$task


base_model_dir="../codeattack/base_model/"


if [ ${base_model_name} == 'CodeBERT' ];then
	base_model=${base_model_dir}/codebert-base
elif [ ${base_model_name} == 'ContraBERT_C' ];then
	base_model=${base_model_dir}/contraBert_C
elif [ ${base_model_name} == 'GraphCodeBERT' ];then
	base_model=${base_model_dir}/graphcodebert-base
elif [ ${base_model_name} == 'contraBERT_G' ];then
	base_model=${base_model_dir}/contraBert_G
fi

output_dir=${root_dir}data/${task}/model/${lang}/${CausalCode_model_name}
mkdir -p ${output_dir}
test_filename=${data_dir}/test.jsonl
dev_filename=${data_dir}/valid.jsonl
adv_filename=${root_dir}/data/Translate/model/java_cs/CodeBERT/topk_80_5_output/test.jsonl

python -u run_causal.py \
    --do_train --do_eval --do_test  \
    --model_type roberta \
    --model_name_or_path $base_model \
    --config_name $base_model \
    --tokenizer_name $base_model \
    --train_filename ${train_filename} \
    --dev_filename ${dev_filename} \
    --test_filename ${test_filename} \
  --adv_filename ${adv_filename} \
    --output_dir ${output_dir} \
    --max_source_length 512 \
    --max_target_length 512 \
    --beam_size 10 \
    --train_batch_size ${batch_size} \
    --eval_batch_size ${batch_size} \
    --load_model_path $load_model_path \
    --learning_rate 5e-5 \
    --wandb_name $wandb_name \
    --train_epochs 35

```
