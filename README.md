# Pref-FEND论文

> **Integrating Pattern- and Fact-based Fake News Detection via Model Preference Learning.**
>
> Qiang Sheng\*, Xueyao Zhang\*, Juan Cao, and Lei Zhong.
>
> *Proceedings of the 30th ACM International Conference on Information and Knowledge Management (CIKM 2021)*
>
> [PDF](https://dl.acm.org/doi/10.1145/3459637.3482440) / [Poster](https://www.zhangxueyao.com/data/cikm2021-PrefFEND-poster.pdf) / [Code](https://github.com/ICTMCG/Pref-FEND)



## 数据集

论文原始实验数据集可以在“数据集”文件夹中看到，包括[Weibo Dataset](https://github.com/ICTMCG/Pref-FEND/tree/main/dataset/Weibo)和[Twitter Dataset](https://github.com/ICTMCG/Pref-FEND/tree/main/dataset/Twitter)。不过请注意，只有在提交了[“Application to Use the Datasets for Pattern- and Fact-based Joint Fake News Detection”](https://forms.office.com/r/HF00qdb3Zk)之后，您才能下载获取该数据集。

此外，我还使用了其它数据集（包括原始的和处理后的）。比如[CHEF]（https://github.com/THU-BPM/CHEF）（强烈推荐🤓），GossipCop,不过，由于该数据集太大，故而没有上传到GitHub。需要的朋友请email我：loyiv5477@gmail.com


## 代码相关

### 📦安装依赖环境

请确保你已安装 Python 和 pip，然后运行以下命令安装本项目所需的全部依赖：

```bash
pip install -r requirements.txt

### 准备工作

#### 步骤1：Stylistic Tokens & Entities Recognition

这一步是进行词项的识别：实体词，风格词，其它词。你可以通过`process.py`来对数据集进行处理。正常来说，执行完后会得到raw目录下识别好的post,article的json文件。



#### 步骤2：Tokenize

顾名思义，这一步是tokenizer，使用的预训练模型是谷歌的bert，值得注意的是，如果所使用的数据集为中文，请采用bert-cased-chinese，英文则为bert-cased-english。bert-base-cased获取：[bert-base-cased获取](https://github.com/rohithjoginapally/bert-base-cased)

```
cd preprocess/tokenize
```

正如`run.sh`所示, 您需要运行:

```
python get_post_tokens.py --dataset [dataset] --pretrained_model [bert_pretrained_model]
```

#### 步骤3：Heterogeneous Graph Initialization

```
cd preprocess/graph_init
```

正如`run.sh`所示, 您需要运行:

```
python init_graph.py --dataset [dataset] --max_nodes [max_tokens_num]
```

#### 步骤4：Preparation of the Fact-based Models

注意，如果您不使用基于事实的模型作为Pref-FEND的一个组件，那么这一步就不是必需的。

##### Tokenize

```
cd preprocess/tokenize
```

正如`run.sh`所示, 您需要运行:

```
python get_articles_tokens.py --dataset [dataset] --pretrained_model [bert_pretrained_model]
```

##### Retrieve by BM25

```
cd preprocess/bm25
```

正如`run.sh`所示, 您需要运行:

```
python retrieve.py --dataset [dataset]
```

#### 步骤5：Preparation for some special fake news detectors

注意，如果您不使用“EANN-Text”或“BERT-Emo”作为Pref-FEND的一个组件，那么这一步就不是必需的。

##### EANN-Text

```
cd preprocess/EANN_Text
```

正如`run.sh`所示, 您需要运行:

```
python events_clustering.py --dataset [dataset] --events_num [clusters_num]
```

##### BERT-Emo

```
cd preprocess/BERT_Emo/code/preprocess
```

正如`run.sh`所示, 您需要运行:

```
python input_of_emotions.py --dataset [dataset]
```

### Training and Inferring

```
cd model
mkdir ckpts
```

`run_gossip.sh`中列出了训练和推理步骤中的所有配置和运行脚本。例如，如果您想在GossipCop上运行“BiLSTM(基于模式的) +DeClarE(基于事实的)”，您可以运行:

```

# BiLSTM + DeClarE (Pref-FNED)
CUDA_VISIBLE_DEVICES=0 python main.py --dataset 'gossip' \
--use_preference_map True --use_pattern_based_model True --use_fact_based_model True \
--pattern_based_model 'BiLSTM' --fact_based_model 'DeClarE' \
--lr 1e-4 --batch_size 4 --epochs 20 \
--save 'ckpts/BiLSTM+DeClarE_with_Pref-FEND'

```

之后结果将会被保存在“ckpts/BiLSTM+DeClarE_with_Pref-FEND”中。

### 实验结果

相关实验结果均保存在“Pref-FEND-main/model/ckpts”中。此外，课程大作业中也展示了相关实验结果。

## Citation

```
@inproceedings{Pref-FEND,
  author    = {Qiang Sheng and
               Xueyao Zhang and
               Juan Cao and
               Lei Zhong},
  title     = {Integrating Pattern- and Fact-based Fake News Detection via Model
               Preference Learning},
  booktitle = {{CIKM} '21: The 30th {ACM} International Conference on Information
               and Knowledge Management, Virtual Event, Queensland, Australia, November
               1 - 5, 2021},
  pages     = {1640--1650},
  year      = {2021},
  url       = {https://doi.org/10.1145/3459637.3482440},
  doi       = {10.1145/3459637.3482440}
}
```
