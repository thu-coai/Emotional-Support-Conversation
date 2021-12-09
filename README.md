# Emotional-Support-Conversation
### *Copyright © 2021 CoAI Group, Tsinghua University. All rights reserved. Data and codes are for academic research use only.*

Data and codes for the ACL 2021 paper: [**Towards Emotional Support Dialog Systems**](https://arxiv.org/abs/2106.01144)

If you use our codes or your research is related to our paper, please kindly cite our paper:

```bib
@inproceedings{liu-etal-2021-towards,
  title={Towards Emotional Support Dialog Systems},
  author={Liu, Siyang  and 
    Zheng, Chujie  and 
    Demasi, Orianna  and 
    Sabour, Sahand  and 
    Li, Yu  and 
    Yu, Zhou  and 
    Jiang, Yong  and 
    Huang, Minlie},
  booktitle={Proceedings of the 59th annual meeting of the Association for Computational Linguistics},
  year={2021}
}
```

## Data

The corpus file is `ESConv.json`. We have collected **more** conversations with more problem topics. ESConv now contians 1,300 conversations with 10 topic problems.

### Statistics
#### Problem Category

| Problem Category | ongoing depression | breakup with partner | job crisis | problems with friends | academic pressure | procras-<br>tination* | alcohol abuse* | issues with parent* | sleep problems* |  appearance anxiety* | school bullying* | issues with children* |
| :-------- | :---------- | :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- |  :---------- | :---------- | :---------- | 
| Number| 351 | 239 | 280 | 179 | 156 |  13 | 12 | 18 | 28 | 12 | 2 | 10 |


\* denotes the new topics added during the second collection. We hope new data supports the future research in transferring the ability of models from old topics to new ones. 

<font size=1>

#### Strategy Category 
| Strategy Category| Number   |
| :--------------  | :------- |
| Questions | 3801(20.7%)|
| Self-disclosure | 1713(9.3%) |
| Affirmation and Reassurance | 2827(15.4%) |
| Providing Suggestions | 2954(16.1%) |
| Other | 3661(18.3%) / 3341(18.2%) |
| Reflection of feelings |  1436(7.8%) | 
| Information | 1215(6.6%) | 
| Restatement or Paraphrasing | 1089(5.9%) |

</font>


## Model Implementation

We provide two versions of model implementation:

- `codes` is the version that we used in the original experiments
- `codes_zcj` is the version reproduced by  [@chujiezheng](https://github.com/chujiezheng)



