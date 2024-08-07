# Black-Box Prompt Optimization: Aligning Large Language Models without Model Training
论文链接：https://arxiv.org/abs/2311.04155

github地址：https://github.com/thu-coai/BPO

## BPO方法原理
1. 反馈数据收集：为了建模人类偏好，首先搜集了一系列带有反馈信号的开源指令微调数据集，并对这些数据经过精心筛选和过滤；
2. 构造提示优化对：使用这些反馈数据来引导大型模型识别用户喜欢的回复和不喜欢的回复，基于这些特征，再利用模型优化原始的用户输入，以期得到更符合用户喜好的模型输出；
3. 训练提示优化器：经过上述两个步骤，得到了大量优化前后的Prompt pair，利用这些Prompt pair训练一个seq2seq模型（作者使用llama2-7b-chat作为bachbone模型），这样后期就可以使用该seq2seq模型进行自动化优化用户的Prompt了

## 总结
通过对收集到的有标注Prompt进行筛选过滤，对Prompt进行进一步优化，并训练成模型，用户在使用的时候，对用户的Prompt进行优化