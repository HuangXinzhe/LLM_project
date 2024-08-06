# 大模型高效微调技术原理
## 高效微调技术简介
BitFit、Prefix Tuning、Prompt Tuning、P-Tuning、P-Tuning v2、Adapter Tuning及其变体、LoRA、AdaLoRA、QLoRA、MAM Adapter、UniPELT等。
## 高效微调技术原理
### BitFit
BitFIt只对模型的bias进行微调。在小规模-中等规模的训练数据上，BitFit的性能与全量微调的性能相当，甚至有可能超过，在大规模训练数据上，与其他fine-tuning方法也差不多。在大模型中bias存在Q,K,V,MLP,LayerNorm中。
 
BitFit只对Q,K,V,MLP,LayerNorm的bias进行微调，而不对权重进行微调。BitFit的优势在于，bias的数量远远小于权重的数量，因此BitFit的参数量远远小于全量微调，因此BitFit的速度更快，同时BitFit的性能也不错。

bias参数仅占模型全部参数量的0.08%～0.09%，因此BitFit的参数量远远小于全量微调，因此BitFit的速度更快，同时BitFit的性能也不错。

### Prefix Tuning
prefix-tuning方法是一个轻量级的fine-tuning方法用于自然语言处理的生成任务。该方法可以保持预训练语言模型参数固定（frozen），而只需要在task-specific vector（称为prefix）上进行优化。即只需要少量（约0.1%）的优化参数，即可以在量和小量数据上达到不错的效果。

在每层中都可以加入prefix，prefix的维度可以自定义，可以是1维，也可以是多维。prefix的维度越大，模型的参数量越大，但是模型的性能也会更好。

### Prompt Tuning
Prompt Tuning可以看作是Prefix Tuning的简化版本，面向NLU任务，进行了更全面的效果对比，并且在大模型上成功打平了LM微调的效果，它给每个任务定义了自己的Prompt，然后拼接到数据上作为输入，但只在输入层加入prompt tokens，并且不需要加入 MLP 进行调整来解决难训练的问题。通过反向传播更新参数来学习prompts，而不是人工设计prompts；同时冻结模型原始权重，只训练prompts参数，训练完以后，用同一个模型可以做多任务推理。

### P-Tuning
P-Tuning是一种轻量级的fine-tuning方法，用于自然语言处理的生成任务。P-Tuning方法可以保持预训练语言模型参数固定（frozen），而只需要在task-specific vector（称为prompt）上进行优化。即只需要少量（约0.1%）的优化参数，即可以在量和小量数据上达到不错的效果。

### P-Tuning v2
P-Tuning v2是P-Tuning的改进版本，P-Tuning v2在P-Tuning的基础上，引入了一个新的技术，即使用一个额外的MLP来调整prompt的输出，以解决难训练的问题。P-Tuning v2在P-Tuning的基础上，引入了一个新的技术，即使用一个额外的MLP来调整prompt的输出，以解决难训练的问题。

### Adapter Tuning
Adapter Tuning是一种轻量级的fine-tuning方法，用于自然语言处理的生成任务。Adapter Tuning方法可以保持预训练语言模型参数固定（frozen），而只需要在task-specific vector（称为adapter）上进行优化。即只需要少量（约0.1%）的优化参数，即可以在量和小量数据上达到不错的效果。