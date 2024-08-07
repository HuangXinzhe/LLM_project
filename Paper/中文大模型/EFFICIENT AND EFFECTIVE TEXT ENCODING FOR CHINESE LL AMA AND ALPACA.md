# EFFICIENT AND EFFECTIVE TEXT ENCODING FOR CHINESE LL AMA AND ALPACA
- 中文大语言模型-Llama即使报告
- https://arxiv.org/pdf/2304.08177v1.pdf
- https://github.com/ymcui/Chinese-LLaMA-Alpaca
## 介绍
- 通过在原有的LLaMA词汇中增加20,000个中文符号来提高中文编码和解码的效率，并提高LLaMA的中文理解能力。
- 采用低秩适应（LoRA）的方法来有效地训练和部署中国的LLaMA和Alpaca模型，使研究人员能够在不产生过多计算成本的情况下使用这些模型。
- 评估了中国羊驼7B和13B模型在各种自然语言理解（NLU）和自然语言生成（ NLG）任务中的表现，表明在中文语言任务中比原来的LLaMA对应模型有明显的改进。
- 公开了我们的研究资源和结果，促进了NLP社区的进一步研究和合作，并鼓励将LLaMA和Alpaca模型改编为其他语言。
## CHINESE LLaMA
- 扩展中文词汇表，增加了20,000个中文符号，以提高中文编码和解码的效率。最终词汇量为49953。
## CHINESE Alpaca
- 应用自我训练的微调来训练指令跟随模型
- Alpaca模型有一个额外的填充标记，导致词汇量为49,954。
## 用LORA进行参数有效的微调
