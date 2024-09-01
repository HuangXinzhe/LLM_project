# Chinese-LLaMA-Alpaca：包含中文 LLaMA 模型和经过指令微调的 Alpaca 大型模型
efficient and effective text encoding for chinese llama and alpaca
资料地址：https://mp.weixin.qq.com/s/nq7xkMkDTeus3l-JvzcB4w
论文地址：https://arxiv.org/pdf/2304.08177v1.pdfGithub
地址：https://github.com/ymcui/Chinese-LLaMA-Alpaca

## 一、项目介绍
- 通过在原有的LLaMA词汇中增加20,000个中文符号来提高中文编码和解码的效率，并提高LLaMA的中文理解能力；
- 采用低秩适应（LoRA）的方法来有效地训练和部署中文的LLaMA和Alpaca模型，使研究人员能够在不产生过多计算成本的情况下使用这些模型；
- 评估了中文羊驼7B和13B模型在各种自然语言理解（NLU）和自然语言生成（NLG）任务中的表现，表明在中文语言任务中比原来的LLaMA对应模型有明显的改进；
- 公开了研究资源和结果，促进了NLP社区的进一步研究和合作，并鼓励将LLaMA和Alpaca模型改编为其他语言。
## 二、Chinese Llama
- 为了加强tokenizer对中文文本的支持，作者首先用SentencePiece在中文语料库上训练一个中文tokenizer，使用的词汇量为20,000。然后通过组合它们的词汇，将中文tokenizer合并到原始的LLaMA tokenizer中。最终得到了 一个合并的tokenizer，称之为中文LLaMA tokenizer，其词汇量为49,953；
- 为了适应中文LLaMA tokenizer的模型，作者将词嵌入和语言模型头的大小从形状V×H调整为V′×H，其中V=32,000代表原始词汇量，V′=49,953是中文LLaMA tokenizer的词汇量。新的行被附加到原始嵌入矩阵的末尾，以确保原始词汇中的标记的嵌入仍然不受影响。
## 三、Chinese Alpaca
在获得预训练的中文LLaMA模型后，作者按照斯坦福大学Alpaca中使用的方法，应用self-instructed微调来训练指令跟随模型。每个例子由一条指令和一个输出组成，将指令输入模型，并提示模型自动生成输出。这个过程类似于普通的语言建模任务。作者采用以下来自斯坦福大学Alpaca的提示模板，用于自我指导的微调，这也是在推理过程中使用的：