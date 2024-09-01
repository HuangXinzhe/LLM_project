# AgentTuning：清华大学与智谱AI提出AgentTuning提高大语言模型Agent能力
资料地址：https://mp.weixin.qq.com/s/qOMKbqavbVIX2qsqGYqBNw
论文地址：https://arxiv.org/pdf/2310.12823.pdf
Github地址：https://github.com/THUDM/AgentTuning

AgentTuning是一种简单而通用的方法，既可以增强LLM的Agent能力，有可以同时保持其通用LLM能力。AgentTuning具体方法是首先构造一个包含高质量交互轨迹的轻量级指令调优数据集AgentInstruction，然后采用混合指令微调策略将AgentInstruction与来自通用领域的开源指令相结合。AgentTuning对Llama 2系列模型进行指令微调产生AgentLM。

