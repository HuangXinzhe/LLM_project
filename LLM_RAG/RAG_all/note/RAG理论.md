# RAG理论
## 1. 笔记链（CHAIN-OF-NOTE）来提高检索增强模型（RAG）的透明度
笔记链的关键思想是通过对检索到的每个文档进行总结和评估，让模型生成阅读笔记，然后再生成最终的回应:
- 评估检索到文档的相关性
- 识别可靠信息与误导信息
- 过滤掉无关或不可信的内容
- 认识到知识差距并回应“未知”
阅读笔记类型：
- 相关（Relevant）：文档可以直接回答问题，最终的回复只来自该文档；
- 无关但有用的上下文（Irrelevant but useful context）：文档没有回答问题，但提供了有用的背景。该模型将其知识与上下文相结合可以推断出答案；
- 无关（Irrelevant）：文档是无关的，模型缺乏知识来回答。默认响应为“未知”。       该系统允许模型在直接检索信息、进行推断和承认其局限性之间取得平衡。
关键要点
- 笔记链增强了RALM对噪声检索和未知场景的鲁棒性；
- 记笔记为RALM推理过程提供了可解释性；
- 平衡检索信息、进行推断和确认限制；
- 分解复杂问题的简单而有效的方法。
## 2. self-rag
- 对检索到的文本段进行相关性的判断
- 利用具有相关性的文本段作为答案生成的上下文
- 利用第二步上下文通过大模型得出问题答案