# 本地知识问答系统
## 实现步骤
1. 加载文档（PDF、HTML、文本、数据库等）；
2. 将数据分割成块，并对这些块建立embedding索引，方便使用向量检索工具进行语义搜索；
3. 对于每个问题，通过搜索索引和embedding数据来获取与问题相关的信息；
4. 将问题和相关数据输入到LLM模型中。在这个系列中使用OpenAI的LLM；

实现上述过程主要的两个框架，分别是：
- Langchain（https://python.langchain.com/en/latest/）
- LLamaIndex（https://gpt-index.readthedocs.io/en/latest/）

## 环境配置
- Python>=3.7
- pip install -r requirements.txt

## 总结
本地知识问答就是对知识文本分段Embedding存储，针对问题Embedding，找到存储中相似的文本，并使用该文本进行总结回答