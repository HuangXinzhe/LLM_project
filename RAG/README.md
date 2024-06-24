# RAG
## RAG介绍
RAG主要有四个步骤
- 资料文本embedding化
- embedding文本存储入向量数据库中
- 问题embedding，并查询向量数据库中的相似文本
- 利用大模型对问题和检索到的文本进行答案生成

## 文件结构
- 01_Mistral-7b_LangChain_ChromaDB.ipynb
    - 基座模型：Mistral-7b
    - 数据库：ChromaDB
    - 工具：LangChain