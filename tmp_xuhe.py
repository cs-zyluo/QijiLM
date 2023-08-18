# # 这是一个用于各种奇怪测试的文件
# from QiJiModel.GLM import GLM
# from QiJiPrompt.compare_car import CompareCarPrompt
#
# MODEL_PATH = "/home/qiji/chatglm2-6b/"  # 模型路径
# llm = GLM()
# llm.load_model(model_name_or_path=MODEL_PATH)
#
# from langchain.chains import LLMChain
# from langchain.chains import SequentialChain
#
# chain1 = LLMChain(llm=llm, prompt=CompareCarPrompt()._get_car_names_prompt(), output_key="car_names")
# chain2 = LLMChain(llm=llm, prompt=CompareCarPrompt()._get_car_attributes_prompt(), output_key="car_attributes")
#
# chain = SequentialChain(
#     chains=[chain1, chain2],
#     input_variables=["question"],
#     output_variables=["car_attributes"],
#     verbose=True
# )
#
# print(chain.run(question="雷克萨斯RX和起亚K9哪个好"))

# from langchain.vectorstores.chroma import Chroma
# from QiJiModel import Text2Vec
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
#
# embedding_model_name = "/home/qiji/text2vec-large-chinese"  # 模型路径
# embeddings = Text2Vec(model_name=embedding_model_name)
# # 然后就可以放入'langchain' 中使用
#
# # Load the document, split it into chunks, embed each chunk and load it into the vector store.
# raw_documents = TextLoader('./xuhe/data/car_names.txt').load()  # 读取文本
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)  # 实例化用来分割文本的类
# documents = text_splitter.split_documents(raw_documents)  # 分割文本
# db = Chroma.from_documents(documents, embeddings)  # 将文本转换为向量并存储

# query = input('> ')  # 输入查询文本
# docs = db.similarity_search(query, k=2)  # 查询，注意`k`是返回的文本数量
# print(docs[0].page_content)  # 打印最相似的文本
# print(docs[1].page_content)


# from QiJiOther.SimilaritySearch import SimilaritySearch
# from langchain.document_loaders import TextLoader
#
# ss = SimilaritySearch()  # 实例化
# dir_path = "./xuhe/data"
# ss.load_dir(dir_path=dir_path, loader_cls=TextLoader)  # 加载文本
#
# res_list = ss.search("丰田C-HR")
#
# for _ in res_list:
#     print()
#     print(_)

# google search api
# from QiJiModel.GLM import GLM
# # Import things that are needed generically
# from langchain import LLMMathChain, SerpAPIWrapper
# from langchain.agents import AgentType, initialize_agent
# from langchain.tools import BaseTool, StructuredTool, Tool, tool
#
# from QiJiModel.GLM import GLM
#
# MODEL_PATH = "/home/qiji/chatglm2-6b/"  # 模型路径
# llm = GLM()
# llm.load_model(model_name_or_path=MODEL_PATH)
#
# # Initialize the agent
# # Load the tool configs that are needed.
# search = SerpAPIWrapper()
# llm_math_chain = LLMMathChain(llm=llm, verbose=True)
# tools = [
#     Tool.from_function(
#         func=search.run,
#         name="Search",
#         description="useful for when you need to answer questions about current events"
#         # coroutine= ... <- you can specify an async method if desired as well
#     ),
# ]
#
# agent = initialize_agent(
#     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
# )
# agent.run("今天天气如何？")


from langchain.llms import OpenAI