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


# 测试buffer memory


# from QiJiModel.GLM import GLM
#
# MODEL_PATH = "/home/qiji/chatglm2-6b/"  # 模型路径
# llm = GLM()
# llm.load_model(model_name_or_path=MODEL_PATH)
#
# # 请注意,你需要使用这些模块,可以使用更加高级的
# from langchain.memory import ConversationBufferMemory  # 这是最简单的,后面会教学使用高级记忆
# from langchain.chains import ConversationChain
#
# memory = ConversationBufferMemory()
# conversation = ConversationChain(
#     llm=llm,
#     verbose=True,
#     memory=ConversationBufferMemory()
# )
#
# index = 0
# while index < 2:
#     index += 1
#     q = input('> ')
#     print(conversation.predict(input=q))


# 测试是否能正确保存我的向量数据库
from typing import Union, Type

from langchain.vectorstores.chroma import Chroma
from QiJiModel import Text2Vec
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.html_bs import BSHTMLLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

# 常量
FILE_LOADER_TYPE = Union[
    Type[UnstructuredFileLoader], Type[TextLoader], Type[BSHTMLLoader]
]


# 用来实现便捷的相似度文本匹配
class SimilaritySearch(object):
    def __init__(self, embeddings=None, persist_path: str = None):  # 注意:默认的模型的类型是`text2vec-large-chinese`
        self.db: Chroma = None  # 数据库
        if embeddings is None:
            embedding_model_path: str = "/home/qiji/text2vec-large-chinese"  # `text2vec-large-chinese`的模型路径
            embeddings: Text2Vec = Text2Vec(model_name=embedding_model_path)  # 加载模型

        self.embeddings = embeddings  # 词向量模型

        self.persist_path = persist_path  # 持久化路径

    def load_text(self, file_path: str, chunk_size: int = 300, chunk_overlap: int = 100) -> bool:
        """
        加载文本
        :param file_path: 文本文件路径
        :return: 是否加载成功
        """
        try:
            raw_documents = TextLoader(file_path).load()  # 读取文本
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                           chunk_overlap=chunk_overlap)  # 实例化用来分割文本的类
            documents = text_splitter.split_documents(raw_documents)  # 分割文本
            self.db = Chroma.from_documents(documents, self.embeddings,
                                            persist_directory=self.persist_path)  # 将文本转换为向量并存储

        except Exception as e:
            return False

        return True

    def load_dir(self, dir_path: str, loader_cls: FILE_LOADER_TYPE, chunk_size: int = 300, chunk_overlap: int = 100,
                 show_progress: bool = True) -> bool:
        try:
            raw_documents = DirectoryLoader(dir_path, loader_cls=loader_cls, show_progress=show_progress,
                                            use_multithreading=True).load()  # 我在这使用了多线程,如果出现问题,删除`use_multithreading=True`
            print(raw_documents)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                           chunk_overlap=chunk_overlap)  # 实例化用来分割文本的类
            documents = text_splitter.split_documents(raw_documents)  # 分割文本
            self.db = Chroma.from_documents(documents, self.embeddings,
                                            persist_directory=self.persist_path)  # 将文本转换为向量并存储

        except Exception as e:
            return False

        return True

    def load_db(self, db_path: str) -> bool:
        try:
            # 从持久化路径加载数据库
            self.db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)

        except Exception as e:
            return False
        return True

    def load(self, file_path: str) -> bool:
        """
        加载数据
        :param file_path: 文件路径
        :return: 是否加载成功
        """
        if file_path.endswith(".txt"):
            return self.load_text(file_path)

        return False

    def save(self):
        if self.persist_path is not None:
            self.db.persist()  # 持久化
        else:
            # 如果没有指定持久化路径,则抛出异常
            raise Exception("没有持久化路径")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """
        搜索
        :param query: 查询语句
        :param top_k: 返回的最大结果数
        :return: 返回的结果
        """
        if self.db is None:
            raise Exception("请先加载数据")

        docs = self.db.similarity_search(query, k=top_k)  # 查询，注意`k`是返回的文本数量
        return [doc.page_content for doc in docs]  # 返回文本


PERSIST_PATH = "/home/qiji/tmp/chroma"
ss = SimilaritySearch(persist_path=PERSIST_PATH)  # 实例化
FILE_PATH = "/home/qiji/Container/xuhe/compare_car/data/car_names.txt"

ss.load_db(PERSIST_PATH)

for _ in ss.search('丰田C-HR'):
    print(_)
