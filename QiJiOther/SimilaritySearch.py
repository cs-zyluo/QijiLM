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


if __name__ == '__main__':
    ss = SimilaritySearch()

    DB_PATH = '/home/qiji/tmp/chroma'  # 数据库路径!

    ss.load_db(DB_PATH)  # 加载数据库!

    for _ in ss.search('丰田C-HR'):
        print(_)
