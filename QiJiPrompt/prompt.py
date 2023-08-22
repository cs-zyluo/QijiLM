# xuhe is coding

from abc import abstractmethod

from langchain.prompts import PromptTemplate


# 是否应该提供run()方法实现直接查找答案的功能？
class Prompt:
    def __init__(self, question: str):  # 初始化时传入问题
        self.question = question

    @abstractmethod
    def __str__(self) -> str:  # 获取提示词的字符串形式
        pass

    @abstractmethod
    def get_prompt(self) -> PromptTemplate:  # 获取提示词的模板形式
        pass
