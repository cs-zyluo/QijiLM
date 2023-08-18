# xuhe is coding

from abc import abstractmethod

from langchain.prompts import PromptTemplate

# 是否应该提供run()方法实现直接查找答案的功能？
class Prompt:
    def __init__(self):
        pass

    @abstractmethod
    def get_prompt(self, question: str) -> PromptTemplate:
        pass


