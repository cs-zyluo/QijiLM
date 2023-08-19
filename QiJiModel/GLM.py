import os
import sys

from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional
from typing import List, Optional

from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig


class GLM(LLM):
    max_token: int = 4096
    temperature: float = 0.2
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024

    # author:xuhe
    # 我要在这里设置一个常量,这个常量标志是否使用流式输出
    # 注意,使用流式输出的时候,你不能自己print输出,它会自动输出的,但是你可以从run方法中获取返回值
    STREAMING_OUTPUT: bool = False

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "GLM"

    def load_model(self, llm_device="gpu", model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config,
                                               trust_remote_code=True).half().cuda()

    # BUG:我怀疑这个方法现在叫做__call__,但是我不确定(closed)
    # 最新版本是_call
    def _call(self, prompt: str, history: List[str] = [], stop: Optional[List[str]] = None):
        # 让这段最初始的代码留着吧,我不想为此存一个GIT了
        # response = ""  # 这个变量是用来存储返回的结果的
        # if not self.STREAMING_OUTPUT:
        #     response, _ = self.model.chat(
        #         self.tokenizer, prompt,
        #         history=history[-self.history_len:] if self.history_len > 0 else [],
        #         max_length=self.max_token, temperature=self.temperature,
        #         top_p=self.top_p)
        # else:
        #     # 使用流式输出
        #     # wait for test
        #     # 这两个代码可以合并减少代码量
        #     for response, _ in self.model.stream_chat(
        #             self.tokenizer, prompt,
        #             history=history[-self.history_len:] if self.history_len > 0 else [],
        #             max_length=self.max_token, temperature=self.temperature,
        #             top_p=self.top_p):
        #         os.system('clear')  # 清屏
        #         print(response, flush=True)  # 输出

        # 以下是修改之后的更易懂的代码
        response = ""  # 这个变量是用来存储返回的结果的
        pre_response = ""  # 这个变量是用来存储上一次的返回结果的
        for response, _ in self.model.stream_chat(
                self.tokenizer, prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                max_length=self.max_token, temperature=self.temperature,
                top_p=self.top_p):
            if self.STREAMING_OUTPUT:  # 如果使用流式输出
                sys.stdout.write(str(response)[len(pre_response):])  # 输出 response 字符串,去掉上一次的 response
                sys.stdout.flush()  # 立即刷新到标准输出
                pre_response = str(response)  # 更新 pre_response
        if self.STREAMING_OUTPUT:  # 如果使用流式输出
            print()  # 输出一个换行符,因为上面的代码没有输出换行符

        return response


if __name__ == '__main__':
    MODEL_PATH = "/home/qiji/chatglm2-6b/"
    llm = GLM()
    llm.load_model(model_name_or_path=MODEL_PATH)

    llm.STREAMING_OUTPUT = True  # 注意!

    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    # test:测试是否可以正常运行回答
    prompt = PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    index = 0
    while index < 2:
        index += 1
        product = input('> ')
        # print(chain.run(product))
        chain.run(product)
