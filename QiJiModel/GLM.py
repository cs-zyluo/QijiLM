from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Optional
from typing import List, Optional

from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig


# from torch.mps import empty_cache


class GLM(LLM):
    max_token: int = 4096
    temperature: float = 0.2
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024

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

    # DEBUG:我怀疑这个方法现在叫做__call__,但是我不确定(closed)
    # 最新版本是_call
    def _call(self, prompt: str, history: List[str] = [], stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
            self.tokenizer, prompt,
            history=history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token, temperature=self.temperature,
            top_p=self.top_p)
        return response

    # # DEBUG:同上
    # def __call__(self, prompt: str, history: List[str] = [], stop: Optional[List[str]] = None) -> str:
    #     response, _ = self.model.chat(
    #         self.tokenizer, prompt,
    #         history=history[-self.history_len:] if self.history_len > 0 else [],
    #         max_length=self.max_token, temperature=self.temperature,
    #         top_p=self.top_p)
    #     return response

    # 大佬快来写一下流式输出！！！


if __name__ == '__main__':
    MODEL_PATH = "/home/qiji/chatglm2-6b/"
    llm = GLM()
    llm.load_model(model_name_or_path=MODEL_PATH)
