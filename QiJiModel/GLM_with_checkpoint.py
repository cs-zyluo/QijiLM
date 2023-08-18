import os

import torch
from langchain.llms.base import LLM
from transformers import AutoConfig, AutoModel, AutoTokenizer
from typing import List, Optional


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

    def load_model(self, llm_device="gpu", model_name_or_path: str = None, checkpoint_path: str = None):
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True, pre_seq_len=128)  # 加载配置文件
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)  # 加载分词器

        model = AutoModel.from_pretrained(model_name_or_path, config=config, trust_remote_code=True)
        prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

        # Comment out the following line if you don't use quantization
        model = model.half()
        model = model.cuda()
        model = model.eval()
        self.model = model  # 加载模型

    # DEBUG:我怀疑这个方法现在叫做__call__,但是我不确定
    def _call(self, prompt: str, history: List[str] = [], stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
            self.tokenizer, prompt,
            history=history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token, temperature=self.temperature,
            top_p=self.top_p)
        return response


if __name__ == '__main__':
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    MODEL_PATH = "/home/qiji/chatglm-6b-int8"
    CHECKPOINT_PATH = '/home/qiji/chatglm-6b-int8/ptuning/output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000'

    llm = GLM()
    llm.load_model(model_name_or_path=MODEL_PATH, checkpoint_path=CHECKPOINT_PATH)

    # test:测试是否可以正常运行回答
    prompt = PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    product = input('> ')
    print(chain.run(product))
