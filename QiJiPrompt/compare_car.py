# xuhe is coding
# -*- coding: utf-8 -*-

from langchain.prompts import PromptTemplate


class CompareCarPrompt:
    def _get_car_names_prompt(self) -> PromptTemplate:
        prompt_template = "你是一名专业的汽车专家,你需要帮助我分析几种车辆之间的优劣,所以，你的第一件事情是`发现我需要进行比较的车辆的名字和型号`\n" \
                          "你可以从我的询问中总结出我需要进行比较的汽车的名字和型号，我的询问是`{question}`,\n" \
                          "输出格式为:比较的车辆的名字和型号:`名字,名字`,\n" \
                          "* 例如,一个询问是`我是买宝马x1好，还是买奔驰c200好`的结果是\"\"\"比较的车辆的名字和型号:`宝马x1,奔驰c200`\"\"\"\n" \
                          "* 请注意,这是你做的第一件事情,所以,你只需要`发现`并且`告诉我`我需要比较的汽车的名字和型号就可以了,不 需要进行比较"
        prompt = PromptTemplate(
            input_variables=["question"],
            template=prompt_template
        )
        return prompt

    def _get_car_attributes_prompt(self) -> PromptTemplate:
        prompt_template = "你是一名专业的汽车专家,你需要帮助我分析几种车辆之间的优劣,现在你要完成你的第二个任务,根据之前获取的需要比较的车辆的名字和型号\n" \
                          "你需要列出几个比较的属性,需要比较的车的名字和型号的信息是`{car_names}`,\n" \
                          "输出格式为:比较的车辆名字和型号:`名字,名字\n" \
                          "比较的属性1,比较的属性2,比较的属性3,...`"
        print(prompt_template)
        prompt = PromptTemplate.from_template(prompt_template)
        return prompt


if __name__ == '__main__':
    from QiJiModel.GLM import GLM

    MODEL_PATH = "/home/qiji/chatglm2-6b/"  # 模型路径
    llm = GLM()
    llm.load_model(model_name_or_path=MODEL_PATH)

    llm.STREAMING_OUTPUT = True  # 设置为True,使用流式输出,可以提高生成速度,但是会牺牲一定的生成质量

    from langchain.chains import LLMChain
    from langchain.chains import SequentialChain

    chain1 = LLMChain(llm=llm, prompt=CompareCarPrompt()._get_car_names_prompt(), output_key="car_names")
    chain2 = LLMChain(llm=llm, prompt=CompareCarPrompt()._get_car_attributes_prompt(), output_key="car_attributes")

    chain = SequentialChain(
        chains=[chain1, chain2],
        input_variables=["question"],
        output_variables=["car_attributes"],
        verbose=True
    )

    # print(chain.run(question="雷克萨斯RX和起亚K9哪个好"))
    index = 0
    while index<2:
        q = input()
        chain.run(question=q)
