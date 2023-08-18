* 使用`驼峰命名法`命名类
* `常量`使用全大写,单词之间用`下划线`分割
* 对于所有的类和它的方法,请编写文档来辅助使用
* 个人的全部`不适合在共有文件操作的`操作请放到`以自己名字命名`的文件夹下面操作,最后的成果放到工程文件夹
* 代码编写中,嵌套层数请不要超过3层(偶尔超过没事)
* 编写函数的时候请指出参数的类型,例如`def func(p:str)->int:`接受一个`str`类型参数返回一个`int`类型的值
* `!!!如果需要写特殊的注释,请放在文件的开头!!!`
* 修改之后提倡加上`自己的名字,什么时候修改了什么部分`,不写也没事

# tips

1. 所有的`模型类`全部放在package`QiJiModel`中
2. 所有的`提示词类`全部放在package`QiJiPrompt`中

* 临时文件一律需要包含`tmp`来区分
* 当你正在或者修改或者编写某个文件的时候,请在文件开头加上`{your name} is coding`,例如`xuhe is coding`

# 仅自己可修改

## 仅自己可修改的文件

> 在文件末尾加上`_{your name}`,方便别人区分
> 例如:`tmp_xuhe.py`就是一个xuhe的临时文件

* 下面是代码模块文档(没写完的别加进来)(修改中的代码请在`class`这个级别的标题边上加上`DEBUG`)

# `QiJiModel`package

> 放`模型类`的地方,这个模型类适用于`langchain`

## `GLM.py`file

> 基本的实现,不能放`checkpoing_path`

### `GLM`class

* 使用方法

```python
from QiJiModel.GLM import GLM

MODEL_PATH = "/home/qiji/chatglm2-6b/"  # 模型路径
llm = GLM()
llm.load_model(model_name_or_path=MODEL_PATH)
# 然后就可以放入`langchain`中使用
```

## `GLM_with_checkpoing.py`file

* 可以使用`checkpoint_path`

### `GLM`class

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from QiJiModel.GLM_with_checkpoint import GLM

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
```

## `text2vec_embedding.py`file

* 使用方法

```python
from QiJiModel import Text2Vec

embedding_model_name = "/home/qiji/text2vec-large-chinese"  # 模型路径
embeddings = Text2Vec(model_name=embedding_model_name)
# 然后就可以放入'langchain' 中使用
```

* 补充使用示例(by xuhe)

```python
from langchain.vectorstores.chroma import Chroma
from QiJiModel import Text2Vec
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

embedding_model_name = "/home/qiji/text2vec-large-chinese"  # 模型路径
embeddings = Text2Vec(model_name=embedding_model_name)
# 然后就可以放入'langchain' 中使用

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('./xuhe/data/car_names.txt').load()  # 读取文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # 实例化用来分割文本的类
documents = text_splitter.split_documents(raw_documents)  # 分割文本
db = Chroma.from_documents(documents, embeddings)  # 将文本转换为向量并存储

query = input('> ')  # 输入查询文本
docs = db.similarity_search(query, k=2)  # 查询，注意`k`是返回的文本数量
print(docs[0].page_content)  # 打印最相似的文本
print(docs[1].page_content)
```

* 介绍
    * embedding可以理解为一个向量转化器，将文字转化为向量，方便模型进行搜索和分类等。

# `QiJiOther`package

* 用来放一些`其他`的东西,不属于`模型类`和`提示词类`的东西

## `SimilaritySearch.py`file

### `SimilaritySearch`class

#### `load_text`加载文本内容

* 相似度匹配

```python
from QiJiOther.SimilaritySearch import SimilaritySearch

ss = SimilaritySearch()  # 实例化
file_path = "./xuhe/data/car_names.txt"
ss.load_text(file_path=file_path, chunk_size=100, chunk_overlap=50)  # 加载文本
```

#### `load_dir`加载一个文件夹下面的全部内容

* 参数介绍
* `def load_dir(self, dir_path: str, loader_cls: FILE_LOADER_TYPE, chunk_size: int = 300, chunk_overlap: int = 100,
                 show_progress: bool = True) -> bool:`
* `dir_path`你需要导入的文件夹的路径
* `loader_cls`你要使用的`loader`的类型,例如`TextLoader`
* `show_progress`是否显示进度条

```python
from QiJiOther.SimilaritySearch import SimilaritySearch
from langchain.document_loaders import TextLoader

ss = SimilaritySearch()  # 实例化
dir_path = "./xuhe/data"
ss.load_dir(dir_path=dir_path, loader_cls=TextLoader)  # 加载文本

res_list = ss.search("丰田C-HR")

for _ in res_list:
    print()
    print(_)
```

#### `search`查找相似文本 

* 返回的是一个`list`,里面是`str`类型的文本
* `def search(self, query: str, top_k: int = 3) -> list[str]:`中的`query`是查询的文本,`top_k`是返回的文本数量

```python
from QiJiOther.SimilaritySearch import SimilaritySearch

ss = SimilaritySearch()  # 实例化
res_list = ss.search("丰田C-HR")

for _ in res_list:
    print()
    print(_)
```



