# Embedding模型召回测试

在RAG中使用bge-m3的效果远好于其他两个。

qwen3的0.6B embedding模型略好于bge-m3。



# dify添加模型

如果只是要用 PyTorch（不是自己编译），完全不需要手动安装 CUDA Toolkit 和 cuDNN

魔搭社区默认下载地址：`~/.cache/modelscope/hub`。

关闭代理`proxy_off`，设置为魔搭环境`export VLLM_USE_MODELSCOPE=True`

## 启动模型

使用vllm 启动模型：

```python
CUDA_VISIBLE_DEVICES=3 vllm serve /home/os/.cache/modelscope/hub/models/Embedding-GGUF/bge-m3-Q4_K_M-GGUF/bge-m3-q4_k_m.gguf --port 9580 --api-key vl-5bgrMOCJ5OSBKQV5XbHz --served-model-name=bge-m3-Q4_K_M-GGUF --load-format gguf

CUDA_VISIBLE_DEVICES=3 vllm serve /home/os/.cache/modelscope/hub/models/mlx-community/Qwen3-Embedding-4B-4bit-DWQ/ --port 9544 --api-key vl-5bgrMOCJ5OSBKQV5XbHz --served-model-name=Qwen3-Embedding-4B-4bit-DWQ

CUDA_VISIBLE_DEVICES=3 vllm serve /home/os/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct-AWQ/ --port 8811 --api-key vl-5bgrMOCJ5OSBKQV5XbHz --served-model-name=Qwen/Qwen2.5-VL-3B-Instruct-AWQ

CUDA_VISIBLE_DEVICES=3 vllm serve /home/os/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct-AWQ  --served-model-name=Qwen/Qwen2.5-VL-7B-Instruct-AWQ --max-model-len 8192 --max-num-seqs 16 --port 8811 --gpu-memory-utilization 0.6

---下面的命令需要在指定目录下运行.cache/modelscope/hub/models
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-VL-3B-Instruct-AWQ  --max-model-len 8192   --gpu-memory-utilization 0.5 --max-num-seqs 64 --port 8811

CUDA_VISIBLE_DEVICES=3 vllm serve /home/os/.cache/modelscope/hub/models/Qwen/Qwen3-14B-AWQ/ --port 8000 --served-model-name=Qwen/Qwen3-14B-AWQ --gpu-memory-utilization 0.8
CUDA_VISIBLE_DEVICES=3 vllm serve /home/os/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B/ --port 9596 --api-key vl-5bgrMOCJ5OSBKQV5XbHz --served-model-name=Qwen/Qwen3-Embedding-0.6B --gpu-memory-utilization 0.2

---多卡运行
CUDA_VISIBLE_DEVICES=2,3 vllm serve /home/os/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-3B-Instruct-AWQ  --served-model-name=Qwen/Qwen2.5-VL-3B-Instruct-AWQ --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 64 --port 8811 --gpu-memory-utilization 0.2

CUDA_VISIBLE_DEVICES=2,3 vllm serve /home/os/.cache/modelscope/hub/models/Qwen/Qwen2.5-VL-7B-Instruct-AWQ  --served-model-name=Qwen/Qwen2.5-VL-7B-Instruct-AWQ --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 64 --port 8811 --gpu-memory-utilization 0.35

CUDA_VISIBLE_DEVICES=2,3 vllm serve /home/os/.cache/modelscope/hub/models/Qwen/Qwen3-32B-AWQ/ --served-model-name=Qwen/Qwen3-32B-AWQ --port 39832 --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.7

LD_LIBRARY_PATH=/usr/local/cuda/lib64 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 CUDA_VISIBLE_DEVICES=2,3 vllm serve /home/os/.cache/modelscope/hub/models/openai-mirror/gpt-oss-20b/ --served-model-name=openai-mirror/gpt-oss-20b --port 39820 --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.7

CUDA_VISIBLE_DEVICES=2,3 vllm serve /home/os/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B/ --port 9596 --api-key vl-5bgrMOCJ5OSBKQV5XbHz --served-model-name=Qwen/Qwen3-Embedding-0.6B --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.01

CUDA_VISIBLE_DEVICES=3 vllm serve /home/os/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-0.6B/ --port 9596 --api-key vl-5bgrMOCJ5OSBKQV5XbHz --served-model-name=Qwen/Qwen3-Embedding-0.6B --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.1

CUDA_VISIBLE_DEVICES=0,1 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_MODELSCOPE=true vllm serve /models/gpt-oss-20b/ --served-model-name=gpt-oss-20b --port 39820 --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.7

VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_MODELSCOPE=true vllm serve /models/gpt-oss-20b/ --served-model-name gpt-oss-20b --trust_remote_code --port 8801
CUDA_VISIBLE_DEVICES=0,3 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_MODELSCOPE=true vllm serve /models/gpt-oss-20b/ --served-model-name=gpt-oss-20b --port 39820 --max-model-len 8192 --max-num-seqs 64 
```

详细参数可以查看：https://docs.vllm.com.cn/en/latest/serving/engine_args.html

~~`--gpu-memory-utilization`是在**每张卡**的**总显存上设置一个可使用率**，VLLM这个进程在每张卡上被允许使用的、包含了一切开销的、总的显存预算上限，包括模型权重、框架开销和kv cache占用空间。~~

**vLLM的** **gpu-memory-utilization** **参数控制的仅仅是KV缓存池的大小，它是在模型权重加载之后，从剩余的显存中去申请的**

模型权重的计算，以32B计算，如果是单精度FP32的，那么就是32B\*32/8=128GB，如果是int4的，那么就是32B\*4/8=16GB。



`--max-num-seqs 64` 是一个核心的资源和性能控制参数，它定义了你的VLLM服务的最大并发处理能力，直接影响显存的规划和系统的吞吐量。默认是256，对于聊天一般设置小批次，如16，32。

1. **什么是序列 (Sequence)？**
   在 VLLM 的语境下，一个序列就是**一个独立的、需要生成文本的输入**。通常情况下，一个API请求就包含一个序列。但 VLLM 也支持一个请求里包含多个序列（比如"prompt": ["你好，", "今天天气"]），这被称为“批处理请求”。max-num-seqs 控制的是所有请求加起来的总序列数。
2. **它如何影响资源（尤其是KV缓存）？**
   - 
   - **每个序列都需要KV缓存**：只要一个序列在处理中（不管是正在计算prompt还是在逐字生成token），它就必须在显存中保留自己的KV缓存。
   - **座位越多，需要的空间越大**：max-num-seqs 的值越大，意味着VLLM需要准备好容纳更多序列同时存在的KV缓存，因此在启动时就需要预留更大的显存池。
3. **它如何影响性能？**
   - **提高并发和吞吐量**：设置一个较高的 max-num-seqs 可以让VLLM在同一时间“批处理”更多的请求。GPU是一种并行计算设备，一次处理64个请求的效率远高于一次处理1个请求重复64次。这能极大地**提高系统的总吞吐量**（每秒处理的token数）。
   - **可能增加延迟**：如果并发数非常高，你的GPU可能需要一点时间来处理完整个批次。对于单个请求来说，它的等待时间（延迟）可能会略微增加，因为它需要等同批次的其他请求一起处理完

实际测试基本没啥影响，批次从默认的256降低到16，一共少了不到600M的显存。



客户端测试：

```
curl http://192.168.0.131:9506/v1/embeddings -H "Authorization: Bearer vl-5bgrMOCJ5OSBKQV5XbHz" -H "Content-Type: application/json" -d "{\"input\": \"The food was delicious and the waiter...\", \"model\": \"Qwen/Qwen3-Embedding-0.6B\", \"encoding_format\": \"float\" }"
```

vllm里设置了`--served-model-name=Qwen/Qwen3-Embedding-4B`，在客户端的model中就可以使用这个名称去找到名字，包括dify的名称设置。



ollama的模型格式是gguf的，vllm现在支持gguf但是没支持全，如果该模型是基于transform架构的，vllm可以运行，如果不是就不能直接运行ollama的模型。比如ollama的bge-m3就无法运行。

# 模型量化

mlx-community/Qwen3-Embedding-4B-4bit-DWQ

- **DWQ**: 这个后缀代表 **Dynamic Weight Quantization**。这是一种比较新或比较小众的量化技术。
- **mlx-community**: 你提供的模型路径中包含了 mlx-community。**MLX 是 Apple 为其 Apple Silicon (M1/M2/M3 芯片) 设计的机器学习框架**

**AWQ（Activation Weight Quantization）**，对权重和激活值同时量化，可以选择量化4bit,8bit或者16bit。

**W4A16 (4-bit Weight, 16-bit Activation)**，对权重值进行量化，保留激活值为fp16。

**GGUF** 是 **General Graph Universal Format** 的缩写，是一种用于表示神经网络计算图的格式，vllm目前基本上不支持。

**GPTQ** (Generalized Post-Training Quantization)，只对权重进行量化。

# dify添加外部知识库

可点击这里查看外部知识库API：

![image-20250701180605298](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250701180605298.png)

同样可以添加和删除api：

![image-20250701180805265](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250701180805265.png)

以ragflow为例，添加外部新的知识库api，apikey可以去ragflow里直接申请，但是api endpoint需要填`http://127.0.0.1:9998/api/v1/dify`，可以换成服务器对应ip。

![image-20250701181012009](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250701181012009.png)

至于知识库id，在ragflow中需要点进知识库，然后在网址栏中可以直接找到

![image-20250701181153048](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250701181153048.png)

![image-20250701181333288](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250701181333288.png)

# Linux命令

查看所有端口号：`ss -tuln`

`-t`：显示 TCP 端口

`-u`：显示 UDP 端口

`-l`：只显示监听状态（LISTEN）

`-n`：不解析域名（显示数字 IP 和端口）

查看指定端口号`ss -tuln | grep :8000`;`lsof -i :8000`



使用`tmux new -s name`创建可后台执行的终端窗口，进入后重新执行程序。

进入tmux窗口后，按`Ctrl + b，然后按 d`可以退出但不终止程序，或者执行`tmux detach`也可以，后续任何时候可以使用`tmux attach -t name`进入。

`tmux ls`查看所有会话。

在会话中运行 `exit`就会退出并关闭会话。





http://192.168.0.131:9000/ragflow-image-assets/R-C.jpg

# Word导入RAGFlow问题

word导入ragflow解析时可能会报错

![image-20250708101542512](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250708101542512.png)

这种情况是word中有为空的图片

![image-20250708101953857](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250708101953857.png)

将这个图片删掉就好。

但是实际情况是，这个图你可能根本找不到在哪，如下图，在你没点到它之前是不会显示的

![image-20250708102058271](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250708102058271.png)

对于这种情况可以将word作为压缩文件打开，找到这个路径下的文件

![image-20250708102220029](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250708102220029.png)

打开以后发现里面有为NULL的图片

![image-20250708102337568](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250708102337568.png)

但还是需要**在word中找到这个图然后删掉**才行，不能在直接删掉这一行，会出问题。



# dify修改上传文件大小配置

找到dify/docker/.env文件，对其中的配置进行修改

```python
# 知识库文件上传大小限制
UPLOAD_FILE_SIZE_LIMIT=1024
UPLOAD_FILE_BATCH_LIMIT=50
# 多模态 上传图片、视频、音频大小限制
UPLOAD_IMAGE_FILE_SIZE_LIMIT=1000
UPLOAD_VIDEO_FILE_SIZE_LIMIT=10000
UPLOAD_AUDIO_FILE_SIZE_LIMIT=50
# 调整总容量大小
NGINX_CLIENT_MAX_BODY_SIZE=150000M
```

详细解释可看https://zhuanlan.zhihu.com/p/26611604776。



修改完成后使用`docker compose down`和`docker compose up -d`重新启动docker



# RAG对于word的手动分块

dify中父子分块的逻辑是按照从上到下的顺序去寻找父分块，先按照父分块的分段标识符进行分段，如果超过了父分块的最大长度就分成相应数量；如果父分块中存在子分块的分段标识符，会按照子段标识符分成子块，如果没有就按照子段最大长度进行分段。



所以基于此，设想的方案是：

父块之间用======做间隔，子块之间用------做间隔；

标题格式存在以下几种：第一章，第二章/1.2.5，2.1/##3.5，#第八章，等形式，最后一种形式只以markdown格式进行判断，如果不是#开头的那么就不是标题；标题格式使用单独的函数或者正则表达式决定；

先进行一次初步分块，按照最小标题进行分块，将最小标题及内容全部分为子块，然后向上追溯，将最小标题的上一级标题作为父块，父块包含上一级标题与内容，及其子标题下的所有内容，父块内容与子块之间也需要------做间隔；

若当前标题不存在上一级标题，则该标题与内容直接变为父块；

对于三级及以上标题嵌套，最小标题的上一级作为父块会被视为上上一级的子块，然后继续循环前面的逻辑直到处理整个文档；

初步分块以后继续处理：

设置子块变量，规定子块内容最大字符数量，当超过该数量时子块会进行分裂；

设置父块变量，规定父块内容最大字符数量，当低于子块变量时变更为子块，当高于父块变量时父块进行分裂；

父块进行分裂时，若父块内容中存在子块，且子块字数不超过父块字数，那么按照子块的分隔去分父块，例如：父块限制500字，子块限制300字：子块1：300字；子块2：150字；子块3：200字；那么就将子块2和子块3合并为一个新的父块，子块1留在原本父块中；

表格、图片（链接）、代码块不进行分裂，即使超过了子块或父块的限制字数也需要保持为一个子块或者父块，根据其本身的字数决定是子块还是父块；

需要对表格进行额外处理：对于html格式的表格需要转成markdown格式，表格里的内容如果有图片格式，包括\<img>标签以及md的图片格式!\[](src)，都需要转成!\[image](BASE_URL)的格式；

不是表格中的图片格式也需要按照上述要求进行修改；

增加LLM分块函数，可以选择是否启用，该函数的作用是将包含子块的父块内容发送给大模型，让大模型判断子块之间是否存在逻辑关系，例如父块标题是：数据库配置，子块分别是步骤配置1，步骤配置2这种，那么该父块就保持不变；如果子块之间不存在逻辑关系，是不同内容的关系，例如父块标题是：遭遇的其他问题，子块是问题1，问题2这种不同的问题，那么子块需要晋升为父块（将------替换为======）;

# dify 并发配置

访问dify的docker文件夹下的.env配置文件,有下面的配置

## API服务层

`SERVER_WORKER_CLASS`：定义API服务的运行模式，推荐设置`gevent`，这是一个异步I/O模型。

`SERVER_WORKER_AMOUNT`：设置 API 服务（Gunicorn）启动的**工作进程数量**。一般要和服务器核心数挂钩。64

`SERVER_WORKER_CONNECTIONS`：**仅在 SERVER_WORKER_CLASS=gevent 时有效**。它定义了**每个**工作进程内部能同时处理的并发连接数（协程数）。100

## 数据库层

`SQLALCHEMY_POOL_SIZE`：Dify 应用为**每个** API 工作进程（SERVER_WORKER_AMOUNT）创建的数据库连接池的大小。100

`POSTGRES_MAX_CONNECTIONS`：PostgreSQL **数据库服务器自身**所允许的最大并发连接总数。必须 **大于等于** `SERVER_WORKER_AMOUNT × SQLALCHEMY_POOL_SIZE`。6400

## 后端模型层

**Dify Worker 的扩展**

- **作用**: Dify 中负责调用 LLM API 和执行异步任务（如数据集处理）的是 dify-worker 服务。增加其数量可以提高异步任务和 LLM 调用的吞吐量。

- **操作**: 这不在 .env 文件中，而是在命令行通过 docker-compose 实现。

  Generated bash

  ```
  # 启动 4 个 worker 实例
  docker compose up -d --scale worker=4
  ```

- **推荐值**: 通常可以设置为与 SERVER_WORKER_AMOUNT 接近或稍高的值，例如 4 到 16 不等，取决于你的 LLM 服务处理能力。

**LLM 推理服务自身的并发配置 (以 vLLM 为例)**

- **作用**: 如果你是自托管模型（如使用 vLLM），你需要配置推理服务本身能接受的并发请求数。
- **关键参数**: --max-num-seqs
- **为什么**: 这个参数直接定义了 vLLM 引擎能同时在 GPU 中处理多少个请求序列。如果 Dify 发送了 10 个请求，但 vLLM 这里只设置为 --max-num-seqs 8，那么多出的 2 个请求就必须排队等待。

# Nginx

下载：

```python
sudo apt-get update
sudo apt-get install nginx
```

测试是否安装成功`sudo nginx -t`

启动：

```
# 启动Nginx服务
sudo systemctl start nginx

# 设置Nginx开机自启
sudo systemctl enable nginx
```

检查nginx状态：

```
sudo systemctl status nginx
```

创建新的配置文件：

```
server {
    # 1. 设置你希望对外暴露的端口号
    listen 9978; 

    # 如果你有域名，可以替换成你的域名，否则用IP访问时这个设置可以忽略
    server_name _; 

    # 2. 这是对外暴露的URL路径
    # 例如，访问 http://140.206.180.133:9978/images/1040-image/image-023.jpg
	# 	       http://192.168.0.131:9978/images/1040-image/image-023.jpg
	# 只允许通过图片
    location ~* ^/images/.*\.(jpg|jpeg|png)$ {

		# 使用 rewrite 重写 URL
        #    ^/images/(.*)$  -> 捕获 /images/ 后面的所有内容 (例如 "1040-image/image-023.jpg")
        #    /ragflow-image-assets/$1  -> 将捕获的内容拼接到新的路径后面
        #    break  -> 停止处理其他 rewrite 规则
        rewrite ^/images/(.*)$ /ragflow-image-assets/$1 break;
		
        # proxy_pass 只包含上游服务器地址，不带URI
        proxy_pass http://192.168.0.131:9000;
		
        # --- 以下是推荐的代理设置，可以保证代理的稳定性和兼容性 ---
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 允许MinIO的响应头通过
        proxy_hide_header "x-amz-id-2";
        proxy_hide_header "x-amz-request-id";
    }
	# --- 新增的配置块 ---
    # 这个规则会捕获所有其他不匹配的请求
    location / {
        return 403; 
    }
}
```

测试配置是否有语法错误：`sudo nginx -t`，显示syntax is ok和test is successful即为成功。

创建符号链接（相当于配置一个快捷方式）

```hxml
sudo ln -s /home/os/36ssd/lxl/config_list/minio-image-proxy.conf /etc/nginx/sites-enabled/
# 删除符号链接
rm /home/os/36ssd/lxl/config_list/minio-image-proxy.conf
# 查看指定目录下的软连接
ls -l /path/to/directory
```

重新加载nginx

```
sudo systemctl reload nginx
```

# Docker

```python
# 在指定目录下终止和启动docker服务
docker-compose down 
docker-compose up -d

openai-mirror/gpt-oss-20b

# 使用当前目录下的Dockerfile文件构建一个名为gpt-oss-fixed的新镜像，若已存在则覆盖
docker build -t gpt-oss-fixed .
# 执行当前目录下的docker-compose.yml文件
docker compose up -d 
# 查看日志
docker compose logs -f
# 导出日志
docker logs <您的容器名> > <您想要的文件名>.log 
docker compose logs --no-color gpt-oss-service > gpt_oss_logs.txt


docker run -d --name gptoss-docker-container \
--gpus all \
vllm/vllm-openai:gptoss \
tail -f /dev/null

docker exec -it gptoss-docker-container /bin/bash


docker run -d --gpus '"device=2,3"' \
    --ipc=host \
    -p 39820:39820 \
    -v /home/os/.cache/modelscope/hub/models/openai-mirror/gpt-oss-20b/:/models/gpt-oss-20b \
    --name gptoss \
    vllm/vllm-openai:gptoss \
    --model /models/gpt-oss-20b \
    --served-model-name=openai-mirror/gpt-oss-20b \
    --port 39820 \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.7
docker logs -f -t gptoss > test.log

docker run --gpus '"device=2,3"' \
    -p 39820:39820 \
	-v /home/os/.cache/modelscope/hub/models/openai-mirror/gpt-oss-20b/:/models/gpt-oss-20b \
    --ipc=host \
    vllm/vllm-openai:gptoss \
    --model /models/gpt-oss-20b
```

## Docker如何构建镜像

先构建Dockerfile文件：

```
# Step 1: 使用基于 Ubuntu 24.04 的基础镜像
# ！！！重要提醒：请先确认 nvidia/cuda:12.8.1-devel-ubuntu24.04 这个标签确实存在于 Docker Hub 上
# 如果不存在，构建会失败。您可能需要选择一个已有的、更新的标签，例如 nvidia/cuda:12.5.1-devel-ubuntu22.04
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# Step 2: 设置环境变量和时区
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Step 3: 安装依赖
# 在 Ubuntu 24.04 中，python3.12 是默认提供的，所以我们不再需要 PPA
# 直接安装即可。
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: 配置 Python (在24.04中，此步骤通常是可选的)
# Ubuntu 24.04 的 `python3` 命令默认就应该指向 `python3.12`
# 但为保险起见，明确设置一下可以保证行为的一致性
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && update-alternatives --set python3 /usr/bin/python3.12

# Step 5: 升级 pip 并安装 vLLM
RUN pip install --upgrade --break-system-packages pip vllm

# Step 6: 设置工作目录
WORKDIR /app
```

**然后执行`docker build -t [image-name] .`构建镜像**,`.`代表的是当前目录。

镜像构建完毕以后执行下述命令运行容器：

```bash
# 挂载完磁盘以后进入容器的命令行手动输入
docker run -it --rm --gpus '"device=2,3"' \
    -v /home/os/.cache/modelscope/hub/models/openai-mirror/gpt-oss-20b/:/models/gpt-oss-20b \
    my-vllm-gptoss:latest \
    /bin/bash

CUDA_VISIBLE_DEVICES=0,1 VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 VLLM_USE_MODELSCOPE=true vllm serve /models/gpt-oss-20b/ --served-model-name=gpt-oss-20b --port 39820 --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 64 --gpu-memory-utilization 0.7

# 直接使用docker运行容器
docker run --rm -it \
    --gpus '"device=0,3"' \
    --ipc=host \
    -p 39820:39820 \
    -v /home/os/.cache/modelscope/hub/models/openai-mirror/gpt-oss-20b/:/models/gpt-oss-20b \
    -e VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1 \
    -e VLLM_USE_MODELSCOPE=true \
    --name my-final-gptoss-service \
    gpt-oss:latest \
    vllm serve /models/gpt-oss-20b/ \
    --served-model-name=gpt-oss-20b \
    --port 39820 \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.7 
```

- **-it**: 启动一个交互式的终端。
- **--rm**: 当您退出容器的 Shell 时，这个容器会自动被删除，非常适合临时任务。
- **my-vllm-gptoss:latest**: 同样，使用您自己的镜像。
- **/bin/bash**: 这个命令覆盖了 Dockerfile 的默认行为，告诉 Docker 在容器启动后直接运行一个 Bash Shell。

进入容器后再执行启动程序的命令。



或者创建docker-compose.yml文件：

```bash
services:
  # 为您的服务命名
  gpt-oss-service:
    # 指定您最终构建成功的镜像
    image: gpt-oss:latest
    # 给容器起一个固定的名字
    container_name: gptoss-vllm
    # 让容器在 Docker 重启后也能自动启动
    restart: unless-stopped
    
    # 部署和资源配置 (包括 GPU)
    deploy:
      resources:
        reservations:
          devices:
            # 对应 CUDA_VISIBLE_DEVICES=0,3
            - driver: nvidia
              device_ids: ['0', '3']
              capabilities: [gpu]
              
    # 网络和进程间通信
    ipc: host
    ports:
      - "39820:39820"
      
    # 数据卷挂载
    volumes:
      # 请务必将 /path/to/your/models/gpt-oss-20b/ 替换成真实路径
      - /home/os/.cache/modelscope/hub/models/openai-mirror/gpt-oss-20b/:/models/gpt-oss-20b/
      
    # 环境变量
    environment:
      - VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
      - VLLM_USE_MODELSCOPE=true
      
    # 容器启动命令和参数
    command:
      - "vllm"
      - "serve"
      - "/models/gpt-oss-20b/"
      - "--served-model-name=gpt-oss-20b"
      - "--port"
      - "39820"
      - "--max-model-len"
      - "8192"
      - "--max-num-seqs"
      - "64"
      - "--tensor-parallel-size"
      - "2"
      - "--gpu-memory-utilization"
      - "0.5"

```

然后在当前目录执行`docker compose up -d`

- up：根据 docker-compose.yml 文件创建并启动容器。
- -d：detached 模式，让容器在后台运行。
- -f：执行指定路径下的docker-compose.yml文件

查看日志

`docker compose logs -f`

`docker logs -f --tail 0 容器id`，从当下开始刷新日志

停止并移除服务

`docker compose down`

清理悬空镜像

`docker image prune`

# Pytorch

`x.size()==x.shape`:展示张量的形状

`cat()`: 在指定维度上拼接张量

外积：升维，外积像是把两个向量的每一对元素组合起来，形成一个“网格”或“表格”，因此维度升高了。

内积：降维，内积像是把两个向量的对应元素相乘后累加，最终得到一个“汇总”的数值，因此维度降低了



## 广播机制

使用*，是两个张量进行逐元素相乘，并且会进行广播机制将低维张量从左侧开始补1，到两者维度数量相等，然后将维度为1的值向高维值进行补正。

注意：只能补维度为1的，比如

```python
cos = torch.rand(5, 64) 
q = torch.rand(2, 8, 5, 64)
z = q * cos  # 完美匹配
```

q可以与cos相乘，因为cos会被广播机制变为`(1,1,5,64)->(2,8,5,64)`

但如果

```python
cos = torch.rand(5, 1, 64) 
q = torch.rand(2, 8, 5, 64)
z = q * cos  # 完美匹配
```

就不行，因为cos会被广播机制变为`(1,5,1,64)->(2,5,1,64)`，与q的`(2,5,5,64)`不相等，无法相乘，报错。

# Git

## 双远程同步

同时关联两个仓库地址，一个用来拉取，一个用来推送。

### 更改远程地址

upstream用来关联原始的fork来源，origin用来关联自己的目标仓库，可以使用`git remote -v`来查看当前的关联地址

![image-20250815104744086](https://raw.githubusercontent.com/denghuishenmi/note-picture/main/picture/image-20250815104744086.png)

使用`git remote add upstream\origin git地址`就能够修改upstream或者origin的地址。

### 推送代码

先使用`git status`查看当前状态；

然后使用`git add 文件名` 将修改的文件加入暂存区，或者使用`git add .`加入所有修改文件；

使用`git commit -m "修改描述"`进行记录；

最后`git push 远程名 分支名`推送到远程

### 拉取与合并代码

然后可以使用`git fetch upstream`和`git merge upstream/分支名`能够从fork的仓库下载并合并代码，这两句代码等于`git pull upstream 分支名`。但是建议使用前者，因为能够处理冲突后再合并。

默认情况下`git pull`等价于`git pull origin 当前本地分支名`，可以使用`git branch`查看当前分支名，或者使用`git status`查看git的详细信息，包括当前分支，是否存在未提交文件等。

### 删除提交

`git rm <文件名>`会从工作区(磁盘)和暂存区(git 索引)中同步删除文件；

`git rm --cached <文件名>`只从暂存区删除；

`git rm -r <目录名>` -r表示递归删除；`git rm -f <文件名>`如果文件在工作区被修改过，Git 默认不会删除，-f 表示强制删除；

删除以后可以再进行提交，推送，就能删除远程仓库的内容。

# python

**list** → `[]` 包起来的，用下标访问 (`data[0]`)。

**dict** → `{}` 包起来的，用 key 访问 (`data["content"]`)。

JSON → 只是一种数据格式，在 Python 里通常会解析成 dict 或 list。

