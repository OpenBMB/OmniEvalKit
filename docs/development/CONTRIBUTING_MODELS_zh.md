# 如何贡献一个新模型

本指南详细说明了如何将一个新的模型接入到 Omni-Eval Kit 框架中。

## 设计原则

框架的设计核心是**灵活性**：
- **动态分发**：通过 `--generate_method` 参数动态调用模型的不同生成方法
- **多样化接口**：支持批处理、流式、双工等多种推理模式
- **模块化管理**：模型加载逻辑与推理逻辑分离

## 核心步骤

### 1. 注册模型参数

首先，在 `o_e_Kit/utils/args/model_args.py` 中添加您的模型相关参数（如需要）。

### 2. 创建模型封装文件

在 `o_e_Kit/models/` 目录下，为您的模型创建新的目录和文件：
```
o_e_Kit/models/
└── my_model/
    ├── __init__.py
    └── my_model.py
```

### 3. 实现模型封装类

在模型文件中定义封装类，包含初始化和生成方法：

```python
class MyModel:
    def __init__(self, model_path, device, **kwargs):
        """
        初始化模型
        
        Args:
            model_path: 模型权重路径
            device: 运行设备 (cuda/cpu)
            **kwargs: 其他模型配置参数
        """
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.to(device)
        self.device = device
        
        # 加载 tokenizer/processor
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 其他初始化
        self.model.eval()
    ```

### 4. 实现生成方法

根据模型功能，实现一个或多个生成方法。方法名称应与 `--generate_method` 参数值对应：

```python
def generate_batch(self, **batch):
    """
    批量生成方法
    
    Args:
        **batch: 包含输入数据的字典，可能包括：
            - wav_path: 音频文件路径列表
            - question: 问题列表
            - images: 图像数据列表
            等等
    
    Returns:
        list[str]: 模型输出的字符串列表
    """
    # 获取输入数据
    wav_paths = batch.get('wav_path', [])
    questions = batch.get('question', [])
    
    # 执行推理
    outputs = []
    for wav_path, question in zip(wav_paths, questions):
        # 处理输入
        audio = self.load_audio(wav_path)
        
        # 模型推理
        response = self.model.generate(audio, question)
        
        # 后处理
        outputs.append(self.tokenizer.decode(response))
    
    return outputs

def generate_duplex(self, **batch):
    """双工模式生成方法"""
    # 不同的推理逻辑
    return ["response_1", "response_2"]
```

### 5. 注册模型加载器

在 `o_e_Kit/utils/model_loader.py` 中添加模型加载逻辑：

```python
def load_model(args):
    """动态加载模型"""
    model_type = args.model_type
    
    if model_type == "minicpmo":
        from o_e_Kit.models.minicpm.minicpmo import MiniCPM_o
        return MiniCPM_o(args.model_path, args.pt_path, args.device, args.config_path)
    
    elif model_type == "my_model":  # 添加您的模型
        from o_e_Kit.models.my_model.my_model import MyModel
        return MyModel(args.model_path, args.device, **vars(args))
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### 6. 注册推理方法

在 `o_e_Kit/utils/infer.py` 的 `run_inference` 函数中，确保您的生成方法能被调用：

```python
def run_inference(model, dataloader, args):
    # ... 
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if args.generate_method == "generate_batch":
                outputs = model.generate_batch(**batch)
            elif args.generate_method == "generate_duplex":
                outputs = model.generate_duplex(**batch)
            # 添加您的方法
            elif args.generate_method == "my_custom_method":
                outputs = model.my_custom_method(**batch)
            else:
                raise ValueError(f"Unknown generate method: {args.generate_method}")
```

### 7. 运行评估

现在您可以使用新模型运行评估：

```bash
python eval_main.py \
    --model_type my_model \
    --model_path /path/to/model \
    --generate_method generate_batch \
    --eval_gigaspeech_test
```

## 最佳实践

1. **错误处理**：在生成方法中添加适当的错误处理
2. **批处理优化**：充分利用批处理提高推理效率
3. **内存管理**：注意清理不需要的中间变量
4. **日志记录**：添加适当的日志便于调试

## 示例：集成现有模型

以 MiniCPM-O 为例，展示完整的集成流程：

1. 模型结构组织在 `o_e_Kit/models/minicpm/` 下
2. 通过 `minicpmo.py` 实现统一的评测模型封装
3. 支持多种生成方法（`generate_batch`、`generate_chat`、`generate` 等）
4. 通过命令行参数灵活选择模型和生成方法 