"""
OVO-Bench 工具函数
包含prompt模板、任务分类等

使用情况:
- BACKWARD_TASKS, REAL_TIME_TASKS, FORWARD_TASKS: 被 mcq_eval.py 使用
- Prompt模板和其他函数: 预留，暂未使用
"""

# 任务分类（被 mcq_eval.py 使用）
BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

# Prompt template for backward-tracing and real-time visual perception task
BR_PROMPT_TEMPLATE = """
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C). 
Do not include any additional text or explanation in your response.
"""

# Prompt template for REC task
REC_PROMPT_TEMPLATE = """
You're watching a video in which people may perform a certain type of action repetively. 
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one. 
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

# Prompt template for SSR task
SSR_PROMPT_TEMPLATE = """
You're watching a tutorial video which contain a sequential of steps. 
The following is one step from the whole procedures: 
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

# Prompt template for CRR task
CRR_PROMPT_TEMPLATE = """
You're responsible of answering questions based on the video content. 
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

def build_ovo_bench_prompt(task, question, options, _anno_, index):
    """
    构建OVO-Bench的prompt
    Args:
        task: 任务类型
        question: 问题（对于Backward/Realtime任务）
        options: 选项列表（对于Backward/Realtime任务）
        _anno_: 标注信息（对于Forward任务）
        index: 索引（对于Forward任务）
    Returns:
        prompt: 构建好的prompt字符串
    """
    if task in ["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"]:
        # Backward和Realtime任务：多选题
        formatted_options = '; '.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(options)) + ';'
        prompt = BR_PROMPT_TEMPLATE.format(question, formatted_options)
        
    elif task == "REC":
        # REC任务：计数任务
        activity = _anno_["activity"]
        question = "How many times did they " + activity + "?"
        prompt = REC_PROMPT_TEMPLATE.format(question)
        
    elif task == "SSR":
        # SSR任务：步骤识别任务
        # 优先使用官方方式：从test_info[index]获取step
        if _anno_ and 'test_info' in _anno_ and index is not None and index < len(_anno_["test_info"]):
            step = _anno_["test_info"][index]["step"]
        elif _anno_ and 'current_test_info' in _anno_:
            # 备选方案：从current_test_info获取step（我们的数据加载方式）
            step = _anno_["current_test_info"]["step"]
        else:
            # 兜底方案
            step = "unknown step"
        prompt = SSR_PROMPT_TEMPLATE.format(step)
        
    elif task == "CRR":
        # CRR任务：上下文推理任务
        question = _anno_["question"]
        prompt = CRR_PROMPT_TEMPLATE.format(question)    
    return prompt

def get_task_group(task):
    """
    获取任务所属的组
    Args:
        task: 任务类型
    Returns:
        group: 任务组名称 ('backward', 'realtime', 'forward')
    """
    if task in BACKWARD_TASKS:
        return 'backward'
    elif task in REAL_TIME_TASKS:
        return 'realtime'
    elif task in FORWARD_TASKS:
        return 'forward'
    else:
        return 'unknown'

def get_all_tasks():
    """
    获取所有任务列表
    Returns:
        tasks: 所有任务的列表
    """
    return BACKWARD_TASKS + REAL_TIME_TASKS + FORWARD_TASKS 