import os
import torch
import json
import itertools
from typing import Dict, List, Any, Optional, Union, Tuple
from o_e_Kit.utils.logger.simple_progress import smart_progress
from o_e_Kit.utils.eval import evaluate_dataset
from o_e_Kit.utils.dataloader import create_dataloader
import time as time_module
import torch.distributed
from concurrent.futures import ThreadPoolExecutor, Future
import threading

# ============== 异步评估模块 ==============
# 全局评估线程池（单线程，保证评估顺序执行）
_eval_executor: Optional[ThreadPoolExecutor] = None
_eval_futures: List[Tuple[str, Future]] = []
_eval_lock = threading.Lock()


def get_eval_executor() -> ThreadPoolExecutor:
    """获取或创建评估线程池（懒加载单例）"""
    global _eval_executor
    if _eval_executor is None:
        _eval_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="async_eval")
    return _eval_executor


def submit_async_evaluate(answer_file_path: str, dataset_name: str) -> Future:
    """
    异步提交评估任务到线程池
    
    Args:
        answer_file_path: 推理结果文件路径
        dataset_name: 数据集名称
    
    Returns:
        Future 对象，可用于获取评估结果
    """
    executor = get_eval_executor()
    future = executor.submit(evaluate_dataset, answer_file_path=answer_file_path, dataset_name=dataset_name)
    with _eval_lock:
        _eval_futures.append((dataset_name, future))
    print(f"📤 已提交异步评估任务: {dataset_name}")
    return future


def wait_all_evaluations(timeout: float = 7200.0) -> Dict[str, Optional[float]]:
    """
    等待所有异步评估任务完成并收集结果
    
    Args:
        timeout: 单个评估任务的最长等待时间（秒），默认2小时
    
    Returns:
        字典，key 为数据集名称，value 为评估分数（失败时为 None）
    """
    global _eval_futures
    results: Dict[str, Optional[float]] = {}
    
    with _eval_lock:
        futures_copy = _eval_futures.copy()
        _eval_futures = []
    
    if not futures_copy:
        print("📭 没有待处理的异步评估任务")
        return results
    
    print(f"\n⏳ 开始等待 {len(futures_copy)} 个异步评估任务完成...")
    
    for dataset_name, future in futures_copy:
        try:
            score = future.result(timeout=timeout)
            results[dataset_name] = score
            print(f"✅ {dataset_name} 评估完成，分数: {score}")
        except TimeoutError:
            print(f"⏰ {dataset_name} 评估超时（超过 {timeout} 秒）")
            results[dataset_name] = None
        except Exception as e:
            print(f"❌ {dataset_name} 评估失败: {e}")
            results[dataset_name] = None
    
    print(f"🎉 所有异步评估任务已完成，共 {len(results)} 个数据集")
    return results


def shutdown_eval_executor():
    """关闭评估线程池（程序退出时调用）"""
    global _eval_executor
    if _eval_executor is not None:
        _eval_executor.shutdown(wait=True)
        _eval_executor = None
        print("🔒 评估线程池已关闭")


def get_pending_eval_count() -> int:
    """获取当前待处理的评估任务数量"""
    with _eval_lock:
        return len(_eval_futures)
# ============== 异步评估模块结束 ==============

def build_generation_kwargs(batch_data: tuple, dataset_name: str) -> Dict[str, Any]:
    _, paths_list, annotations_list = batch_data
    
    kwargs = {
        'dataset_name': dataset_name,
        'paths': paths_list,
        'items': annotations_list,
    }
    
    # 支持单音频（audio_path）和多音频（audio_path_list）格式
    audio_paths = []
    video_paths = []
    image_paths = []
    
    for path in paths_list:
        if isinstance(path, dict):
            # 检查单音频路径
            if path.get('audio_path'):
                audio_paths.append(path.get('audio_path'))
            # 检查多音频路径列表（MMAU-Pro等数据集）
            elif path.get('audio_path_list'):
                audio_paths.append(path.get('audio_path_list'))  # 保留列表
            # 检查音频路径字典（UNO-Bench, AV-Odyssey）
            elif path.get('audio_paths_dict'):
                audio_paths.append(path.get('audio_paths_dict'))
            
            # 检查视频路径
            if path.get('video_path'):
                video_paths.append(path.get('video_path'))
            elif path.get('video_paths_dict'):
                video_paths.append(path.get('video_paths_dict'))
            
            # 检查图片路径
            if path.get('image_path'):
                image_paths.append(path.get('image_path'))
            elif path.get('image_paths_dict'):
                image_paths.append(path.get('image_paths_dict'))
    
    # 判断模态类型
    has_audio = len(audio_paths) > 0
    has_video = len(video_paths) > 0
    has_image = len(image_paths) > 0
    
    if has_audio and not has_video and not has_image:
        kwargs['modality'] = 'audio'
    elif has_video and not has_audio:
        kwargs['modality'] = 'video'
    elif has_image and not has_audio and not has_video:
        kwargs['modality'] = 'image'
    elif has_audio or has_video or has_image:
        # 多种模态组合都算 omni
        kwargs['modality'] = 'omni'
    else:
        raise ValueError("至少需要提供音频、视频或图片路径")
    
    return kwargs


def run_model_generation(model, generate_method: str, **kwargs) -> List[Union[str, Dict[str, Any]]]:
    if generate_method == "batch":
        if hasattr(model, 'generate_batch'):
            return model.generate_batch(**kwargs)
        else:
            raise NotImplementedError(f"模型 {type(model)} 不支持 generate_batch 方法")
    elif generate_method == "generate":
        if hasattr(model, 'generate'):
            return model.generate(**kwargs)
        else:
            raise NotImplementedError(f"模型 {type(model)} 不支持 generate 方法")
    elif generate_method == "chat":
        if hasattr(model, 'generate_chat'):
            return model.generate_chat(**kwargs)
        else:
            raise NotImplementedError(f"模型 {type(model)} 不支持 generate_chat 方法")
    else:
        raise ValueError(f"Unsupported generate method: {generate_method}")


def run_inference(model, dataloader, dataset_name: str, 
                 generate_method: str = "batch") -> List[Dict[str, Any]]:
    answers = []
    
    for batch in smart_progress(dataloader, desc="Running inference"):
        idx, paths_list, annotations_list = batch
        
        generation_kwargs = build_generation_kwargs(
            batch_data=(idx, paths_list, annotations_list),
            dataset_name=dataset_name,
        )
        
        with torch.no_grad():
            outputs = run_model_generation(model, generate_method, **generation_kwargs)
            
            for i, output in enumerate(outputs):
                # 处理不同方法返回的格式
                if isinstance(output, dict):
                    answer_dict = {
                        'idx': idx[i],
                        'prediction': output.get('response', ''),
                        'annotation': annotations_list[i],
                    }
                    # 可选字段：只有存在时才添加
                    if 'prediction_ctc_data' in output:
                        answer_dict['prediction_ctc_data'] = output['prediction_ctc_data']
                    if 'output_audio_path' in output:
                        answer_dict['output_audio_path'] = output['output_audio_path']
                    if 'sequence' in output:
                        answer_dict['sequence'] = output['sequence']
                    if 'system_prompt' in output:
                        answer_dict['system_prompt'] = output['system_prompt']
                    if 'other' in output:
                        answer_dict['other'] = output['other']
                    if 'error' in output:
                        answer_dict['error'] = output['error']
                    if 'audio_speed' in output:
                        answer_dict['audio_speed'] = output['audio_speed']
                else:
                    # 字符串格式的结果
                    answer_dict = {
                        'idx': idx[i],
                        'prediction': output,
                        'annotation': annotations_list[i],
                    }
                answer_dict['path'] = paths_list[i]
                
                answers.append(answer_dict)
    
    return answers

def infer_and_evaluate(
    model, 
    dataset, 
    model_name, 
    dataset_name, 
    time, 
    answer_path, 
    batch_size: int = 1, 
    generate_method: str = "batch", 
    evaluate: bool = True,
    skip_inference: bool = False,
    async_evaluate: bool = False) -> Optional[float]:
    """
    执行推理和评估
    
    Args:
        model: 模型实例
        dataset: 数据集实例
        model_name: 模型名称
        dataset_name: 数据集名称
        time: 时间戳
        answer_path: 结果保存路径
        batch_size: 批大小
        generate_method: 生成方法 ("batch", "chat", "generate")
        evaluate: 是否执行评估
        skip_inference: 是否跳过推理（使用已有结果）
        async_evaluate: 是否使用异步评估（提交到线程池，不阻塞主线程）
    
    Returns:
        评估分数（async_evaluate=True 时返回 None，需通过 wait_all_evaluations() 获取结果）
    """
    
    # 检查是否已存在推理结果
    answer_dir = os.path.join(answer_path, model_name, time)
    answer_file_path = os.path.join(answer_dir, f"{dataset_name}.json")
    
    if skip_inference and os.path.exists(answer_file_path):
        print(f"📁 发现已存在的推理结果: {answer_file_path}")
        print(f"⏭️  跳过推理，直接进行评估...")
        
        # 读取已存在的predictions
        with open(answer_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'predictions' in data:
                predictions = data['predictions']
            else:
                # 兼容旧格式
                predictions = data
    else:
        # 进行推理
        print(f"🚀 开始推理: {dataset_name}")
        dataloader = create_dataloader(dataset, batch_size=batch_size)
        predictions = run_inference(model, dataloader, dataset_name, generate_method)

    # 分布式环境下：各 rank 独立保存结果，使用文件轮询方式同步
    # 完全避免使用 NCCL collective 操作，防止因推理速度不均匀导致的超时
    if not skip_inference and torch.distributed.is_initialized():
        torch.cuda.empty_cache()
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        
        # 每个 rank 保存自己的结果到临时文件
        answer_dir = os.path.join(answer_path, model_name, time)
        os.makedirs(answer_dir, exist_ok=True)
        
        # 临时文件：dataset_name_rank{rank}.json
        temp_file_path = os.path.join(answer_dir, f"{dataset_name}_rank{rank}.json")
        with open(temp_file_path, "w", encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False)
        print(f"💾 rank {rank} 推理结果已保存到临时文件: {temp_file_path}")
        
        # 只有 rank 0 负责合并所有结果
        if rank == 0:
            # 使用文件轮询等待所有 rank 完成（避免 NCCL barrier 超时）
            all_rank_files = [os.path.join(answer_dir, f"{dataset_name}_rank{r}.json") for r in range(world_size)]
            max_wait_time = 3600  # 最长等待 1 小时
            poll_interval = 60  # 每 60 秒检查一次
            waited_time = 0
            
            while waited_time < max_wait_time:
                all_ready = all(os.path.exists(f) for f in all_rank_files)
                if all_ready:
                    break
                missing = [f for f in all_rank_files if not os.path.exists(f)]
                if waited_time % 60 == 0:  # 每分钟打印一次状态
                    print(f"⏳ rank 0 等待其他 rank 完成... 已等待 {waited_time}s, 缺少: {len(missing)} 个文件")
                time_module.sleep(poll_interval)
                waited_time += poll_interval
            
            if waited_time >= max_wait_time:
                print(f"⚠️ 等待超时（{max_wait_time}s），部分 rank 可能未完成")
            
            # 合并所有结果
            all_predictions: List[Dict[str, Any]] = []
            for r in range(world_size):
                rank_file = os.path.join(answer_dir, f"{dataset_name}_rank{r}.json")
                if os.path.exists(rank_file):
                    with open(rank_file, 'r', encoding='utf-8') as f:
                        rank_preds = json.load(f)
                        if isinstance(rank_preds, list):
                            all_predictions.extend(rank_preds)
                    # 删除临时文件
                    os.remove(rank_file)
            predictions = all_predictions
            print(f"📊 rank 0 合并完成，共 {len(predictions)} 条预测结果")
        else:
            # 非 rank 0 直接返回，不参与后续评估
            return None
    elif not skip_inference:
        # 非分布式环境，直接使用 predictions
        pass

    # 只有 rank0 负责保存和评估（分布式环境下其他 rank 已返回）
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    
    # 设置最终结果文件路径
    answer_dir = os.path.join(answer_path, model_name, time)
    os.makedirs(answer_dir, exist_ok=True)
    answer_file_path = os.path.join(answer_dir, f"{dataset_name}.json")
    
    # 如果跳过推理且文件已存在，就不需要再保存
    if not (skip_inference and os.path.exists(answer_file_path)):
        # 合并 job_id 和 predictions 到一个字典中
        job_id = os.getenv('JOB_ID', -1)
        result_data = {
            "job_id": job_id,
            "predictions": predictions,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "timestamp": time
        }
        
        with open(answer_file_path, "w", encoding='utf-8') as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        print(f"💾 推理结果已保存到: {answer_file_path}")
    else:
        print(f"📋 使用已存在的推理结果: {answer_file_path}")

    if evaluate:
        if async_evaluate:
            # 异步评估：提交到线程池后立即返回，不阻塞主线程
            submit_async_evaluate(answer_file_path=answer_file_path, dataset_name=dataset_name)
            return None  # 返回 None，实际结果通过 wait_all_evaluations() 获取
        else:
            # 同步评估：阻塞直到评估完成
            return evaluate_dataset(answer_file_path=answer_file_path, dataset_name=dataset_name)
    else:
        return 0.0