"""
评估运行模块
负责运行所有数据集的评估
每个数据集都显式处理，便于理解和维护

支持异步评估模式：通过 async_evaluate 参数控制
- async_evaluate=False: 同步评估（默认，保持向后兼容）
- async_evaluate=True: 异步评估，推理完成后立即返回，评估在后台线程执行
"""

import torch
import json
import os
from o_e_Kit.utils.infer import infer_and_evaluate, wait_all_evaluations, get_pending_eval_count
from o_e_Kit.utils.dataset_loader import load_dataset
from o_e_Kit.utils.model_loader import load_model
from o_e_Kit.utils.evaluation_runner_audio import evaluate_all_audio_datasets


def evaluate_video_datasets(args, model, device, time, async_evaluate: bool = False):
    """评估视频数据集
    
    Args:
        async_evaluate: 是否使用异步评估模式
    """
    result = {}
    
    # StreamingBench评估
    if args.eval_streamingbench:
        for task in args.streamingbench_tasks:
            dataset = load_dataset(args, "StreamingBench", task)
            result_key = f'StreamingBench-{task.upper()}'
            
            # 根据任务类型选择推理方法
            if task == "proactive":
                generate_method = "proactive"
                print(f"  -> 使用proactive推理模式（循环时间判断 + 主动输出）")
            else:
                generate_method = "chat"
                print(f"  -> 使用chat推理模式（标准问答评估）")
            
            result[result_key] = infer_and_evaluate(
                model, dataset, args.model_name, result_key, time, 
                answer_path=args.answer_path, batch_size=args.batchsize, generate_method=generate_method,
                async_evaluate=async_evaluate
            )
            print(f"StreamingBench-{task.upper()}{'推理' if async_evaluate else '评估'}完成")
    
    return result


def evaluate_duplex_datasets(args, device, time, async_evaluate: bool = False):
    """评估Duplex数据集
    
    Args:
        async_evaluate: 是否使用异步评估模式
    """
    result = {}
    
    # LiveSports-3K CC 双工评估 (统一格式)
    if getattr(args, 'eval_livesports3k_cc', False):
        dataset = load_dataset(args, "livesports3k_cc")
        duplex_model = load_model(args, device, duplex_type="video")
        result['livesports3k_cc'] = infer_and_evaluate(
            duplex_model, dataset, args.model_name, "livesports3k_cc", time,
            answer_path=args.answer_path, batch_size=args.batchsize, generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # OV-AVEL: Open-Vocabulary Audio-Visual Event Localization
    if getattr(args, 'eval_ovavel', False):
        dataset = load_dataset(args, "ovavel")
        duplex_model = load_model(args, device, duplex_type="omni")
        result['ovavel'] = infer_and_evaluate(
            duplex_model, dataset, args.model_name, "ovavel", time,
            answer_path=args.answer_path, batch_size=args.batchsize, generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    return result


def evaluate_omni_datasets(args, model, time, async_evaluate: bool = False):
    """评估Omni数据集
    
    Args:
        async_evaluate: 是否使用异步评估模式
    """
    result = {}
    
    if args.eval_livesports3k_cc:
        dataset = load_dataset(args, "livesports3k_cc")
        result['livesports3k_cc'] = infer_and_evaluate(
            model, dataset, args.model_name, "livesports3k_cc", time, 
            answer_path=args.answer_path, batch_size=args.batchsize, generate_method=args.generate_method, evaluate=False,
            async_evaluate=async_evaluate
        )
    
    # Daily-Omni 评估
    if getattr(args, 'eval_daily_omni', False):
        dataset = load_dataset(args, "daily_omni")
        result['daily_omni'] = infer_and_evaluate(
            model, dataset, args.model_name, "daily_omni", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # OmniBench 评估
    if getattr(args, 'eval_omnibench', False):
        dataset = load_dataset(args, "omnibench")
        result['omnibench'] = infer_and_evaluate(
            model, dataset, args.model_name, "omnibench", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # UNO-Bench 评估
    if getattr(args, 'eval_unobench', False):
        dataset = load_dataset(args, "unobench")
        result['unobench'] = infer_and_evaluate(
            model, dataset, args.model_name, "unobench", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )

    # UNO-Bench MCQ 评估
    if getattr(args, 'eval_unobench_mc', False):
        dataset = load_dataset(args, "unobench_mc")
        result['unobench_mc'] = infer_and_evaluate(
            model, dataset, args.model_name, "unobench_mc", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # WorldSense 评估
    if getattr(args, 'eval_worldsense', False):
        dataset = load_dataset(args, "worldsense")
        result['worldsense'] = infer_and_evaluate(
            model, dataset, args.model_name, "worldsense", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AV-Odyssey 评估
    if getattr(args, 'eval_av_odyssey', False):
        dataset = load_dataset(args, "av_odyssey")
        result['av_odyssey'] = infer_and_evaluate(
            model, dataset, args.model_name, "av_odyssey", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # Video-MME 评估
    if getattr(args, 'eval_videomme', False):
        dataset = load_dataset(args, "videomme")
        result['videomme'] = infer_and_evaluate(
            model, dataset, args.model_name, "videomme", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # Video-MME Short 评估（仅 duration=short 的子集）
    if getattr(args, 'eval_videomme_short', False):
        dataset = load_dataset(args, "videomme_short")
        result['videomme_short'] = infer_and_evaluate(
            model, dataset, args.model_name, "videomme_short", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # JointAVBench（音视频联合理解 MCQ）评估
    if getattr(args, 'eval_jointavbench', False):
        dataset = load_dataset(args, "jointavbench")
        result['jointavbench'] = infer_and_evaluate(
            model, dataset, args.model_name, "jointavbench", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # OVO-Bench（统一 Omni JSONL，离线 MCQ/QA）评估
    if getattr(args, 'eval_ovobench', False):
        dataset = load_dataset(args, "ovobench")
        result['ovobench'] = infer_and_evaluate(
            model, dataset, args.model_name, "ovobench", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # StreamingBench-Real（统一 Omni JSONL，离线 MCQ/QA）评估
    if getattr(args, 'eval_streamingbench_real', False):
        dataset = load_dataset(args, "streamingbench_real")
        result['streamingbench_real'] = infer_and_evaluate(
            model, dataset, args.model_name, "streamingbench_real", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # StreamingBench-Omni（统一 Omni JSONL，离线 MCQ/QA）评估
    if getattr(args, 'eval_streamingbench_omni', False):
        dataset = load_dataset(args, "streamingbench_omni")
        result['streamingbench_omni'] = infer_and_evaluate(
            model, dataset, args.model_name, "streamingbench_omni", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # StreamingBench-Omni-Fix（筛选后的 Omni JSONL，仅 4 种 task_type）评估
    if getattr(args, 'eval_streamingbench_omni_fix', False):
        dataset = load_dataset(args, "streamingbench_omni_fix")
        result['streamingbench_omni_fix'] = infer_and_evaluate(
            model, dataset, args.model_name, "streamingbench_omni_fix", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # StreamingBench-SQA（统一 Omni JSONL，离线 MCQ/QA）评估
    if getattr(args, 'eval_streamingbench_sqa', False):
        dataset = load_dataset(args, "streamingbench_sqa")
        result['streamingbench_sqa'] = infer_and_evaluate(
            model, dataset, args.model_name, "streamingbench_sqa", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # Video-Holmes 评估
    if getattr(args, 'eval_video_holmes', False):
        dataset = load_dataset(args, "video_holmes")
        result['video_holmes'] = infer_and_evaluate(
            model, dataset, args.model_name, "video_holmes", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AVUT-Benchmark Human 评估
    if getattr(args, 'eval_avut_benchmark_human', False):
        dataset = load_dataset(args, "avut_benchmark_human")
        result['avut_benchmark_human'] = infer_and_evaluate(
            model, dataset, args.model_name, "avut_benchmark_human", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AVUT-Benchmark Gemini 评估
    if getattr(args, 'eval_avut_benchmark_gemini', False):
        dataset = load_dataset(args, "avut_benchmark_gemini")
        result['avut_benchmark_gemini'] = infer_and_evaluate(
            model, dataset, args.model_name, "avut_benchmark_gemini", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # FutureOmni 评估（未来预测基准）
    if getattr(args, 'eval_futureomni', False):
        dataset = load_dataset(args, "futureomni")
        result['futureomni'] = infer_and_evaluate(
            model, dataset, args.model_name, "futureomni", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AVMeme-Exam Full 评估（音视频 Meme 理解）
    if getattr(args, 'eval_avmeme_full', False):
        dataset = load_dataset(args, "avmeme_full")
        result['avmeme_full'] = infer_and_evaluate(
            model, dataset, args.model_name, "avmeme_full", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    # AVMeme-Exam Main 评估（去除 text_cheat 后的数据集）
    if getattr(args, 'eval_avmeme_main', False):
        dataset = load_dataset(args, "avmeme_main")
        result['avmeme_main'] = infer_and_evaluate(
            model, dataset, args.model_name, "avmeme_main", time,
            answer_path=args.answer_path, batch_size=args.batchsize,
            generate_method=args.generate_method,
            async_evaluate=async_evaluate
        )
    
    return result


def run_all_evaluations(args, model, device, time, async_evaluate: bool = True):
    """运行所有评估任务
    
    Args:
        args: 命令行参数
        model: 模型实例
        device: 设备
        time: 时间戳
        async_evaluate: 是否使用异步评估模式（默认 True）
            - True: 推理完成后立即返回，评估在后台线程执行，所有推理完成后统一等待评估结果
            - False: 同步模式，每个数据集推理后立即评估（可能导致多卡超时）
    
    Returns:
        评估结果字典
    """
    result = {}
    
    if async_evaluate:
        print("\n📌 使用异步评估模式：推理和评估并行执行，避免多卡超时")
    
    # 评估音频数据集（ASR和QA）
    print("\n" + "="*60)
    print("【开始评估音频数据集】")
    print("="*60)
    audio_results = evaluate_all_audio_datasets(args, model, time, async_evaluate=async_evaluate)
    result.update(audio_results)
    if audio_results:
        print(f"完成{len(audio_results)}个音频数据集{'推理' if async_evaluate else '评估'}")
    
    # 评估视频数据集
    print("\n" + "="*60)
    print("【开始评估视频数据集】")
    print("="*60)
    video_results = evaluate_video_datasets(args, model, device, time, async_evaluate=async_evaluate)
    result.update(video_results)
    if video_results:
        print(f"完成{len(video_results)}个视频数据集{'推理' if async_evaluate else '评估'}")
    
    # 评估Duplex数据集
    print("\n" + "="*60)
    print("【开始评估Duplex数据集】")
    print("="*60)
    duplex_results = evaluate_duplex_datasets(args, device, time, async_evaluate=async_evaluate)
    result.update(duplex_results)
    if duplex_results:
        print(f"完成{len(duplex_results)}个Duplex数据集{'推理' if async_evaluate else '评估'}")
    
    # 评估Omni数据集
    print("\n" + "="*60)
    print("【开始评估Omni数据集】")
    print("="*60)
    omni_results = evaluate_omni_datasets(args, model, time, async_evaluate=async_evaluate)
    result.update(omni_results)
    if omni_results:
        print(f"完成{len(omni_results)}个Omni数据集{'推理' if async_evaluate else '评估'}")
    
    # 异步模式下，等待所有评估任务完成
    if async_evaluate:
        # 只有 rank 0 需要等待评估结果
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            pending_count = get_pending_eval_count()
            if pending_count > 0:
                print("\n" + "="*60)
                print(f"【等待异步评估任务完成】共 {pending_count} 个任务")
                print("="*60)
                async_results = wait_all_evaluations()
                result.update(async_results)
    
    print("\n" + "="*60)
    print(f"【所有评估任务完成】总计评估{len(result)}个数据集")
    print("="*60)
    
    return result

def save_evaluation_results(result, args, time):
    """保存评估结果，按照数据集名称分别保存"""
    print(f"\n最终评估结果汇总: {result}")
    
    # 构建结果保存路径
    result_dir = os.path.join(args.answer_path, args.model_name, time)
    os.makedirs(result_dir, exist_ok=True)
    
    # 检查是否有有效结果
    output_flag = False
    saved_files = []
    
    for dataset_name, score in result.items():
        if score is not None and score >= 0.0:
            output_flag = True
            
            # 为每个数据集创建单独的结果文件
            result_filename = f"result_{dataset_name}.json"
            result_path = os.path.join(result_dir, result_filename)
            job_id = os.getenv('JOB_ID', -1)
            # 保存单个数据集的结果
            dataset_result = {
                "dataset_name": dataset_name,
                "score": score,
                "model_name": args.model_name,
                "evaluation_time": time,
                "job_id": job_id,
            }
            
            with open(result_path, "w", encoding='utf-8') as f:
                json.dump(dataset_result, f, indent=4, ensure_ascii=False)
            
            saved_files.append(result_filename)
            print(f"✅ {dataset_name} 结果已保存到: {result_filename}")
    
    if output_flag:
        print(f"✅ 共保存了 {len(saved_files)} 个数据集的结果文件")
        print(f"📁 结果保存在目录: {result_dir}")
    else:
        print("⚠️ 没有有效的评估结果需要保存")