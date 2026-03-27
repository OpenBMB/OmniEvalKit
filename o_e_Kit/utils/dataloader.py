"""
统一的数据加载器模块 - 支持三元组格式
提供现代化的数据加载接口和分布式支持
"""

import torch
from typing import List, Dict, Any, Tuple
from torch.utils.data import DataLoader, Sampler


class InferenceSampler(Sampler):
    """分布式推理采样器"""
    
    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self._world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def triplet_collate_fn(batches: List[Tuple[int, Dict[str, str], Dict[str, Any]]]) -> Tuple[List[int], List[Dict[str, str]], List[Dict[str, Any]]]:
    """
    Args:
        batches: 批次数据，每个元素是(idx, paths, annotation)三元组
    """
    indices = []
    paths_list = []
    annotations_list = []
    
    for item in batches:
        idx, paths, annotation = item
        indices.append(idx)
        paths_list.append(paths)
        annotations_list.append(annotation)
    
    return indices, paths_list, annotations_list


def create_dataloader(dataset, 
                     batch_size: int = 1, 
                     use_distributed: bool = None) -> DataLoader:
    if use_distributed is None:
        use_distributed = torch.distributed.is_initialized()
    
    sampler = None
    if use_distributed:
        sampler = InferenceSampler(len(dataset))
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=triplet_collate_fn,
        shuffle=False if sampler is not None else False,
        num_workers=0,  # 推理时通常设为0避免多进程问题
        pin_memory=True
    )


if __name__ == '__main__':
    """测试数据加载器功能"""
    print("🧪 现代数据加载器测试")
    print("=" * 50)
    
    # 模拟数据集
    class MockDataset:
        def __init__(self):
            self.data = [
                (0, {'audio_path': 'audio/test1.wav'}, {'question': 'What is this?', 'gt_answer': 'Test 1', 'task_type': 'ASR'}),
                (1, {'video_path': 'video/test2.mp4'}, {'question': 'Describe the video', 'gt_answer': 'Test 2', 'task_type': 'VideoQA'}),
                (2, {'audio_path': 'audio/test3.wav', 'video_path': 'video/test3.mp4'}, {'question': 'Multimodal test', 'gt_answer': 'Test 3', 'task_type': 'Multimodal'})
            ]
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # 测试三元组格式
    print("--- 测试三元组数据加载器 ---")
    dataset = MockDataset()
    dataloader = create_dataloader(dataset, batch_size=2, use_distributed=False)
    
    for batch in dataloader:
        indices, paths_list, annotations_list = batch
        print(f"批次大小: {len(indices)}")
        print(f"索引: {indices}")
        print(f"路径: {paths_list}")
        print(f"任务类型: {[ann['task_type'] for ann in annotations_list]}")
        break
    
    print("\n✅ 现代数据加载器测试完成！")
    print("🎯 特性:")
    print("- 三元组格式的统一数据接口")
    print("- 自动分布式推理支持")
    print("- 简洁清晰的API设计")
    print("- 高性能的数据处理")