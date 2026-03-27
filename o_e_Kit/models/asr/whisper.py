import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch 
import soundfile as sf
import tqdm
import torchaudio

def read_audio(audio_path):
    # 防止librosa抽风，换成torchaudio了..
    y, sr = torchaudio.load(audio_path)
    y = torchaudio.functional.resample(y, sr, 16000)[0]
    y = y.numpy()
    return y

class Whisper:
    def __init__(self, model_path="openai/whisper-large-v3", device="cuda", batch_size=32):
        # 初始化Whisper模型和处理器
        processor = AutoProcessor.from_pretrained(model_path)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path, torch_dtype=torch.bfloat16).to(device=device)

        print(f"whisper model loaded on {device}")
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        print(f"asr pipeline loaded on {device}")
        self.asr_tmp_file_path = './asr_tmp'
        os.makedirs(self.asr_tmp_file_path, exist_ok=True)
        self.batch_size=batch_size
        
    def generate_batch(self, paths: list[dict], items: list[dict], dataset_name: str, modality: str = 'audio', **kwargs) -> list[str]:
        # 按batch_size分批处理
        batch_size=self.batch_size
        for i in tqdm.tqdm(range(0, len(paths), batch_size)):
            batch_files = paths[i:i + batch_size]
            print(f"processing {len(batch_files)} / total {len(paths)} files")
            
            # 处理音频文件
            processed_segments = []  # 存储所有需要处理的音频段
            wav_file_map = {}  # 记录每个音频段对应的原始文件
            
            for wav_file in batch_files:
                try:
                    wav_file = wav_file['audio_path']
                    audio_data= read_audio(wav_file)
                    sample_rate=16000
                    duration = len(audio_data) / sample_rate
                    
                    if duration <= 30.0:
                        # 短音频直接添加
                        processed_segments.append(wav_file)
                        wav_file_map[wav_file] = wav_file
                    else:
                        print(f"处理长音频文件: {wav_file}, 长度: {duration:.1f}s")
                        
                        # 将长音频切分成 30s 的片段
                        segment_length = 30 * sample_rate
                        segments = []
                        
                        for start in range(0, len(audio_data), segment_length):
                            
                            end = min(start + segment_length, len(audio_data))
                            
                            segment = audio_data[start:end]
                            
                            # 保存临时音频段
                            segment_path = os.path.join(self.asr_tmp_file_path, f"{os.path.basename(wav_file)}_segment_{start//segment_length}.wav")
                            print(f"saving segment to {segment_path}")
                            
                            sf.write(segment_path, segment, sample_rate)
                            
                            segments.append(segment_path)
                            
                            wav_file_map[segment_path] = wav_file
                            
                        processed_segments.extend(segments)
                        
                except Exception as e:
                    print(f"读取音频文件失败: {wav_file}, 错误: {e}")
                    continue
            
            if len(processed_segments) == 0:
                continue
                
            # 批量处理所有音频段
            results = self.pipe(processed_segments, batch_size=batch_size)
            
            # print(results["chunks"])
            
            # 合并结果并保存
            result_map = {}  # 存储每个原始文件的合并结果
            for segment_path, result in zip(processed_segments, results):
                original_file = wav_file_map[segment_path]
                if original_file not in result_map:
                    result_map[original_file] = []
                result_map[original_file].append(result["text"])
                
                # 删除临时音频段文件
                if segment_path != original_file:
                    os.remove(segment_path)
            all_result = []
            for ori_path in paths:
                ori_path = ori_path['audio_path']
                all_result.append(result_map[ori_path][0])
            print(f"all_result: {all_result[:5]}")
            return all_result