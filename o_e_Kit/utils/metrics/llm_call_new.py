
import json
import time
import requests
import os
from enum import Enum
from pathlib import Path

# 自动加载 .env 文件
def _load_env_file():
    """加载 .env 文件（支持软链接）"""
    # 查找 .env 文件的位置
    search_paths = [
        Path(__file__).parent.parent.parent.parent.parent / ".env",
        Path.cwd() / ".env",
        Path.cwd() / ".env.local",
    ]
    
    for env_path in search_paths:
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, _, value = line.partition('=')
                            key = key.strip()
                            value = value.strip()
                            # 只设置未设置的环境变量
                            if key and key not in os.environ:
                                os.environ[key] = value
                return str(env_path)
            except Exception:
                pass
    return None

_loaded_env = _load_env_file()

class APIModelName(Enum):
    """
    API 模型名称枚举
    
    可通过环境变量覆盖模型名称，例如：
    - MODEL_GPT_4O_MINI=gpt-4o-mini-2024-07-18
    """
    # GPT-4 系列
    GPT_4_TURBO = os.environ.get("MODEL_GPT_4_TURBO", "gpt-4-turbo")
    
    # GPT-4o 系列
    GPT_4O = os.environ.get("MODEL_GPT_4O", "gpt-4o")
    GPT_4O_MINI = os.environ.get("MODEL_GPT_4O_MINI", "gpt-4o-mini")
    
    # Gemini 系列（需要配合 Gemini API 使用）
    GEMINI_2_5_FLASH = os.environ.get("MODEL_GEMINI_2_5_FLASH", "gemini-2.5-flash")
    GEMINI_2_5_PRO = os.environ.get("MODEL_GEMINI_2_5_PRO", "gemini-2.5-pro")
    

class ChatClient:
    """
    OpenAI兼容的聊天客户端，与llm_call.py的ChatClient接口保持一致
    支持多 API Key 备选：当主 key 余额不足时自动切换备选 key
    
    API Key 配置方式（按优先级）：
    1. 构造函数参数 api_key
    2. 环境变量 OPENAI_API_KEY
    3. 环境变量 OPENAI_API_KEY_1, OPENAI_API_KEY_2, ... (备选 keys)
    """
    
    @staticmethod
    def _load_api_keys_from_env() -> list:
        """从环境变量加载 API keys"""
        keys = []
        # 主 key
        main_key = os.environ.get('OPENAI_API_KEY', '')
        if main_key:
            keys.append(main_key)
        # 备选 keys: OPENAI_API_KEY_1, OPENAI_API_KEY_2, ...
        for i in range(1, 10):
            key = os.environ.get(f'OPENAI_API_KEY_{i}', '')
            if key:
                keys.append(key)
        return keys
    
    def __init__(self, app_code='general_qa', user_token='', api_key=None, base_url=None):
        """
        初始化ChatClient
        
        Args:
            app_code: 应用代码（为了兼容性保留，但不使用）
            user_token: 用户token（为了兼容性保留，但不使用）
            api_key: OpenAI API密钥
            base_url: API基础URL
        """
        self.app_code = app_code
        self.user_token = user_token
        
        # 加载 API keys
        self._fallback_keys = self._load_api_keys_from_env()
        
        # 使用传入的 key 或环境变量 key
        if api_key:
            self.api_key = api_key
        elif self._fallback_keys:
            self.api_key = self._fallback_keys[0]
        else:
            self.api_key = None
            print("⚠️ Warning: No API key configured. Please set OPENAI_API_KEY environment variable.")
        
        self.base_url = base_url or os.environ.get(
            'OPENAI_API_BASE', 
            'https://api.openai.com/v1/chat/completions'
        )
        self._current_key_index = 0  # 当前使用的 key 索引
    
    
    def get_app_token(self):
        """
        为了兼容性保留此方法，但OpenAI API不需要app token
        """
        return self.api_key
    
    def _switch_to_next_key(self):
        """
        切换到下一个备选 API Key
        
        Returns:
            bool: 是否成功切换（如果已经是最后一个 key 则返回 False）
        """
        if self._current_key_index < len(self._fallback_keys) - 1:
            self._current_key_index += 1
            self.api_key = self._fallback_keys[self._current_key_index]
            print(f"🔄 切换到备选 API Key #{self._current_key_index + 1}")
            return True
        return False
    
    def _is_quota_or_auth_error(self, error_msg):
        """
        判断是否是余额不足或认证错误
        """
        error_keywords = [
            'insufficient_quota', 'quota', 'balance', 'credit',
            'rate_limit', 'billing', 'payment', 'exceeded',
            'unauthorized', 'invalid_api_key', 'authentication',
            '余额', '额度', '欠费', '认证失败'
        ]
        error_lower = str(error_msg).lower()
        return any(keyword in error_lower for keyword in error_keywords)
    
    def _call_api_with_retry(self, model_name, system_message, user_message, **kwargs):
        """
        带重试日志的API调用包装函数，支持 API Key 自动切换
        """
        max_attempts = 5
        wait_time = 0
        
        for attempt in range(1, max_attempts + 1):
            try:
                return self._call_api(model_name, system_message, user_message, **kwargs)
            except Exception as e:
                error_msg = str(e)
                
                # 如果是余额/认证错误，尝试切换 key
                if self._is_quota_or_auth_error(error_msg):
                    if self._switch_to_next_key():
                        print(f"⚠️  检测到余额/认证问题，已切换 API Key，重新尝试...")
                        continue
                
                if attempt < max_attempts:
                    print(f"⚠️  API调用失败，正在进行第 {attempt} 次重试（最多{max_attempts}次），等待{wait_time}秒...")
                    print(f"    失败原因: {e}")
                    wait_time += 1
                    time.sleep(wait_time)
                else:
                    raise
    
    def _call_api(self, model_name, system_message, user_message, **kwargs):
        """
        内部API调用方法
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
            
            # 构建消息列表
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            
            if isinstance(user_message, str):
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": user_message}]
                })
            
            # 构建请求payload
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0),
                "max_tokens": kwargs.get("max_tokens", 128),
            }
            
            # 如果需要JSON格式响应
            if kwargs.get("response_format"):
                payload["response_format"] = kwargs["response_format"]
            
            response = requests.post(
                self.base_url, 
                headers=headers, 
                data=json.dumps(payload)
            )
            # print(response.text)
            if response.status_code == 200:
                # 处理非流式响应
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                error_text = response.text
                print(f"API错误: {response.status_code}, {error_text}")
                # 如果是余额/认证相关错误，抛出异常让上层处理
                if response.status_code in [401, 402, 403, 429]:
                    raise Exception(f"API错误 {response.status_code}: {error_text}")
                return ""
                
        except Exception as e:
            print(f"API请求异常: {e}")
            raise
    
    def get_eval(self, content, chat_gpt_system=None, 
                 max_tokens=2048, fail_limit=2, return_resp=False, model_name=APIModelName.GPT_4O_MINI, temperature=0.1):
        """
        与原始ChatClient兼容的评估方法
        
        Args:
            content: 用户消息内容
            chat_gpt_system: 系统消息
            max_tokens: 最大token数
            fail_limit: 失败重试次数（由retry装饰器处理）
            return_resp: 是否返回完整响应（当前不支持）
            model_id: 模型ID（将映射到模型名称）
            temperature: 生成温度
            
        Returns:
            模型生成的内容
        """
        
        try:
            result = self._call_api_with_retry(
                model_name=model_name.value,
                system_message=chat_gpt_system,
                user_message=content,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if return_resp:
                # 构造兼容的响应格式
                resp = {
                    'data': {
                        'messages': [{'content': result}]
                    }
                }
                return result, resp
            else:
                return result
                
        except Exception as e:
            print(f"评估失败: {e}")
            if fail_limit > 0:
                # retry装饰器会处理重试
                raise ValueError(f"Error: API call failed - {str(e)}")
            return ""
    
def test_all_api_keys():
    """
    测试所有 API Key 是否可用
    """
    print("=" * 60)
    print("🔑 测试所有 API Keys")
    print("=" * 60)
    
    # 从环境变量加载 keys
    keys = ChatClient._load_api_keys_from_env()
    if not keys:
        print("❌ 没有配置 API Key，请设置环境变量 OPENAI_API_KEY")
        return
    
    results = []
    for i, key in enumerate(keys):
        print(f"\n📌 测试 Key #{i+1}: {key[:8]}...{key[-4:]}")
        client = ChatClient(api_key=key)
        try:
            response = client._call_api(
                model_name=APIModelName.GPT_4O_MINI.value,
                system_message=None,
                user_message="请回复：OK",
                max_tokens=10
            )
            if response and len(response) > 0:
                print(f"   ✅ 可用 - 响应: {response[:50]}")
                results.append((key, True, response[:50]))
            else:
                print(f"   ❌ 响应为空")
                results.append((key, False, "响应为空"))
        except Exception as e:
            print(f"   ❌ 失败 - {str(e)[:80]}")
            results.append((key, False, str(e)[:80]))
    
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    available = sum(1 for _, ok, _ in results if ok)
    print(f"可用: {available}/{len(results)}")
    for i, (key, ok, msg) in enumerate(results):
        status = "✅" if ok else "❌"
        print(f"  Key #{i+1} [{key[:8]}...]: {status} {msg}")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    # 测试0: 测试所有 API Key
    print("\n" + "🔥" * 30)
    print("测试0 - 测试所有 API Keys 是否可用:")
    print("🔥" * 30)
    test_all_api_keys()
    
    # 测试1: 使用get_eval方法（使用默认model_id）
    print("\n" + "-" * 60)
    print("测试1 - get_eval方法:")
    client = ChatClient()
    try:
        response = client.get_eval(content="你好")
        print(f"响应: {response}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试2: 直接使用_call_api方法，指定GEMINI模型
    print("\n测试2 - 使用GEMINI模型:")
    try:
        response = client._call_api(
            model_name="GEMINI_0g1iy4",
            system_message=None,
            user_message="你是谁",
        )
        print(f"响应: {response}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 测试3: 测试自动切换 Key 功能
    print("\n测试3 - 测试自动切换 Key 功能:")
    print("=" * 60)
    print("（如果第一个 key 失败，应自动切换到备选 key）")
    client_auto = ChatClient()
    try:
        response = client_auto.get_eval(content="测试自动切换", model_name=APIModelName.GPT_4O_MINI)
        print(f"响应: {response}")
        print(f"当前使用 Key 索引: {client_auto._current_key_index}")
    except Exception as e:
        print(f"最终失败: {e}")
    print("=" * 60)
    