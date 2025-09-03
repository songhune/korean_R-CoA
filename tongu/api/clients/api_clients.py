"""API 클라이언트 모듈"""

import aiohttp
import backoff
import asyncio
from typing import List, Dict, Any, Tuple
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.config.config import TranslationConfig, APIConfig
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from error_handler import track_broken_pipe_error, track_connection_error, track_model_error


class BaseAPIClient:
    """기본 API 클라이언트"""
    
    def __init__(self, config: TranslationConfig, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def translate_batch(self, texts: List[str], target_lang: str) -> Tuple[List[str], Dict[str, int]]:
        """배치 번역 - 서브클래스에서 구현"""
        raise NotImplementedError


class OpenAIClient(BaseAPIClient):
    """OpenAI API 클라이언트"""
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def translate_batch(self, texts: List[str], target_lang: str) -> Tuple[List[str], Dict[str, int]]:
        """OpenAI API를 사용한 배치 번역"""
        target_language = APIConfig.LANGUAGE_MAP.get(target_lang, target_lang)
        
        batch_prompt = f"""
Translate the following Modern Chinese texts to {target_language}. 
These texts are already translated from Classical Chinese to Modern Chinese.
Return ONLY the translations, one per line, maintaining the same order.
Do not include any explanations or numbering.

Texts to translate:
{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(texts))}
"""
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": f"You are a professional translator specializing in Modern Chinese to {target_language} translation. Focus on preserving the contextual meaning and cultural nuances."},
                {"role": "user", "content": batch_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": len(' '.join(texts)) * 3
        }
        
        endpoint = APIConfig.get_endpoint("openai")
        
        async with self.session.post(endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                translations = result["choices"][0]["message"]["content"].strip().split('\n')
                
                # 결과 정제
                cleaned_translations = self._clean_translations(translations)
                
                # 사용량 정보 (OpenAI는 토큰 정보를 제공하지만 일단 추정값 사용)
                usage_info = {
                    "input_tokens": len(batch_prompt) // 4,
                    "output_tokens": len(result["choices"][0]["message"]["content"]) // 4
                }
                
                return cleaned_translations[:len(texts)], usage_info
            else:
                error_text = await response.text()
                error_msg = f"OpenAI API Error {response.status}: {error_text}"
                print(f"\n❌ {error_msg}")
                self.logger.error(error_msg)
                raise Exception(error_msg)
    
    def _clean_translations(self, translations: List[str]) -> List[str]:
        """번역 결과 정제"""
        cleaned = []
        for trans in translations:
            trans = trans.strip()
            # 번호 제거 (1., 2. 등)
            if trans and trans[0].isdigit() and '.' in trans[:5]:
                trans = trans.split('.', 1)[1].strip()
            cleaned.append(trans)
        return cleaned


class AnthropicClient(BaseAPIClient):
    """Anthropic Claude API 클라이언트"""
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def translate_batch(self, texts: List[str], target_lang: str) -> Tuple[List[str], Dict[str, int]]:
        """Anthropic API를 사용한 배치 번역 (사용량 정보 포함)"""
        target_language = APIConfig.LANGUAGE_MAP.get(target_lang, target_lang)
        
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        prompt = f"""Translate the following Modern Chinese texts to {target_language}. 
These texts have been translated from Classical Chinese and contain rich contextual information.
Provide accurate, natural translations that preserve the original meaning and cultural context.
Return only the translations, one per line, in the same order:

{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(texts))}"""
        
        payload = {
            "model": self.config.model,
            "max_tokens": len(' '.join(texts)) * 4,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        
        endpoint = APIConfig.get_endpoint("anthropic")
        
        async with self.session.post(endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                content = result["content"][0]["text"].strip()
                translations = content.split('\n')
                
                # 사용량 정보 추출
                usage = result.get("usage", {})
                usage_info = {
                    "input_tokens": usage.get("input_tokens", len(prompt) // 4),
                    "output_tokens": usage.get("output_tokens", len(content) // 4)
                }
                
                # 결과 정제
                cleaned_translations = self._clean_translations(translations, len(texts))
                return cleaned_translations, usage_info
            else:
                error_text = await response.text()
                error_msg = f"Anthropic API Error {response.status}: {error_text}"
                print(f"\n❌ {error_msg}")
                self.logger.error(error_msg)
                raise Exception(error_msg)
    
    def _clean_translations(self, translations: List[str], expected_count: int) -> List[str]:
        """번역 결과 정제"""
        cleaned = []
        for trans in translations:
            trans = trans.strip()
            # 번호 제거
            if trans and trans[0].isdigit() and '.' in trans[:5]:
                trans = trans.split('.', 1)[1].strip()
            if trans:  # 빈 번역 방지
                cleaned.append(trans)
        
        # 원본과 같은 수가 되도록 조정
        while len(cleaned) < expected_count:
            cleaned.append("[Translation Error: Missing translation]")
        
        return cleaned[:expected_count]


class OllamaClient(BaseAPIClient):
    """Ollama API 클라이언트 - 로컬 오픈소스 모델"""
    
    def __init__(self, config: TranslationConfig, session: aiohttp.ClientSession):
        super().__init__(config, session)
        self.base_url = getattr(config, 'ollama_base_url', 'http://localhost:11434')
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def translate_batch(self, texts: List[str], target_lang: str) -> Tuple[List[str], Dict[str, int]]:
        """Ollama API를 사용한 배치 번역"""
        target_language = APIConfig.LANGUAGE_MAP.get(target_lang, target_lang)
        
        # 언어별 모델 선택
        if target_language == "Korean" and hasattr(self.config, 'korean_model') and self.config.korean_model:
            model_to_use = self.config.korean_model
            print(f"🇰🇷 한글 번역용 모델 사용: {model_to_use}")
        elif target_language == "English" and hasattr(self.config, 'english_model') and self.config.english_model:
            model_to_use = self.config.english_model
            print(f"🇺🇸 영어 번역용 모델 사용: {model_to_use}")
        else:
            model_to_use = self.config.model
        
        if target_language == "Korean":
            prompt = f"""다음 현대 중국어 텍스트를 자연스러운 한국어로 번역하세요. 각 텍스트를 한 줄씩 번역하여 같은 순서로 제공하세요:

{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(texts))}

한국어 번역:"""
        else:
            prompt = f"""Please translate the following Modern Chinese texts to {target_language}:

{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(texts))}

{target_language} translations:
"""
        
        payload = {
            "model": model_to_use,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        endpoint = f"{self.base_url}/api/generate"
        
        try:
            # Set longer timeout for model processing
            timeout = aiohttp.ClientTimeout(total=300, connect=30)  # 5 minutes total, 30s connect
            
            async with self.session.post(endpoint, json=payload, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get("response", "").strip()
                    
                    # 빈 응답 처리
                    if not content:
                        if target_language == "English":
                            # 이 모델은 영어 번역에 적합하지 않으므로 건너뛰기
                            print(f"\n⚠️ {self.config.model}은 영어 번역을 지원하지 않습니다. 건너뛰는 중...")
                            skip_translations = ["[Skip: Model does not support English translation]" for _ in texts]
                            usage_info = {"input_tokens": 0, "output_tokens": 0}
                            return skip_translations, usage_info
                        else:
                            print(f"\n⚠️ Ollama에서 빈 응답을 받았습니다")
                            error_msg = "Empty response from Ollama"
                            track_model_error(error_msg, model=model_to_use, target_lang=target_language)
                            error_translations = ["[Translation Error: Empty response from Ollama]" for _ in texts]
                            usage_info = {"input_tokens": 0, "output_tokens": 0}
                            return error_translations, usage_info
                    
                    translations = content.split('\n')
                    
                    # 사용량 정보 (Ollama는 토큰 카운트를 제공하지 않으므로 추정)
                    usage_info = {
                        "input_tokens": len(prompt) // 4,  # 대략적 추정
                        "output_tokens": len(content) // 4
                    }
                    
                    # 결과 정제
                    cleaned_translations = self._clean_translations(translations, len(texts))
                    return cleaned_translations, usage_info
                else:
                    error_text = await response.text()
                    error_msg = f"Ollama API Error {response.status}: {error_text}"
                    print(f"\n❌ {error_msg}")
                    raise Exception(error_msg)
        except aiohttp.ClientConnectorError as e:
            error_msg = f"❌ Ollama 서버 연결 실패 ({self.base_url}): {str(e)}"
            print(f"\n{error_msg}")
            print("   해결방법:")
            print("   1. ollama serve 명령으로 서버 시작")
            print(f"   2. 모델 다운로드: ollama pull {self.config.model}")
            print(f"   3. 서버 URL 확인: {self.base_url}")
            self.logger.error(error_msg)
            
            # Track connection error for monitoring
            track_connection_error(error_msg, 
                                        model=getattr(self, 'config', {}).get('model', 'unknown'),
                                        base_url=self.base_url,
                                        target_lang=target_lang)
            
            # [Translation Error] 토큰으로 명시적 표시
            error_translations = ["[Translation Error: Ollama 서버 연결 실패]" for _ in texts]
            usage_info = {"input_tokens": 0, "output_tokens": 0}
            return error_translations, usage_info
            
        except (ConnectionResetError, BrokenPipeError) as e:
            error_msg = f"❌ 연결이 끊어졌습니다 (Broken Pipe): {str(e)}"
            print(f"\n{error_msg}")
            print("   대용량 모델 처리 중 연결이 끊어질 수 있습니다.")
            print("   잠시 후 재시도하거나 배치 크기를 줄여보세요.")
            self.logger.error(error_msg)
            
            # Track broken pipe error for monitoring
            track_broken_pipe_error(error_msg,
                                  model=getattr(self, 'config', {}).get('model', 'unknown'),
                                  base_url=self.base_url,
                                  target_lang=target_lang,
                                  batch_size=len(texts))
            
            error_translations = [f"[Translation Error: Connection broken - {str(e)}]" for _ in texts]
            usage_info = {"input_tokens": 0, "output_tokens": 0}
            return error_translations, usage_info
            
        except aiohttp.ServerTimeoutError as e:
            error_msg = f"❌ 서버 응답 시간 초과: {str(e)}"
            print(f"\n{error_msg}")
            print("   대용량 모델이 응답하는데 시간이 오래 걸릴 수 있습니다.")
            self.logger.error(error_msg)
            
            track_model_error(error_msg,
                                     model=getattr(self, 'config', {}).get('model', 'unknown'),
                                     target_lang=target_lang,
                                     timeout=True)
            
            error_translations = [f"[Translation Error: Timeout - {str(e)}]" for _ in texts]
            usage_info = {"input_tokens": 0, "output_tokens": 0}
            return error_translations, usage_info
            
        except Exception as e:
            error_msg = f"❌ Ollama API 호출 오류: {str(e)}"
            print(f"\n{error_msg}")
            self.logger.error(error_msg)
            
            # Track general API errors
            track_model_error(error_msg,
                                     model=getattr(self, 'config', {}).get('model', 'unknown'),
                                     target_lang=target_lang,
                                     error_type=type(e).__name__)
            
            # 구체적인 오류 정보와 함께 에러 토큰 반환
            error_translations = [f"[Translation Error: {str(e)}]" for _ in texts]
            usage_info = {"input_tokens": 0, "output_tokens": 0}
            return error_translations, usage_info
    
    def _clean_translations(self, translations: List[str], expected_count: int) -> List[str]:
        """번역 결과 정제"""
        cleaned = []
        for trans in translations:
            trans = trans.strip()
            # 번호 제거
            if trans and trans[0].isdigit() and '.' in trans[:5]:
                trans = trans.split('.', 1)[1].strip()
            if trans:  # 빈 번역만 추가
                cleaned.append(trans)
        
        # 원본과 같은 수가 되도록 조정 - 명확한 에러 메시지로 패딩
        while len(cleaned) < expected_count:
            cleaned.append("[Translation Error: Empty response]")
        
        return cleaned[:expected_count]


class APIClientFactory:
    """API 클라이언트 팩토리"""
    
    @staticmethod
    def create_client(config: TranslationConfig, session: aiohttp.ClientSession) -> BaseAPIClient:
        """설정에 따라 적절한 API 클라이언트 생성"""
        if config.api_provider == "openai":
            return OpenAIClient(config, session)
        elif config.api_provider == "anthropic":
            return AnthropicClient(config, session)
        elif config.api_provider == "ollama":
            return OllamaClient(config, session)
        else:
            raise ValueError(f"Unsupported API provider: {config.api_provider}")