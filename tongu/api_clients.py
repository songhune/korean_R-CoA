"""API 클라이언트 모듈"""

import aiohttp
import backoff
from typing import List, Dict, Any, Tuple
import logging

from config import TranslationConfig, APIConfig


class BaseAPIClient:
    """기본 API 클라이언트"""
    
    def __init__(self, config: TranslationConfig, session: aiohttp.ClientSession):
        self.config = config
        self.session = session
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
        """배치 번역 - 서브클래스에서 구현"""
        raise NotImplementedError


class OpenAIClient(BaseAPIClient):
    """OpenAI API 클라이언트"""
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def translate_batch(self, texts: List[str], target_lang: str) -> List[str]:
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
                return cleaned_translations[:len(texts)]
            else:
                error_text = await response.text()
                raise Exception(f"OpenAI API Error {response.status}: {error_text}")
    
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
                raise Exception(f"Anthropic API Error {response.status}: {error_text}")
    
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
            cleaned.append("[Translation Error]")
        
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
        
        prompt = f"""Translate the following Modern Chinese texts to {target_language}. 
These texts have been translated from Classical Chinese and contain rich contextual information.
Provide accurate, natural translations that preserve the original meaning and cultural context.
Return only the translations, one per line, in the same order:

{chr(10).join(f"{i+1}. {text}" for i, text in enumerate(texts))}"""
        
        payload = {
            "model": self.config.model,
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
            async with self.session.post(endpoint, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    content = result.get("response", "").strip()
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
                    raise Exception(f"Ollama API Error {response.status}: {error_text}")
        except aiohttp.ClientConnectorError:
            raise Exception("Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요.")
    
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
            cleaned.append("[Translation Error]")
        
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