"""번역 설정 및 구성"""

import os
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path

# .env 파일 로드 시도
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv: api 키가 없으면 환경변수만 사용


@dataclass
class TranslationConfig:
    """번역 설정 클래스"""
    # API 설정
    api_provider: str = "anthropic"  # "openai", "anthropic", "ollama"
    api_key: str = ""
    model: str = "claude-3-haiku-20240307"
    korean_model: str = "jinbora/deepseek-r1-Bllossom:70b"  # 한국어 번역 전용 모델 (Ollama용)
    english_model: str = "winkefinger/alma-13b:Q4_K_M"  # 영어 번역 전용 모델 (Ollama용)
    ollama_base_url: str = "http://localhost:11434"  # Ollama 서버 URL
    
    # 배치 처리 설정
    batch_size: int = 50  # 30 -> 50으로 증가
    max_concurrent: int = 10  # 6 -> 10으로 증가
    delay_between_batches: float = 0.3  # 0.8 -> 0.3으로 감소
    
    # 재시도 설정
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # 파일 처리 설정
    chunk_size: int = 12000  # 8000 -> 12000으로 증가 (메모리가 충분한 경우)
    checkpoint_interval: int = 1000  # 500 -> 1000으로 증가 (체크포인트 빈도 감소)
    
    # 비용 관리
    budget_limit: float = 50.0  # $50 예산 제한
    
    def __post_init__(self):
        """초기화 후 환경변수에서 API 키 로드"""
        if not self.api_key:
            if self.api_provider == "openai":
                self.api_key = os.getenv('OPENAI_API_KEY', '')
            elif self.api_provider == "anthropic":
                self.api_key = os.getenv('ANTHROPIC_API_KEY', '')
            elif self.api_provider == "ollama":
                # Ollama는 API 키가 필요 없음
                self.api_key = "ollama-local"
                # Ollama URL 환경변수에서 로드
                self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', self.ollama_base_url)
                return
        
        if not self.api_key and self.api_provider != "ollama":
            print(f"경고: {self.api_provider.upper()}_API_KEY 환경변수가 설정되지 않았습니다.")
            print("다음 방법 중 하나를 사용하여 API 키를 설정하세요:")
            print(f"1. 환경변수: export {self.api_provider.upper()}_API_KEY='your-key'")
            print(f"2. .env 파일: {self.api_provider.upper()}_API_KEY=your-key")
            print("3. Ollama 사용: api_provider='ollama'로 설정하여 로컬 모델 사용")
            raise ValueError(f"{self.api_provider.upper()}_API_KEY가 필요합니다.")
    
    @classmethod
    def create_fast_config(cls, api_provider: str = "ollama") -> 'TranslationConfig':
        """고속 처리용 설정 생성"""
        config = cls(api_provider=api_provider)
        config.batch_size = 100  # 더 큰 배치 크기
        config.max_concurrent = 15  # 더 높은 동시성
        config.delay_between_batches = 0.1  # 더 짧은 지연
        config.chunk_size = 20000  # 더 큰 청크 크기
        config.checkpoint_interval = 2000  # 덜 빈번한 체크포인트
        return config
    
    @classmethod
    def create_gpu_optimized_config(cls, api_provider: str = "ollama") -> 'TranslationConfig':
        """GPU 최적화 설정 - 배치 관리 및 안정성 중심"""
        config = cls(api_provider=api_provider)
        config.batch_size = 5  # GPU 메모리 고려한 안전한 배치 크기
        config.max_concurrent = 1  # 순차 처리로 안정성 확보
        config.delay_between_batches = 0.5  # 모델 전환 시간 확보
        config.chunk_size = 1000  # 체크포인트 빈도 증가
        config.checkpoint_interval = 50  # 더 자주 체크포인트
        config.max_retries = 5  # 재시도 횟수 증가
        config.retry_delay = 3.0  # 재시도 지연 증가
        return config
    
    @classmethod 
    def create_safe_config(cls, api_provider: str = "anthropic") -> 'TranslationConfig':
        """안전한 처리용 설정 생성 (API 제한 고려)"""
        config = cls(api_provider=api_provider)
        config.batch_size = 20  # 작은 배치 크기
        config.max_concurrent = 3  # 낮은 동시성
        config.delay_between_batches = 1.0  # 긴 지연
        config.chunk_size = 5000  # 작은 청크 크기
        config.checkpoint_interval = 200  # 빈번한 체크포인트
        return config


class APIConfig:
    """API 관련 상수 및 설정"""
    
    # API 엔드포인트
    ENDPOINTS = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "anthropic": "https://api.anthropic.com/v1/messages",
        "ollama": "http://localhost:11434/api/generate"
    }
    
    # 모델별 비용 (per 1K tokens)
    COSTS = {
        "openai": {
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06}
        },
        "anthropic": {
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015}
        },
        "ollama": {
            "llama3": {"input": 0.0, "output": 0.0},  # 로컬 모델은 무료
            "llama3.1": {"input": 0.0, "output": 0.0},
            "mixtral": {"input": 0.0, "output": 0.0},
            "qwen2": {"input": 0.0, "output": 0.0},
            "default": {"input": 0.0, "output": 0.0}
        }
    }
    
    # 언어 매핑
    LANGUAGE_MAP = {
        "korean": "Korean",
        "english": "English",
        "chinese": "Chinese"
    }
    
    @classmethod
    def get_cost_info(cls, api_provider: str, model: str) -> Dict[str, float]:
        """모델별 비용 정보 조회"""
        return cls.COSTS.get(api_provider, {}).get(model, {"input": 0, "output": 0})
    
    @classmethod
    def get_endpoint(cls, api_provider: str) -> str:
        """API 엔드포인트 조회"""
        return cls.ENDPOINTS.get(api_provider, "")