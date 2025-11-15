"""
KEadapter - 대용량 고전 중국어 번역 패키지

이 패키지는 ACCN-INS와 같은 대용량 고전 중국어 데이터셋을
한국어와 영어로 번역하는 도구를 제공합니다.

주요 기능:
- 비동기 배치 번역 처리
- 다중 API 제공업체 지원 (OpenAI, Anthropic)
- 번역 캐싱으로 비용 절약
- 체크포인트 시스템으로 중단 복구 가능
- 예산 관리 및 비용 추적
"""

__version__ = "1.0.0"
__author__ = "한승현"

from .config import TranslationConfig, APIConfig
from .translator import LargeScaleTranslator
from .cost_tracker import estimate_translation_cost

__all__ = [
    "TranslationConfig",
    "APIConfig", 
    "LargeScaleTranslator",
    "estimate_translation_cost"
]