"""비용 추적 및 관리 모듈"""

import logging
from typing import Dict, Any
from config import APIConfig, TranslationConfig


class CostTracker:
    """번역 비용 추적기"""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost = 0.0
        self.logger = logging.getLogger(__name__)
        
        # 모델별 비용 정보
        self.cost_info = APIConfig.get_cost_info(config.api_provider, config.model)
    
    def update_cost_tracking(self, input_tokens: int, output_tokens: int) -> float:
        """비용 추적 업데이트"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # 비용 계산
        input_cost = (input_tokens / 1000) * self.cost_info["input"]
        output_cost = (output_tokens / 1000) * self.cost_info["output"]
        batch_cost = input_cost + output_cost
        self.estimated_cost += batch_cost
        
        self.logger.info(
            f"Batch cost: ${batch_cost:.4f}, "
            f"Total: ${self.estimated_cost:.2f}/${self.config.budget_limit}"
        )
        
        return batch_cost
    
    def check_budget(self) -> bool:
        """예산 확인"""
        # 예산 한도가 0인 경우 (무료 모델) 예산 체크 건너뛰기
        if self.config.budget_limit <= 0:
            return True
            
        if self.estimated_cost >= self.config.budget_limit * 0.95:
            self.logger.warning(
                f"Budget almost exhausted: "
                f"${self.estimated_cost:.2f}/${self.config.budget_limit}"
            )
            return False
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """비용 통계 반환"""
        # 예산 한도가 0인 경우 처리 (Ollama 등)
        if self.config.budget_limit > 0:
            budget_usage_percent = (self.estimated_cost / self.config.budget_limit) * 100
        else:
            budget_usage_percent = 0.0
            
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost": self.estimated_cost,
            "budget_limit": self.config.budget_limit,
            "budget_usage_percent": budget_usage_percent,
            "cost_per_1k_input": self.cost_info["input"],
            "cost_per_1k_output": self.cost_info["output"]
        }
    
    def print_final_statistics(self, processed_count: int, failed_count: int):
        """최종 통계 출력"""
        stats = self.get_statistics()
        
        print(f"\n=== 번역 완료 통계 ===")
        print(f"총 처리 항목: {processed_count:,}")
        print(f"실패 항목: {failed_count}")
        print(f"총 비용: ${stats['estimated_cost']:.2f}")
        print(f"예산 사용률: {stats['budget_usage_percent']:.1f}%")
        print(f"입력 토큰: {stats['total_input_tokens']:,}")
        print(f"출력 토큰: {stats['total_output_tokens']:,}")


def estimate_translation_cost(file_path: str, api_provider: str = "anthropic", model: str = "claude-3-haiku-20240307") -> float:
    """번역 비용 사전 추정"""
    import json
    
    # 파일 분석
    total_chars = 0
    line_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # ACCN-INS 구조에서 텍스트 추출
                    if 'data' in item and 'instruction' in item['data']:
                        text = item['data']['instruction']
                    else:
                        text = str(item)
                    
                    total_chars += len(text)
                    line_count += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return 0.0
    
    # 토큰 수 추정 (중국어: 1 char ≈ 1 token, 번역 결과: 2x)
    input_tokens = total_chars
    output_tokens = total_chars * 2 * 2  # 한글 + 영어
    
    # 비용 계산
    cost_info = APIConfig.get_cost_info(api_provider, model)
    total_cost = (input_tokens * cost_info["input"] + output_tokens * cost_info["output"]) / 1000
    
    print(f"=== 비용 추정 결과 ===")
    print(f"파일: {file_path}")
    print(f"총 라인 수: {line_count:,}")
    print(f"총 문자 수: {total_chars:,}")
    print(f"예상 입력 토큰: {input_tokens:,}")
    print(f"예상 출력 토큰: {output_tokens:,}")
    print(f"예상 비용 ({api_provider} {model}): ${total_cost:.2f}")
    
    return total_cost