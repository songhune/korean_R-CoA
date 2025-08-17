#!/usr/bin/env python3
"""
NLI/STS 데이터 생성 통합 실행 스크립트

이 스크립트는 제안된 알고리즘의 Plan A (점진적 구현) 버전으로,
기존 CC-KR 데이터를 활용하여 NLI/STS 데이터셋을 생성합니다.

사용법:
    python run_nli_sts_generation.py --mode all
    python run_nli_sts_generation.py --mode kr_only
    python run_nli_sts_generation.py --mode cc_kr_only
"""

import argparse
import os
import sys
import json
from pathlib import Path

# 프로젝트 루트 디렉토리를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from network.kr_nli_generator import KoreanNLIGenerator
from network.kr_sts_generator import KoreanSTSGenerator  
from network.cc_kr_processor import ClassicalChineseKoreanProcessor

def load_korean_text_data(data_dir: str) -> list[str]:
    """한국어 텍스트 데이터 로드"""
    korean_texts = []
    
    # 1. 과거시험 한국어 해설 데이터
    try:
        gwashi_path = os.path.join(data_dir, "gwashi.csv")
        if os.path.exists(gwashi_path):
            import pandas as pd
            df = pd.read_csv(gwashi_path, encoding='cp949')
            contents = df['contents'].dropna().astype(str).tolist()
            korean_texts.extend([text for text in contents if len(text.strip()) > 20])
            print(f"✅ gwashi.csv에서 {len(contents)}개 텍스트 로드")
    except Exception as e:
        print(f"❌ gwashi.csv 로드 실패: {e}")
    
    # 2. 문집 데이터
    try:
        munjib_path = os.path.join(data_dir, "munjib.csv") 
        if os.path.exists(munjib_path):
            import pandas as pd
            df = pd.read_csv(munjib_path, encoding='utf-8')
            if 'answer_contents' in df.columns:
                contents = df['answer_contents'].dropna().astype(str).tolist()
                korean_texts.extend([text for text in contents if len(text.strip()) > 20])
                print(f"✅ munjib.csv에서 {len(contents)}개 텍스트 로드")
    except Exception as e:
        print(f"❌ munjib.csv 로드 실패: {e}")
    
    # 3. 사서 한국어 해설 (JSONL에서 추출)
    try:
        saseo_path = os.path.join(data_dir, "sigwon", "SSDB", "saseo_qwen3_chat.jsonl")
        if os.path.exists(saseo_path):
            with open(saseo_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        messages = data.get("messages", [])
                        for msg in messages:
                            if msg.get("role") == "assistant":
                                content = msg.get("content", "")
                                # <think> 태그 제거
                                import re
                                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                                content = content.strip()
                                if len(content) > 20:
                                    korean_texts.append(content)
            print(f"✅ saseo JSONL에서 텍스트 추가 로드")
    except Exception as e:
        print(f"❌ saseo JSONL 로드 실패: {e}")
    
    print(f"📊 총 한국어 텍스트: {len(korean_texts)}개")
    return korean_texts

def generate_korean_nli_sts(korean_texts: list[str], output_dir: str):
    """한국어 NLI/STS 데이터 생성"""
    print("\n=== 한국어 NLI/STS 데이터 생성 시작 ===")
    
    # NLI 생성
    print("1. 한국어 NLI 데이터 생성 중...")
    nli_generator = KoreanNLIGenerator()
    
    # 텍스트 샘플링 (너무 많으면 시간이 오래 걸림)
    sample_size = min(100, len(korean_texts))
    sampled_texts = korean_texts[:sample_size]
    
    nli_triples = nli_generator.generate_nli_dataset(sampled_texts, max_per_premise=3)
    
    nli_output_path = os.path.join(output_dir, "korean_nli.jsonl")
    nli_generator.save_to_jsonl(nli_triples, nli_output_path)
    
    # STS 생성
    print("2. 한국어 STS 데이터 생성 중...")
    sts_generator = KoreanSTSGenerator()
    
    sts_pairs = sts_generator.generate_sts_pairs(
        sampled_texts,
        target_distribution={"high": 30, "medium": 50, "low": 20}
    )
    
    sts_output_path = os.path.join(output_dir, "korean_sts.jsonl")
    sts_generator.save_to_jsonl(sts_pairs, sts_output_path)
    
    print(f"✅ 한국어 NLI: {len(nli_triples)}개")
    print(f"✅ 한국어 STS: {len(sts_pairs)}개")

def generate_cc_kr_nli_sts(data_dir: str, output_dir: str):
    """CC-KR 기반 NLI/STS 데이터 생성"""
    print("\n=== CC-KR 기반 NLI/STS 데이터 생성 시작 ===")
    
    processor = ClassicalChineseKoreanProcessor()
    
    # 기존 CC-KR 데이터 로드
    saseo_path = os.path.join(data_dir, "sigwon", "SSDB", "saseo_qwen3_chat.jsonl")
    sigwon_path = os.path.join(data_dir, "sigwon.csv")
    
    processor.load_existing_data(saseo_path, sigwon_path)
    
    # CC-KR 데이터셋 생성 및 저장
    processor.save_datasets(output_dir)

def create_requirements_file(output_dir: str):
    """필요한 패키지 목록 파일 생성"""
    requirements = [
        "pandas>=1.3.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.0.0",
        "sentence-transformers>=2.2.0",
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "jsonlines>=3.0.0"
    ]
    
    requirements_path = os.path.join(output_dir, "requirements_nli_sts.txt")
    with open(requirements_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(requirements))
    
    print(f"✅ 필요 패키지 목록 생성: {requirements_path}")

def create_dataset_summary(output_dir: str):
    """생성된 데이터셋 요약 정보 생성"""
    summary = {
        "dataset_info": {
            "generation_date": "2024-12-17",
            "algorithm_version": "Plan A - 점진적 구현",
            "description": "기존 CC-KR 데이터를 활용한 NLI/STS 데이터셋",
            "files": []
        },
        "datasets": {}
    }
    
    # 생성된 파일들 확인
    for filename in os.listdir(output_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(output_dir, filename)
            line_count = 0
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for line in f)
                
                summary["dataset_info"]["files"].append({
                    "filename": filename,
                    "size": line_count,
                    "type": "jsonl"
                })
                
                # 첫 번째 샘플 추가
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if first_line:
                        sample = json.loads(first_line)
                        summary["datasets"][filename.replace('.jsonl', '')] = {
                            "count": line_count,
                            "sample": sample
                        }
            except Exception as e:
                print(f"⚠️ {filename} 분석 실패: {e}")
    
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 데이터셋 요약 생성: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description="NLI/STS 데이터 생성 도구")
    parser.add_argument("--mode", choices=["all", "kr_only", "cc_kr_only"], 
                       default="all", help="생성 모드 선택")
    parser.add_argument("--data_dir", default="../data", 
                       help="입력 데이터 디렉토리 (기본: ../data)")
    parser.add_argument("--output_dir", default="./output", 
                       help="출력 디렉토리 (기본: ./output)")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🚀 NLI/STS 데이터 생성 시작")
    print(f"📁 데이터 디렉토리: {args.data_dir}")
    print(f"📁 출력 디렉토리: {args.output_dir}")
    print(f"🔧 모드: {args.mode}")
    
    try:
        if args.mode in ["all", "kr_only"]:
            # 한국어 텍스트 로드
            korean_texts = load_korean_text_data(args.data_dir)
            
            if korean_texts:
                generate_korean_nli_sts(korean_texts, args.output_dir)
            else:
                print("❌ 한국어 텍스트 데이터를 찾을 수 없습니다.")
        
        if args.mode in ["all", "cc_kr_only"]:
            generate_cc_kr_nli_sts(args.data_dir, args.output_dir)
        
        # 부가 파일들 생성
        create_requirements_file(args.output_dir)
        create_dataset_summary(args.output_dir)
        
        print("\n🎉 모든 작업 완료!")
        print(f"📊 결과 확인: {args.output_dir}/dataset_summary.json")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()