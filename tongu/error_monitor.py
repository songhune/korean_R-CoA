#!/usr/bin/env python3
"""번역 오류 모니터링 도구"""

import json
import asyncio
from pathlib import Path
from collections import defaultdict, Counter

def analyze_translation_errors(checkpoint_dir: str = "./checkpoints"):
    """체크포인트 파일들에서 번역 오류 분석"""
    
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        print(f" 체크포인트 디렉토리를 찾을 수 없습니다: {checkpoint_dir}")
        return
    
    # 체크포인트 파일들 찾기
    checkpoint_files = list(checkpoint_path.glob("*.jsonl"))
    
    if not checkpoint_files:
        print(f" {checkpoint_dir}에서 체크포인트 파일을 찾을 수 없습니다.")
        return
    
    print(f" {len(checkpoint_files)}개 체크포인트 파일 분석 중...")
    
    # 오류 통계
    error_stats = {
        "total_items": 0,
        "korean_errors": 0,
        "english_errors": 0,
        "both_errors": 0,
        "successful_translations": 0,
        "error_types": Counter(),
        "files_with_errors": []
    }
    
    for file_path in sorted(checkpoint_files):
        file_errors = analyze_single_file(file_path)
        
        # 통계 업데이트
        error_stats["total_items"] += file_errors["total"]
        error_stats["korean_errors"] += file_errors["korean_errors"]
        error_stats["english_errors"] += file_errors["english_errors"]
        error_stats["both_errors"] += file_errors["both_errors"]
        error_stats["successful_translations"] += file_errors["successful"]
        error_stats["error_types"].update(file_errors["error_types"])
        
        if file_errors["has_errors"]:
            error_stats["files_with_errors"].append(file_path.name)
    
    # 결과 출력
    print_error_summary(error_stats)
    
    return error_stats

def analyze_single_file(file_path: Path):
    """단일 체크포인트 파일 분석"""
    
    stats = {
        "total": 0,
        "korean_errors": 0,
        "english_errors": 0,
        "both_errors": 0,
        "successful": 0,
        "has_errors": False,
        "error_types": Counter()
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    item = json.loads(line)
                    stats["total"] += 1
                    
                    # 번역 결과 확인
                    korean_trans = item.get("korean_translation", "")
                    english_trans = item.get("english_translation", "")
                    
                    korean_is_error = is_translation_error(korean_trans)
                    english_is_error = is_translation_error(english_trans)
                    
                    if korean_is_error and english_is_error:
                        stats["both_errors"] += 1
                        stats["has_errors"] = True
                        extract_error_type(korean_trans, stats["error_types"])
                    elif korean_is_error:
                        stats["korean_errors"] += 1
                        stats["has_errors"] = True
                        extract_error_type(korean_trans, stats["error_types"])
                    elif english_is_error:
                        stats["english_errors"] += 1
                        stats["has_errors"] = True
                        extract_error_type(english_trans, stats["error_types"])
                    else:
                        stats["successful"] += 1
                        
                except json.JSONDecodeError:
                    continue
                    
    except Exception as e:
        print(f"파일 읽기 오류 {file_path.name}: {e}")
    
    return stats

def is_translation_error(translation):
    """번역 결과가 오류인지 확인"""
    if isinstance(translation, str):
        return (
            translation.startswith("[Translation Error") or
            translation == "" or
            "Translation Error" in translation
        )
    elif isinstance(translation, list):
        return any(
            str(item).startswith("[Translation Error") or
            str(item) == "" or
            "Translation Error" in str(item)
            for item in translation
        )
    elif isinstance(translation, dict):
        # 토큰 정보만 있고 실제 번역이 없는 경우
        return "input_tokens" in translation and "output_tokens" in translation
    else:
        return True

def extract_error_type(error_text, error_counter):
    """오류 유형 추출"""
    if isinstance(error_text, str):
        if "서버 연결 실패" in error_text or "Connection" in error_text:
            error_counter["연결 실패"] += 1
        elif "API Error" in error_text:
            error_counter["API 오류"] += 1
        elif error_text == "":
            error_counter["빈 응답"] += 1
        else:
            error_counter["기타"] += 1
    elif isinstance(error_text, list):
        for item in error_text:
            if "[Translation Error" in str(item):
                extract_error_type(str(item), error_counter)
    elif isinstance(error_text, dict):
        error_counter["토큰 정보만"] += 1

def print_error_summary(stats):
    """오류 통계 요약 출력"""
    
    print(f"\n 번역 오류 분석 결과")
    print("=" * 50)
    
    print(f"총 처리 항목: {stats['total_items']:,}")
    print(f"성공한 번역: {stats['successful_translations']:,}")
    print(f"한국어 오류: {stats['korean_errors']:,}")
    print(f"영어 오류: {stats['english_errors']:,}")
    print(f"양쪽 오류: {stats['both_errors']:,}")
    
    if stats['total_items'] > 0:
        success_rate = (stats['successful_translations'] / stats['total_items']) * 100
        print(f"성공률: {success_rate:.1f}%")
    
    if stats['error_types']:
        print(f"\n오류 유형별 통계:")
        for error_type, count in stats['error_types'].most_common():
            print(f"  {error_type}: {count:,}회")
    
    if stats['files_with_errors']:
        print(f"\n오류가 있는 파일 ({len(stats['files_with_errors'])}개):")
        for filename in stats['files_with_errors'][:10]:  # 처음 10개만 표시
            print(f"  - {filename}")
        
        if len(stats['files_with_errors']) > 10:
            print(f"  ... 및 {len(stats['files_with_errors']) - 10}개 추가 파일")

def main():
    """메인 함수"""
    print("KEadapter - 번역 오류 분석 도구")
    print()
    
    checkpoint_dir = input("체크포인트 디렉토리 경로 (기본값: ./checkpoints): ").strip()
    if not checkpoint_dir:
        checkpoint_dir = "./checkpoints"
    
    analyze_translation_errors(checkpoint_dir)

if __name__ == "__main__":
    main()