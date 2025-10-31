#!/usr/bin/env python3
"""
3번 실험: 전체 데이터 번역 스크립트
비용: 약 $23 (₩31,000)
소요시간: 약 1시간

사용법:
    python run_translation.py --abstract --content

옵션:
    --abstract : Abstract 번역 실행
    --content  : Content 번역 실행
    --test N   : 테스트 모드 (N개만 번역)
"""

import os
import sys
import json
import pandas as pd
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from tqdm import tqdm
from datetime import datetime

# 프로젝트 루트 경로
BASE = Path(__file__).parent.parent.parent
IN_JSONL = BASE / "notebook/eda_outputs/1번실험/triples_no_answer.jsonl"
OUT_DIR = BASE / "notebook/experimen../../results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# .env 로드
env_path = BASE / "notebook/experiments/.env"
load_dotenv(env_path)

# API 클라이언트 초기화
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print(" ANTHROPIC_API_KEY가 .env 파일에 없습니다!")
    sys.exit(1)

client = anthropic.Anthropic(api_key=api_key)


def load_data():
    """데이터 로드"""
    print("\n📂 데이터 로드 중...")
    questions = []
    with open(IN_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                q_data = {"question_id": data["question"]["id"]}

                for triple in data["triples"]:
                    if triple["s"].startswith("Q"):
                        if triple["p"] == "hasAbstract":
                            q_data["abstract"] = triple["o"]
                        elif triple["p"] == "hasContent":
                            q_data["content"] = triple["o"]
                        elif triple["p"] == "hasCategory":
                            q_data["category"] = triple["o"]

                questions.append(q_data)

    df = pd.DataFrame(questions)
    df['has_abstract'] = df['abstract'].notna() & (df['abstract'] != '')
    df['has_content'] = df['content'].notna() & (df['content'] != '')

    print(f" 총 {len(df):,}개 문제 로드")
    print(f"   - Abstract 있음: {df['has_abstract'].sum():,}")
    print(f"   - Content 있음: {df['has_content'].sum():,}")

    return df


def translate_text(text, target_lang="Korean", max_retries=3):
    """
    한문 텍스트를 번역

    Args:
        text: 번역할 한문 텍스트
        target_lang: "Korean" 또는 "English"
        max_retries: 최대 재시도 횟수

    Returns:
        번역된 텍스트
    """
    if not text or pd.isna(text) or text.strip() == "":
        return ""

    prompt = f"""다음 한문 텍스트를 {target_lang}로 번역해주세요.
번역만 출력하고, 추가 설명은 하지 마세요.

한문: {text}

{target_lang} 번역:"""

    for attempt in range(max_retries):
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except anthropic.RateLimitError as e:
            wait_time = 2 ** attempt
            print(f" Rate limit - {wait_time}초 대기...")
            time.sleep(wait_time)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f" 재시도 {attempt+1}/{max_retries}: {str(e)[:50]}")
                time.sleep(2 ** attempt)
            else:
                print(f" 번역 실패: {str(e)[:100]}")
                return ""

    return ""


def translate_abstract(df, test_limit=None):
    """Abstract 번역 (한국어 + 영어)"""
    print("\n" + "="*80)
    print(" Abstract 번역 시작")
    print("="*80)

    df['abstract_ko'] = ""
    df['abstract_en'] = ""

    abstract_rows = df[df['has_abstract']]
    if test_limit:
        abstract_rows = abstract_rows.head(test_limit)
        print(f" 테스트 모드: {len(abstract_rows)}개만 번역")

    print(f"\n총 {len(abstract_rows):,}개 Abstract 번역")

    # 한국어 번역
    print("\n🇰🇷 한국어 번역 중...")
    for idx, row in tqdm(abstract_rows.iterrows(), total=len(abstract_rows), desc="Korean"):
        df.at[idx, 'abstract_ko'] = translate_text(row['abstract'], "Korean")
        time.sleep(0.5)  # Rate limit 방지

    # 영어 번역
    print("\n🇺🇸 영어 번역 중...")
    for idx, row in tqdm(abstract_rows.iterrows(), total=len(abstract_rows), desc="English"):
        df.at[idx, 'abstract_en'] = translate_text(row['abstract'], "English")
        time.sleep(0.5)

    print(" Abstract 번역 완료")
    return df


def translate_content(df, test_limit=None):
    """Content 번역 (한국어 + 영어)"""
    print("\n" + "="*80)
    print(" Content 번역 시작")
    print("="*80)

    df['content_ko'] = ""
    df['content_en'] = ""

    content_rows = df[df['has_content']]
    if test_limit:
        content_rows = content_rows.head(test_limit)
        print(f" 테스트 모드: {len(content_rows)}개만 번역")

    print(f"\n총 {len(content_rows):,}개 Content 번역")

    # 한국어 번역
    print("\n🇰🇷 한국어 번역 중...")
    for idx, row in tqdm(content_rows.iterrows(), total=len(content_rows), desc="Korean"):
        df.at[idx, 'content_ko'] = translate_text(row['content'], "Korean")
        time.sleep(0.5)

    # 영어 번역
    print("\n🇺🇸 영어 번역 중...")
    for idx, row in tqdm(content_rows.iterrows(), total=len(content_rows), desc="English"):
        df.at[idx, 'content_en'] = translate_text(row['content'], "English")
        time.sleep(0.5)

    print(" Content 번역 완료")
    return df


def save_results(df, translate_abstract_flag, translate_content_flag):
    """결과 저장"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUT_DIR / f"translated_full_{timestamp}.csv"

    # 저장할 컬럼 선택
    save_cols = ['question_id', 'category', 'abstract', 'content']
    if translate_abstract_flag:
        save_cols.extend(['abstract_ko', 'abstract_en'])
    if translate_content_flag:
        save_cols.extend(['content_ko', 'content_en'])

    df[save_cols].to_csv(output_file, index=False, encoding='utf-8-sig')

    print("\n" + "="*80)
    print(" 번역 완료!")
    print("="*80)
    print(f"📁 저장 위치: {output_file}")
    print(f" 총 {len(df):,}개 문제")

    if translate_abstract_flag:
        translated = (df['abstract_ko'] != "").sum()
        print(f"   - Abstract 번역: {translated:,}개")

    if translate_content_flag:
        translated = (df['content_ko'] != "").sum()
        print(f"   - Content 번역: {translated:,}개")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='3번 실험: 과시 데이터 번역')
    parser.add_argument('--abstract', action='store_true', help='Abstract 번역')
    parser.add_argument('--content', action='store_true', help='Content 번역')
    parser.add_argument('--test', type=int, metavar='N', help='테스트 모드 (N개만 번역)')

    args = parser.parse_args()

    # 옵션 검증
    if not args.abstract and not args.content:
        print(" --abstract 또는 --content 옵션을 지정해주세요")
        parser.print_help()
        sys.exit(1)

    # 시작
    print("="*80)
    print(" 3번 실험: 과시 데이터 번역 시작")
    print("="*80)
    print(f" Abstract 번역: {'' if args.abstract else ''}")
    print(f" Content 번역: {'' if args.content else ''}")
    if args.test:
        print(f" 테스트 모드: {args.test}개씩만 번역")
    else:
        print(f" 예상 비용: ~$23 (약 ₩31,000)")
        print(f"⏱  예상 시간: ~1시간")
    print("="*80)

    # 확인
    if not args.test:
        response = input("\n진행하시겠습니까? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print(" 취소되었습니다")
            sys.exit(0)

    # 데이터 로드
    df = load_data()

    # 번역 실행
    start_time = time.time()

    try:
        if args.abstract:
            df = translate_abstract(df, test_limit=args.test)

        if args.content:
            df = translate_content(df, test_limit=args.test)

        # 결과 저장
        save_results(df, args.abstract, args.content)

    except KeyboardInterrupt:
        print("\n\n 사용자가 중단했습니다")
        print(" 현재까지 번역된 결과를 저장합니다...")
        save_results(df, args.abstract, args.content)
        sys.exit(1)

    # 소요 시간
    elapsed = time.time() - start_time
    print(f"\n⏱  소요 시간: {elapsed/60:.1f}분")


if __name__ == "__main__":
    main()
