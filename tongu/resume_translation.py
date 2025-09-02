#!/usr/bin/env python3
"""체크포인트에서 번역 재개하기"""

import asyncio
from pathlib import Path
from file_handlers import CheckpointManager

async def merge_existing_checkpoints():
    """기존 체크포인트 파일들을 최종 결과로 병합"""
    
    checkpoint_manager = CheckpointManager()
    
    # 사용 가능한 체크포인트 패턴들 확인
    checkpoint_patterns = [
        "sample_ollama_translated",
        "accn_ins_multilingual", 
        "accn_sample_translated"
    ]
    
    print("=== 사용 가능한 체크포인트 확인 ===")
    
    for pattern in checkpoint_patterns:
        checkpoints = checkpoint_manager.list_checkpoints(pattern)
        
        if checkpoints:
            print(f"\n{pattern}: {len(checkpoints)}개 체크포인트 발견")
            print(f"   범위: chunk_0000 ~ chunk_{len(checkpoints)-1:04d}")
            
            # 병합 여부 확인
            output_file = f"{pattern}_merged.jsonl"
            response = input(f"   -> {output_file}로 병합하시겠습니까? (y/n): ")
            
            if response.lower() == 'y':
                print(f"   병합 중...")
                await checkpoint_manager.merge_checkpoints(pattern, output_file)
                print(f"   완료: {output_file}")
                
                # 정리 여부 확인
                cleanup = input(f"   체크포인트 파일들을 삭제하시겠습니까? (y/n): ")
                if cleanup.lower() == 'y':
                    checkpoint_manager.cleanup_checkpoints(pattern)
                    print(f"   체크포인트 정리 완료")
        else:
            print(f"{pattern}: 체크포인트 없음")

async def resume_from_checkpoint():
    """특정 체크포인트부터 번역 재개"""
    
    checkpoint_manager = CheckpointManager()
    
    print("=== 체크포인트에서 재개 ===")
    
    # 패턴 입력받기
    base_name = input("프로젝트 이름 (예: sample_ollama_translated): ").strip()
    
    if not base_name:
        print("프로젝트 이름을 입력해주세요.")
        return
    
    # 체크포인트 확인
    checkpoints = checkpoint_manager.list_checkpoints(base_name)
    
    if not checkpoints:
        print(f"❌ '{base_name}' 체크포인트를 찾을 수 없습니다.")
        return
    
    print(f"📁 {len(checkpoints)}개 체크포인트 발견")
    
    # 마지막 체크포인트 확인
    last_checkpoint = checkpoints[-1]
    last_chunk_id = int(last_checkpoint.stem.split('_')[-1])
    
    print(f"📄 마지막 체크포인트: chunk_{last_chunk_id:04d}")
    
    # 선택지 제공
    print("\n선택하세요:")
    print("1. 모든 체크포인트를 병합하여 최종 파일 생성")
    print("2. 마지막 체크포인트부터 번역 재개")
    print("3. 특정 체크포인트부터 번역 재개")
    
    choice = input("선택 (1-3): ").strip()
    
    if choice == "1":
        output_file = f"{base_name}_final.jsonl"
        print(f"{output_file}로 병합 중...")
        await checkpoint_manager.merge_checkpoints(base_name, output_file)
        print(f"완료: {output_file}")
        
    elif choice == "2":
        print(f"chunk_{last_chunk_id:04d} 이후부터 재개...")
        print("TODO: translator.py에서 resume_from_chunk() 메서드 구현 필요")
        
    elif choice == "3":
        start_chunk = input(f"시작할 청크 번호 (0-{last_chunk_id}): ").strip()
        try:
            start_chunk_id = int(start_chunk)
            print(f"chunk_{start_chunk_id:04d} 부터 재개...")
            print("TODO: translator.py에서 resume_from_chunk() 메서드 구현 필요")
        except ValueError:
            print("올바른 숫자를 입력해주세요.")

def main():
    """메인 함수"""
    print("KEadapter - 체크포인트 관리 도구")
    print()
    print("1. 기존 체크포인트 병합")
    print("2. 체크포인트에서 재개")
    print()
    
    choice = input("선택 (1-2): ").strip()
    
    if choice == "1":
        asyncio.run(merge_existing_checkpoints())
    elif choice == "2":
        asyncio.run(resume_from_checkpoint())
    else:
        print("올바른 선택을 해주세요.")

if __name__ == "__main__":
    main()