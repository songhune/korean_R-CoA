"""
🐲 Tongu - Korean Translation System
깔끔하고 통합된 한국어 번역 시스템

Usage:
    python tongu.py test                    # Test Ollama connection
    python tongu.py sample                  # Run sample translation
    python tongu.py accn                    # Process ACCN dataset
    python tongu.py translate <input> <output>  # Translate file
    python tongu.py restart <command>       # Run with auto-restart
"""

import asyncio
import sys
import subprocess
import time
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core import TranslationConfig, LargeScaleTranslator
from error_handler import setup_logging, get_error_handler, track_error


class Tongu:
    """통합 한국어 번역 시스템"""
    
    def __init__(self):
        # Setup logging
        setup_logging()
        
        # GPU 최적화 설정
        self.config = TranslationConfig.create_gpu_optimized_config(api_provider="ollama")
        self.config.korean_model = "jinbora/deepseek-r1-Bllossom:70b"
        self.config.english_model = "winkefinger/alma-13b:Q4_K_M"
        self.config.budget_limit = 0.0
        self.config.ollama_base_url = "http://localhost:11434"
        
        self.translator = None
        self.error_handler = get_error_handler()
    
    async def initialize(self):
        """번역기 초기화"""
        if not self.translator:
            self.translator = LargeScaleTranslator(self.config)
        return self
    
    async def test_connection(self):
        """Ollama 연결 테스트"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        print("✅ Ollama 서버 연결 성공!")
                        print("🤖 사용 가능한 모델:")
                        for model in models.get('models', []):
                            print(f"   - {model['name']}")
                        return True
                    else:
                        print("❌ Ollama 서버 연결 실패")
                        track_error("Connection Test Failed", f"HTTP {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Ollama 연결 오류: {e}")
            track_error("Connection Test Error", str(e))
            return False
    
    async def translate_sample(self):
        """샘플 번역"""
        sample_data = [
            {
                "task": "Classical Chinese to Modern Chinese",
                "data": {
                    "instruction": "请将迁骑都尉、光禄大夫、侍中。宿卫谨敕，爵位益尊，翻译为现代汉语。",
                    "input": "",
                    "output": "又升任骑都尉光禄大夫侍中。王莽在宫中值宿警卫，谨慎认真，地位越是尊贵，",
                    "history": []
                }
            },
            {
                "task": "Classical Chinese to Modern Chinese",
                "data": {
                    "instruction": "岁年丰穰，九十月禾黍登场。为春酒瓮浮新酿，",
                    "input": "",
                    "output": "庄稼丰收，九月十月禾稼登场。制成春酒飘浓香，",
                    "history": []
                }
            }
        ]
        
        await self.initialize()
        await self.translator.process_sample(sample_data, "sample_output.jsonl")
        print("✅ 샘플 번역 완료! 결과: sample_output.jsonl")
    
    async def translate_file(self, input_file: str, output_file: str):
        """파일 번역"""
        if not Path(input_file).exists():
            print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
            track_error("File Not Found", input_file)
            return False
        
        await self.initialize()
        await self.translator.process_large_dataset(input_file, output_file)
        print(f"✅ 파일 번역 완료: {input_file} -> {output_file}")
        return True
    
    async def translate_accn_dataset(self):
        """ACCN-INS 데이터셋 번역"""
        accn_file = "/home/work/songhune/ACCN-INS.json"
        output_file = "accn_ins_multilingual.jsonl"
        
        if not Path(accn_file).exists():
            print(f"❌ ACCN 파일을 찾을 수 없습니다: {accn_file}")
            track_error("ACCN File Not Found", accn_file)
            return False
        
        await self.initialize()
        await self.translator.process_large_dataset(accn_file, output_file)
        print("✅ ACCN-INS 데이터셋 번역 완료!")
        return True
    
    def check_ollama_status(self) -> bool:
        """Ollama 서버 상태 확인"""
        try:
            result = subprocess.run(
                ['curl', '-s', 'http://localhost:11434/api/tags'],
                capture_output=True, timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def restart_ollama(self) -> bool:
        """Ollama 서버 재시작"""
        try:
            print("🔄 Ollama 재시작 중...")
            subprocess.run(['pkill', '-f', 'ollama serve'], capture_output=True)
            time.sleep(5)
            
            subprocess.Popen(['ollama', 'serve'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            time.sleep(15)
            
            return self.check_ollama_status()
        except Exception as e:
            track_error("Ollama Restart Failed", str(e))
            return False
    
    async def run_with_restart(self, command: str, input_file: str = None, output_file: str = None, max_retries: int = 3):
        """자동 재시작과 함께 실행"""
        
        for attempt in range(1, max_retries + 1):
            print(f"📋 시도 {attempt}/{max_retries}")
            
            # Ollama 상태 확인
            if not self.check_ollama_status():
                print("⚠️ Ollama 서버가 실행되지 않음")
                if not self.restart_ollama():
                    print("❌ Ollama 재시작 실패")
                    continue
            
            try:
                # 명령 실행
                if command == "sample":
                    await self.translate_sample()
                    return True
                elif command == "accn":
                    return await self.translate_accn_dataset()
                elif command == "translate" and input_file and output_file:
                    return await self.translate_file(input_file, output_file)
                elif command == "test":
                    return await self.test_connection()
                else:
                    print(f"❌ 알 수 없는 명령: {command}")
                    return False
                
            except Exception as e:
                error_msg = f"실행 실패: {e}"
                print(f"❌ {error_msg}")
                track_error("Execution Failed", error_msg, attempt=attempt, command=command)
                
                if attempt < max_retries:
                    print(f"😴 30초 후 재시도...")
                    time.sleep(30)
                else:
                    print(f"💥 모든 시도 실패 ({max_retries}회)")
                    return False
        
        return False


async def main():
    """메인 실행 함수"""
    
    if len(sys.argv) < 2:
        print("🐲 Tongu - Korean Translation System")
        print("=" * 50)
        print()
        print("사용법:")
        print("  python tongu.py test                         # Ollama 연결 테스트")
        print("  python tongu.py sample                       # 샘플 번역")
        print("  python tongu.py accn                         # ACCN 데이터셋 번역")  
        print("  python tongu.py translate <input> <output>   # 파일 번역")
        print("  python tongu.py restart <command>            # 자동 재시작과 함께 실행")
        print()
        print("특징:")
        print("  🔧 Broken pipe 문제 해결")
        print("  🚨 자동 에러 알림 (songhune@jou.ac.kr)")
        print("  🔄 자동 재시작 및 GPU 메모리 관리")
        print("  📊 실시간 에러 모니터링")
        print()
        return
    
    command = sys.argv[1]
    tongu = Tongu()
    
    try:
        if command == "restart":
            # 자동 재시작 모드
            if len(sys.argv) < 3:
                print("사용법: python tongu.py restart <command> [args...]")
                return
            
            restart_cmd = sys.argv[2]
            input_file = sys.argv[3] if len(sys.argv) > 3 else None
            output_file = sys.argv[4] if len(sys.argv) > 4 else None
            
            success = await tongu.run_with_restart(restart_cmd, input_file, output_file)
            
        else:
            # 일반 모드
            if command == "test":
                success = await tongu.test_connection()
                
            elif command == "sample":
                await tongu.translate_sample()
                success = True
                
            elif command == "accn":
                success = await tongu.translate_accn_dataset()
                
            elif command == "translate":
                if len(sys.argv) != 4:
                    print("사용법: python tongu.py translate <input_file> <output_file>")
                    return
                input_file = sys.argv[2]
                output_file = sys.argv[3]
                success = await tongu.translate_file(input_file, output_file)
                
            else:
                print(f"❌ 알 수 없는 명령: {command}")
                success = False
        
        # 에러 요약 출력
        error_summary = tongu.error_handler.get_error_summary()
        if error_summary['active_errors']:
            print(f"\n⚠️ 활성 에러: {error_summary['active_errors']}")
        
        if success:
            print("🎉 완료!")
        else:
            print("💥 실패!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 중단됨")
        sys.exit(130)
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        track_error("Unexpected Error", str(e))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())