"""
통합 자동 재시작 스크립트 - Auto Restart with Error Handling
"""

import asyncio
import subprocess
import time
import sys
import signal
import logging
from pathlib import Path

from error_handler import setup_logging, track_error


class AutoRestartManager:
    """자동 재시작 관리자"""
    
    def __init__(self, max_retries: int = 5, retry_delay: int = 30):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.shutdown_requested = False
        
        # 로깅 설정
        setup_logging()
        self.logger = logging.getLogger('AutoRestart')
        
        # 시그널 처리
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """시그널 핸들러"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_requested = True
    
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
            # Stop existing processes
            subprocess.run(['pkill', '-f', 'ollama serve'], capture_output=True)
            time.sleep(5)
            
            # Start server
            subprocess.Popen(['ollama', 'serve'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(15)
            
            return self.check_ollama_status()
        except Exception as e:
            self.logger.error(f"Failed to restart Ollama: {e}")
            return False
    
    def clear_gpu_memory(self):
        """GPU 메모리 정리"""
        try:
            # GPU 메모리 사용량 확인
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode == 0:
                memory_used = int(result.stdout.strip())
                if memory_used > 15000:  # 15GB 이상 사용 시
                    self.logger.info(f"High GPU memory usage: {memory_used}MB, restarting Ollama...")
                    self.restart_ollama()
        except Exception:
            pass  # GPU 없거나 nvidia-smi 없는 경우 무시
    
    async def run_translation(self, command: str, input_file: str = None, output_file: str = None):
        """번역 실행"""
        from tongu_main import TonguTranslator
        
        tongu = TonguTranslator()
        
        if command == "test":
            return await tongu.test_connection()
        elif command == "sample":
            await tongu.translate_sample()
            return True
        elif command == "accn":
            await tongu.translate_accn_dataset()
            return True
        elif command == "translate" and input_file and output_file:
            await tongu.translate_file(input_file, output_file)
            return True
        else:
            print(f"Unknown command: {command}")
            return False
    
    async def run_with_auto_restart(self, command: str, input_file: str = None, output_file: str = None):
        """자동 재시작과 함께 번역 실행"""
        
        self.logger.info(f"Starting auto-restart translation: {command}")
        
        for attempt in range(1, self.max_retries + 1):
            if self.shutdown_requested:
                self.logger.info("Shutdown requested")
                return False
            
            self.logger.info(f"Attempt {attempt}/{self.max_retries}")
            
            # Ollama 상태 확인 및 시작
            if not self.check_ollama_status():
                self.logger.warning("Ollama not running, restarting...")
                if not self.restart_ollama():
                    track_error("Ollama Start Failed", "Failed to start Ollama server", 
                              attempt=attempt)
                    if attempt < self.max_retries:
                        time.sleep(self.retry_delay)
                        continue
                    return False
            
            # GPU 메모리 정리
            self.clear_gpu_memory()
            
            # 번역 실행
            try:
                success = await self.run_translation(command, input_file, output_file)
                
                if success:
                    self.logger.info(f"Translation completed after {attempt} attempts")
                    print(f"✅ 번역 완료! ({attempt}회 시도)")
                    return True
                
            except Exception as e:
                error_msg = f"Translation failed: {e}"
                self.logger.error(error_msg)
                track_error("Translation Failed", error_msg, 
                          attempt=attempt, command=command)
            
            # 재시도 대기
            if attempt < self.max_retries:
                self.logger.warning(f"Attempt {attempt} failed, waiting {self.retry_delay}s...")
                print(f"❌ 시도 {attempt} 실패, {self.retry_delay}초 후 재시도...")
                time.sleep(self.retry_delay)
            else:
                self.logger.error(f"All {self.max_retries} attempts failed")
                track_error("Critical Failure", 
                          f"All {self.max_retries} attempts failed",
                          command=command, input_file=input_file, output_file=output_file)
        
        return False


async def main():
    """메인 실행 함수"""
    
    if len(sys.argv) < 2:
        print("🐲 Tongu Auto-Restart")
        print()
        print("사용법:")
        print("  python auto_restart.py test                    # 연결 테스트")
        print("  python auto_restart.py sample                  # 샘플 번역")
        print("  python auto_restart.py accn                    # ACCN 데이터셋")
        print("  python auto_restart.py translate <input> <output>  # 파일 번역")
        print()
        print("자동 재시작, GPU 메모리 관리, 에러 알림 포함")
        return
    
    command = sys.argv[1]
    input_file = sys.argv[2] if len(sys.argv) > 2 else None
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    manager = AutoRestartManager()
    success = await manager.run_with_auto_restart(command, input_file, output_file)
    
    if success:
        print("🎉 작업 완료!")
        sys.exit(0)
    else:
        print("💥 작업 실패!")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 중단됨")
        sys.exit(130)