"""
통합 에러 처리 및 알림 시스템
Unified error handling and notification system
"""

import logging
import time
import smtplib
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


class ErrorHandler:
    """통합 에러 처리 및 알림 시스템"""
    
    def __init__(self, email: str = "songhune@jou.ac.kr", threshold: int = 5, time_window: int = 300):
        self.email = email
        self.threshold = threshold
        self.time_window = time_window
        
        # Error tracking
        self.error_counts = defaultdict(lambda: deque())
        self.last_notification = {}
        self.notification_cooldown = 1800  # 30분
        
        # Setup logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('ErrorHandler')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler('translation_errors.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def track_error(self, error_type: str, error_message: str, **context):
        """에러 추적 및 알림 체크"""
        current_time = time.time()
        
        # 에러 추가
        self.error_counts[error_type].append(current_time)
        
        # 시간 윈도우 밖의 에러 제거
        cutoff_time = current_time - self.time_window
        while (self.error_counts[error_type] and 
               self.error_counts[error_type][0] < cutoff_time):
            self.error_counts[error_type].popleft()
        
        # 로깅
        self.logger.error(f"{error_type}: {error_message}")
        if context:
            self.logger.error(f"Context: {context}")
        
        # 알림 체크
        error_count = len(self.error_counts[error_type])
        if error_count >= self.threshold:
            self._check_and_notify(error_type, error_count, error_message, context)
    
    def _check_and_notify(self, error_type: str, error_count: int, error_message: str, context: Dict):
        """알림 필요 여부 체크 및 전송"""
        current_time = time.time()
        last_notified = self.last_notification.get(error_type, 0)
        
        if current_time - last_notified < self.notification_cooldown:
            return
        
        self._send_notification(error_type, error_count, error_message, context)
        self.last_notification[error_type] = current_time
    
    def _send_notification(self, error_type: str, error_count: int, error_message: str, context: Dict):
        """알림 전송 (콘솔 출력)"""
        try:
            subject = f"🚨 Korean Translation System Alert: {error_type}"
            
            message = f"""
Korean Translation System Error Alert
====================================

Error Type: {error_type}
Count: {error_count} errors in {self.time_window // 60} minutes
Threshold: {self.threshold}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Latest Error: {error_message}

Context: {context}

Recommended Actions:
1. Check Ollama server: ollama ps
2. Restart if needed: ollama serve  
3. Check GPU memory: nvidia-smi
4. Review logs: translation_errors.log

This is an automated alert from Tongu Translation System.
            """
            
            # 콘솔에 알림 출력
            print(f"\n📧 ERROR NOTIFICATION SENT TO {self.email}")
            print("=" * 60)
            print(f"Subject: {subject}")
            print(message)
            print("=" * 60)
            
            self.logger.info(f"Notification sent for {error_type} ({error_count} errors)")
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
    
    def get_error_summary(self) -> Dict:
        """현재 에러 요약"""
        current_time = time.time()
        cutoff_time = current_time - self.time_window
        
        active_errors = {}
        for error_type, error_times in self.error_counts.items():
            # 시간 윈도우 내의 에러만 카운트
            recent_errors = [t for t in error_times if t > cutoff_time]
            if recent_errors:
                active_errors[error_type] = len(recent_errors)
        
        return {
            'active_errors': active_errors,
            'time_window_minutes': self.time_window // 60,
            'threshold': self.threshold
        }


# 전역 에러 핸들러 인스턴스
_error_handler = None

def get_error_handler() -> ErrorHandler:
    """전역 에러 핸들러 인스턴스 반환"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def track_error(error_type: str, message: str, **context):
    """에러 추적 편의 함수"""
    get_error_handler().track_error(error_type, message, **context)


def track_broken_pipe_error(message: str, **context):
    """Broken Pipe 에러 추적"""
    track_error("Broken Pipe", message, **context)


def track_connection_error(message: str, **context):
    """연결 에러 추적"""
    track_error("Connection Error", message, **context)


def track_model_error(message: str, **context):
    """모델 에러 추적"""
    track_error("Model Error", message, **context)


# 로깅 설정 함수
def setup_logging():
    """전체 시스템 로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('translation.log'),
            logging.StreamHandler()
        ]
    )
    
    # 불필요한 로깅 비활성화
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)


if __name__ == "__main__":
    # 테스트
    handler = ErrorHandler("test@example.com", threshold=2, time_window=60)
    
    print("Testing error tracking...")
    for i in range(3):
        handler.track_error("Test Error", f"Test message {i+1}", test_id=i+1)
        time.sleep(1)
    
    summary = handler.get_error_summary()
    print(f"Error summary: {summary}")