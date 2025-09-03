"""파일 입출력 처리 모듈"""

import json
import asyncio
import aiofiles
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator
import logging


class FileHandler:
    """비동기 파일 처리기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def read_json_or_jsonl_stream(self, file_path: str, chunk_size: int = 1000) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """JSON 배열 또는 JSONL 파일을 청크 단위로 스트리밍 읽기"""
        try:
            # 먼저 전체 파일을 읽어서 JSON 배열인지 확인
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as file:
                content = await file.read()
                content = content.strip()
                
                # JSON 배열 형식인 경우
                if content.startswith('[') and content.endswith(']'):
                    try:
                        data = json.loads(content)
                        self.logger.info(f"Detected JSON array format with {len(data)} items")
                        
                        # 청크 단위로 yield
                        for i in range(0, len(data), chunk_size):
                            chunk = data[i:i + chunk_size]
                            self.logger.debug(f"Yielding chunk with {len(chunk)} items")
                            yield chunk
                        return
                    except json.JSONDecodeError:
                        self.logger.warning("File looks like JSON array but failed to parse, trying JSONL format")
                
                # JSONL 형식으로 다시 시도
                current_chunk = []
                lines = content.split('\n')
                
                for line_count, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        item = json.loads(line)
                        current_chunk.append(item)
                        
                        # 청크 크기에 도달하면 yield
                        if len(current_chunk) >= chunk_size:
                            self.logger.debug(f"Yielding chunk with {len(current_chunk)} items")
                            yield current_chunk
                            current_chunk = []
                    
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Skipping malformed JSON at line {line_count}: {e}")
                        continue
                
                # 마지막 청크 처리
                if current_chunk:
                    self.logger.debug(f"Yielding final chunk with {len(current_chunk)} items")
                    yield current_chunk
                    
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    async def write_jsonl(self, items: List[Dict[str, Any]], file_path: str):
        """JSONL 형태로 파일 쓰기"""
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                for item in items:
                    await f.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.logger.info(f"Wrote {len(items)} items to {file_path}")
        except Exception as e:
            self.logger.error(f"Error writing to {file_path}: {e}")
            raise
    
    async def append_jsonl(self, items: List[Dict[str, Any]], file_path: str):
        """JSONL 파일에 항목 추가"""
        try:
            async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                for item in items:
                    await f.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.logger.debug(f"Appended {len(items)} items to {file_path}")
        except Exception as e:
            self.logger.error(f"Error appending to {file_path}: {e}")
            raise
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """파일 정보 조회"""
        path = Path(file_path)
        
        if not path.exists():
            return {"exists": False}
        
        stat = path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified_time": stat.st_mtime,
            "is_file": path.is_file()
        }


class CheckpointManager:
    """체크포인트 관리자"""
    
    def __init__(self, output_dir: str = "./checkpoints"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    async def save_checkpoint(self, data: List[Dict[str, Any]], chunk_id: int, base_name: str):
        """체크포인트 저장"""
        checkpoint_file = self.output_dir / f"{base_name}_chunk_{chunk_id:04d}.jsonl"
        
        file_handler = FileHandler()
        await file_handler.write_jsonl(data, str(checkpoint_file))
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
        return checkpoint_file
    
    async def load_checkpoint(self, checkpoint_file: str) -> List[Dict[str, Any]]:
        """체크포인트 로드"""
        items = []
        file_handler = FileHandler()
        
        async for chunk in file_handler.read_json_or_jsonl_stream(checkpoint_file):
            items.extend(chunk)
        
        self.logger.info(f"Loaded {len(items)} items from checkpoint: {checkpoint_file}")
        return items
    
    def list_checkpoints(self, base_name: str) -> List[Path]:
        """저장된 체크포인트 파일 목록"""
        pattern = f"{base_name}_chunk_*.jsonl"
        checkpoints = list(self.output_dir.glob(pattern))
        checkpoints.sort()
        return checkpoints
    
    async def merge_checkpoints(self, base_name: str, output_file: str):
        """체크포인트 파일들을 하나로 합치기"""
        checkpoints = self.list_checkpoints(base_name)
        
        if not checkpoints:
            self.logger.warning(f"No checkpoints found for {base_name}")
            return
        
        file_handler = FileHandler()
        total_items = 0
        
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as output_f:
            for checkpoint_file in checkpoints:
                self.logger.info(f"Merging {checkpoint_file}")
                
                async for chunk in file_handler.read_json_or_jsonl_stream(str(checkpoint_file)):
                    for item in chunk:
                        await output_f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        total_items += 1
        
        self.logger.info(f"Merged {len(checkpoints)} checkpoints into {output_file} ({total_items} items)")
    
    def cleanup_checkpoints(self, base_name: str):
        """체크포인트 파일들 정리"""
        checkpoints = self.list_checkpoints(base_name)
        
        for checkpoint_file in checkpoints:
            try:
                checkpoint_file.unlink()
                self.logger.debug(f"Removed checkpoint: {checkpoint_file}")
            except Exception as e:
                self.logger.warning(f"Failed to remove {checkpoint_file}: {e}")
        
        self.logger.info(f"Cleaned up {len(checkpoints)} checkpoint files")