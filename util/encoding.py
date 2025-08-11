import pandas as pd
import chardet
import codecs
from pathlib import Path

class EncodingDetector:
    """CSV 파일의 인코딩을 감지하고 수정하는 클래스"""
    
    def __init__(self):
        self.common_encodings = [
            'utf-8', 'utf-8-sig', 'euc-kr', 'cp949', 
            'iso-8859-1', 'cp1252', 'latin1', 'ascii'
        ]
    
    def detect_file_encoding(self, file_path: str) -> dict:
        """파일의 인코딩을 자동 감지"""
        with open(file_path, 'rb') as f:
            raw_data = f.read(100000)  # 처음 100KB만 읽어서 감지
        
        detected = chardet.detect(raw_data)
        print(f"🔍 자동 감지 결과: {detected}")
        
        return detected
    
    def try_read_with_encodings(self, file_path: str, sample_rows: int = 5):
        """다양한 인코딩으로 읽기 시도"""
        print(f"\n=== {file_path} 인코딩 테스트 ===")
        
        results = {}
        
        for encoding in self.common_encodings:
            try:
                # pandas로 읽기 시도
                df = pd.read_csv(file_path, encoding=encoding, nrows=sample_rows)
                
                # 첫 번째 컬럼의 첫 번째 값을 샘플로 확인
                sample_text = ""
                if len(df.columns) > 0 and len(df) > 0:
                    first_col = df.columns[0]
                    if pd.notna(df.iloc[0][first_col]):
                        sample_text = str(df.iloc[0][first_col])[:100]
                
                results[encoding] = {
                    'success': True,
                    'sample': sample_text,
                    'shape': df.shape,
                    'columns': list(df.columns)[:5]  # 처음 5개 컬럼만
                }
                
                print(f"✅ {encoding:12} | 샘플: {sample_text[:50]}...")
                
            except Exception as e:
                results[encoding] = {
                    'success': False,
                    'error': str(e)[:100]
                }
                print(f"❌ {encoding:12} | 에러: {str(e)[:50]}...")
        
        return results
    
    def check_korean_characters(self, text: str) -> dict:
        """한글 문자 포함 여부 및 품질 검사"""
        if not text:
            return {'has_korean': False, 'quality': 'empty'}
        
        # 한글 문자 범위 체크
        korean_chars = 0
        broken_chars = 0
        total_chars = len(text)
        
        for char in text:
            # 한글 완성형 (가-힣)
            if '\uAC00' <= char <= '\uD7A3':
                korean_chars += 1
            # 한글 자모 (ㄱ-ㅎ, ㅏ-ㅣ)
            elif '\u3131' <= char <= '\u318E':
                korean_chars += 1
            # 깨진 문자 패턴
            elif char in 'ÃÊ½Ì³âÁ¦´ë°úËÏ×øéö':
                broken_chars += 1
        
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        broken_ratio = broken_chars / total_chars if total_chars > 0 else 0
        
        # 품질 판정
        if broken_ratio > 0.1:
            quality = 'broken'
        elif korean_ratio > 0.3:
            quality = 'good'
        elif korean_ratio > 0.1:
            quality = 'fair'
        else:
            quality = 'poor'
        
        return {
            'has_korean': korean_chars > 0,
            'korean_ratio': korean_ratio,
            'broken_ratio': broken_ratio,
            'quality': quality,
            'korean_chars': korean_chars,
            'broken_chars': broken_chars
        }
    
    def find_best_encoding(self, file_path: str) -> str:
        """가장 적합한 인코딩 찾기"""
        print(f"\n🎯 {file_path} 최적 인코딩 찾기")
        
        # 1. 자동 감지
        detected = self.detect_file_encoding(file_path)
        
        # 2. 다양한 인코딩 시도
        results = self.try_read_with_encodings(file_path)
        
        # 3. 품질 평가
        best_encoding = None
        best_score = -1
        
        print(f"\n📊 품질 평가:")
        for encoding, result in results.items():
            if not result['success']:
                continue
            
            quality_info = self.check_korean_characters(result['sample'])
            
            # 점수 계산 (한글 비율 높고, 깨진 문자 적을수록 좋음)
            score = quality_info['korean_ratio'] - quality_info['broken_ratio'] * 2
            
            print(f"{encoding:12} | 품질: {quality_info['quality']:6} | 점수: {score:.3f} | 한글: {quality_info['korean_ratio']:.1%} | 깨짐: {quality_info['broken_ratio']:.1%}")
            
            if score > best_score:
                best_score = score
                best_encoding = encoding
        
        print(f"\n🏆 최적 인코딩: {best_encoding} (점수: {best_score:.3f})")
        return best_encoding
    
    def convert_file_encoding(self, input_path: str, output_path: str, 
                             source_encoding: str, target_encoding: str = 'utf-8'):
        """파일 인코딩 변환"""
        try:
            with open(input_path, 'r', encoding=source_encoding) as f:
                content = f.read()
            
            with open(output_path, 'w', encoding=target_encoding) as f:
                f.write(content)
            
            print(f"✅ 변환 완료: {input_path} → {output_path}")
            print(f"   {source_encoding} → {target_encoding}")
            
        except Exception as e:
            print(f"❌ 변환 실패: {e}")
    
    def check_all_files(self, file_paths: list):
        """여러 파일의 인코딩을 일괄 점검"""
        print("=== 전체 파일 인코딩 점검 ===\n")
        
        results = {}
        for file_path in file_paths:
            if Path(file_path).exists():
                best_encoding = self.find_best_encoding(file_path)
                results[file_path] = best_encoding
            else:
                print(f"❌ 파일 없음: {file_path}")
                results[file_path] = None
        
        print(f"\n📋 최종 권장 인코딩:")
        for file_path, encoding in results.items():
            if encoding:
                print(f"{Path(file_path).name:20} → {encoding}")
        
        return results

# 사용 예시
if __name__ == "__main__":
    detector = EncodingDetector()
    
    # 1. 개별 파일 점검
    files_to_check = ["./data/gwashi.csv", "./data/munjib.csv", "./data/sigwon.csv"]

    # 2. 전체 파일 일괄 점검
    encoding_results = detector.check_all_files(files_to_check)
    
    # 3. 필요시 인코딩 변환
    # detector.convert_file_encoding("gwashi.csv", "gwashi_utf8.csv", "euc-kr", "utf-8")

# 특별히 gwashi.csv 문제 해결
def fix_gwashi_encoding():
    """gwashi.csv의 특수한 인코딩 문제 해결"""
    detector = EncodingDetector()
    
    print("=== gwashi.csv 특별 점검 ===")
    
    # 다양한 방법으로 시도
    file_path = "./data/gwashi.csv"

    # 방법 1: 여러 인코딩 조합 시도
    encoding_combinations = [
        ('utf-8', 'cp1252'),  # UTF-8을 CP1252로 잘못 읽은 경우
        ('euc-kr', 'cp1252'), # EUC-KR을 CP1252로 잘못 읽은 경우
        ('cp949', 'cp1252'),  # CP949를 CP1252로 잘못 읽은 경우
    ]
    
    for original, wrong in encoding_combinations:
        try:
            # 잘못된 인코딩으로 읽기
            with open(file_path, 'r', encoding=wrong) as f:
                content = f.read(1000)  # 샘플만
            
            # 올바른 인코딩으로 재해석
            correct_content = content.encode(wrong).decode(original)
            
            print(f"🧪 {wrong}→{original} 테스트:")
            print(f"   원본: {content[:100]}...")
            print(f"   수정: {correct_content[:100]}...")
            
            # 한글 확인
            quality = detector.check_korean_characters(correct_content)
            print(f"   품질: {quality['quality']} (한글: {quality['korean_ratio']:.1%})")
            print()
            
        except Exception as e:
            print(f"❌ {wrong}→{original} 실패: {e}")

# gwashi.csv 문제 해결 실행
fix_gwashi_encoding()