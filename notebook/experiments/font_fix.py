# -*- coding: utf-8 -*-
"""
한글/한자 폰트 설정 유틸리티 (강력한 버전)
matplotlib 폰트 캐시를 클리어하고 강제로 한글 폰트를 설정합니다.
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import os

def setup_korean_fonts_strong():
    """
    강력한 한글/한자 폰트 설정
    - 폰트 캐시 재빌드
    - 직접 폰트 파일 경로 지정
    - 여러 fallback 옵션
    """
    system = platform.system()

    # 1. 폰트 캐시 클리어
    try:
        import shutil
        cache_dir = fm.get_cachedir()
        if os.path.exists(cache_dir):
            print(f"[INFO] matplotlib 폰트 캐시 클리어: {cache_dir}")
            for f in os.listdir(cache_dir):
                if f.startswith('font'):
                    try:
                        os.remove(os.path.join(cache_dir, f))
                    except:
                        pass
    except Exception as e:
        print(f"[WARNING] 캐시 클리어 실패: {e}")

    # 2. 폰트 매니저 재빌드
    fm._load_fontmanager(try_read_cache=False)

    # 3. 시스템별 폰트 파일 직접 로드
    font_paths = []
    if system == 'Darwin':  # macOS
        font_paths = [
            '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
            '/System/Library/Fonts/AppleSDGothicNeo.ttc',
            '/Library/Fonts/AppleGothic.ttf',
            '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        ]
    elif system == 'Windows':
        font_paths = [
            'C:/Windows/Fonts/malgun.ttf',
            'C:/Windows/Fonts/gulim.ttc',
            'C:/Windows/Fonts/batang.ttc',
        ]
    else:  # Linux
        font_paths = [
            '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        ]

    # 4. 폰트 파일 직접 등록
    found_font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                font_prop = fm.FontProperties(fname=path)
                found_font = font_prop.get_name()
                print(f"[FONT] 폰트 파일 등록 성공: {path} -> {found_font}")
                break
            except Exception as e:
                print(f"[WARNING] 폰트 등록 실패 {path}: {e}")
                continue

    # 5. 이름으로 폰트 찾기 (fallback)
    if not found_font:
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        font_candidates = []
        if system == 'Darwin':
            font_candidates = ['AppleGothic', 'Apple SD Gothic Neo', 'Arial Unicode MS', 'Nanum Gothic']
        elif system == 'Windows':
            font_candidates = ['Malgun Gothic', 'Gulim', 'Batang', 'NanumGothic']
        else:
            font_candidates = ['NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR', 'DejaVu Sans']

        for font in font_candidates:
            if font in available_fonts:
                found_font = font
                print(f"[FONT] 이름으로 폰트 찾음: {font}")
                break

    # 6. rcParams 설정
    if found_font:
        plt.rcParams['font.family'] = found_font
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 한글/한자 폰트 설정 완료: {found_font}")
        return found_font
    else:
        print("❌ 한글/한자 폰트를 찾지 못했습니다!")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        return None

if __name__ == "__main__":
    # 테스트
    setup_korean_fonts_strong()

    # 간단한 테스트 그래프
    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title('한글 제목 테스트 - 年中時宗場試大科傳復所進事生元增光措文祭')
    ax.set_xlabel('년 (年)')
    ax.set_ylabel('값')
    plt.savefig('/tmp/font_test.png', dpi=150, bbox_inches='tight')
    print("✅ 테스트 그래프 저장: /tmp/font_test.png")
    plt.close()
