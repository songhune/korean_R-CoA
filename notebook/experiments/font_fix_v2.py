# -*- coding: utf-8 -*-
"""
한글/한자 폰트 완벽 해결 (matplotlib + seaborn)
이 스크립트를 노트북 최상단에서 import하세요
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

def setup_korean_fonts_robust():
    """
    macOS/Windows/Linux에서 한글/한자를 지원하는 폰트 강제 설정
    matplotlib 백엔드 설정 전에 호출해야 함
    """
    system = platform.system()

    # 시스템별 한글/한자 폰트 우선순위
    font_candidates = []
    if system == 'Darwin':  # macOS
        font_candidates = [
            'AppleGothic',
            'Apple SD Gothic Neo',
            'AppleMyungjo',
            'Apple SD 산돌고딕 Neo',
            'Nanum Gothic',
            'NanumGothic',
        ]
    elif system == 'Windows':
        font_candidates = [
            'Malgun Gothic',
            'Gulim',
            'Batang',
            'NanumGothic',
        ]
    else:  # Linux
        font_candidates = [
            'NanumGothic',
            'Noto Sans CJK KR',
            'Noto Sans KR',
        ]

    # 사용 가능한 폰트 찾기
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break

    if selected_font is None:
        print("[ERROR] 한글 폰트를 찾지 못했습니다!")
        print(f"사용 가능한 폰트 중 'Gothic' 포함: {[f for f in available_fonts if 'Gothic' in f or 'gothic' in f][:10]}")
        return None

    # 강제 설정 (여러 방법 동시 적용)
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['font.sans-serif'] = [selected_font] + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    # matplotlib 전역 설정
    matplotlib.rcParams['font.family'] = selected_font
    matplotlib.rcParams['font.sans-serif'] = [selected_font] + matplotlib.rcParams['font.sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 캐시 클리어 (중요!)
    fm._rebuild()

    print(f"✅ [FONT] 한글/한자 폰트 설정 완료: {selected_font}")
    return selected_font


def get_korean_font():
    """
    설정된 한글 폰트 이름 반환
    그래프 생성 시 font_properties로 사용 가능
    """
    system = platform.system()
    font_candidates = []

    if system == 'Darwin':
        font_candidates = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic']
    elif system == 'Windows':
        font_candidates = ['Malgun Gothic', 'Gulim', 'NanumGothic']
    else:
        font_candidates = ['NanumGothic', 'Noto Sans CJK KR']

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font in font_candidates:
        if font in available_fonts:
            return font

    return None


# 사용법:
# 1. 노트북 최상단에 추가:
#    from font_fix_v2 import setup_korean_fonts_robust
#    setup_korean_fonts_robust()
#
# 2. seaborn 사용 시 반드시 폰트 설정 AFTER seaborn import:
#    import seaborn as sns
#    sns.set()  # 이게 폰트를 리셋할 수 있음!
#    setup_korean_fonts_robust()  # 다시 설정
#
# 3. 그래프 저장 전에 한번 더 확인:
#    plt.rcParams['font.family'] = get_korean_font()
#    plt.savefig(...)
