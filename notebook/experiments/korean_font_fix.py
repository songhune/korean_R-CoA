# -*- coding: utf-8 -*-
"""
한글 폰트 완벽 해결 스크립트
matplotlib + seaborn에서 한글이 깨지지 않도록 강제 설정

사용법:
    from korean_font_fix import setup_korean_font
    setup_korean_font()  # import 직후 바로 호출

    # seaborn 사용 시:
    import seaborn as sns
    sns.set()
    setup_korean_font()  # sns 설정 후 다시 호출!
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform


def setup_korean_font():
    """
    한글/한자 폰트 강제 설정
    seaborn을 사용하는 경우 sns.set() 호출 후에 다시 호출해야 함!
    """
    system = platform.system()

    # 시스템별 한글 폰트 우선순위
    if system == 'Darwin':  # macOS
        font_candidates = [
            'AppleGothic',
            'Apple SD Gothic Neo',
            'AppleMyungjo',
            'NanumGothic',
            'Nanum Gothic',
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

    # 사용 가능한 폰트 검색
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    selected_font = None
    for font in font_candidates:
        if font in available_fonts:
            selected_font = font
            break

    if not selected_font:
        print("[ERROR] 한글 폰트를 찾을 수 없습니다!")
        print(f"Gothic 포함 폰트: {sorted([f for f in available_fonts if 'Gothic' in f or 'gothic' in f])[:5]}")
        return None

    # 강제 설정 (3가지 방법 동시 적용)
    # 1. matplotlib 전역 설정
    matplotlib.rcParams['font.family'] = selected_font
    matplotlib.rcParams['font.sans-serif'] = [selected_font]
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 2. pyplot 설정
    plt.rcParams['font.family'] = selected_font
    plt.rcParams['font.sans-serif'] = [selected_font]
    plt.rcParams['axes.unicode_minus'] = False

    # 3. rc 직접 설정
    plt.rc('font', family=selected_font)

    print(f"✅ 한글 폰트 설정 완료: {selected_font}")
    return selected_font


def get_korean_font_prop():
    """
    한글 폰트의 FontProperties 객체 반환
    특정 텍스트에만 적용하고 싶을 때 사용

    사용 예:
        font_prop = get_korean_font_prop()
        plt.title("한글 제목", fontproperties=font_prop)
    """
    system = platform.system()

    if system == 'Darwin':
        font_candidates = ['AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic']
    elif system == 'Windows':
        font_candidates = ['Malgun Gothic', 'Gulim', 'NanumGothic']
    else:
        font_candidates = ['NanumGothic', 'Noto Sans CJK KR']

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font in font_candidates:
        if font in available_fonts:
            return fm.FontProperties(family=font)

    return None


# 자동 실행 (이 파일을 import하면 바로 적용)
if __name__ != "__main__":
    setup_korean_font()
