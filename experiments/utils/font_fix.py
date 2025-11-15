# -*- coding: utf-8 -*-
"""
한글/한자 폰트 완벽 해결 (matplotlib + seaborn)
이 스크립트를 노트북 최상단에서 import하세요
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from collections import OrderedDict

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
            'Songti SC',           # 한자 지원 (macOS 기본 제공)
            'AppleMyungjo',        # 한글/한자 지원
            'LiSong Pro',          # 한자 지원
            'Hiragino Mincho ProN', # 한자 지원
            'Apple SD Gothic Neo', # 한글 지원
            'AppleGothic',         # 한글 지원
            'Hiragino Sans GB',    # 간체 중국어 지원
            'Hiragino Sans',       # 기본 지원
            'NanumMyeongjo',       # 한글 지원
            'NanumGothic',         # 한글 지원
            'Arial Unicode MS',    # 유니코드 지원
        ]
    elif system == 'Windows':
        font_candidates = [
            'SimSun',              # 한자 지원 (Windows 기본)
            'PMingLiU',            # 번체 중국어
            'MingLiU',             # 번체 중국어
            'MS Mincho',           # 일본어 한자
            'Batang',              # 한글/한자 지원
            'Malgun Gothic',       # 한글 지원
            'Gulim',               # 한글 지원
            'NanumMyeongjo',       # 한글 지원
            'NanumGothic',         # 한글 지원
            'Arial Unicode MS',    # 유니코드 지원
        ]
    else:  # Linux or others
        font_candidates = [
            'Noto Serif CJK KR',   # 한글/한자 지원
            'Noto Serif CJK SC',   # 간체 중국어
            'Noto Sans CJK KR',    # 한글/한자 지원
            'Noto Sans CJK SC',    # 간체 중국어
            'NanumMyeongjo',       # 한글 지원
            'NanumGothic',         # 한글 지원
            'WenQuanYi Micro Hei', # 한자 지원
            'AR PL UMing CN',      # 한자 지원
            'AR PL UKai CN',       # 한자 지원
        ]

    # 사용 가능한 폰트 찾기
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    resolved_fonts = []
    for font in font_candidates:
        if font in available_fonts and font not in resolved_fonts:
            resolved_fonts.append(font)

    if not resolved_fonts:
        print("[ERROR] 한글 폰트를 찾지 못했습니다!")
        print(f"사용 가능한 폰트 중 'Gothic' 포함: {[f for f in available_fonts if 'Gothic' in f or 'gothic' in f][:10]}")
        return None

    selected_font = resolved_fonts[0]

    # 강제 설정 (여러 방법 동시 적용) - 한자 지원 폰트 우선
    fallback_fonts = list(OrderedDict.fromkeys(resolved_fonts + [
        'Songti SC',             # 한자 지원
        'AppleMyungjo',          # 한글/한자 지원
        'LiSong Pro',            # 한자 지원
        'Hiragino Mincho ProN',  # 한자 지원
        'Apple SD Gothic Neo',   # 한글 지원
        'AppleGothic',           # 한글 지원
        'Hiragino Sans GB',      # 간체 중국어
        'Noto Serif CJK KR',     # 한글/한자 지원
        'Noto Serif CJK SC',     # 간체 중국어
        'Noto Sans CJK KR',      # 한글/한자 지원
        'Noto Sans CJK SC',      # 간체 중국어
        'NanumMyeongjo',         # 한글 지원
        'NanumGothic',           # 한글 지원
        'SimSun',                # 한자 지원
        'PMingLiU',              # 번체 중국어
        'MingLiU',               # 번체 중국어
        'Arial Unicode MS',      # 유니코드 지원
        'DejaVu Sans'            # 기본 폰트
    ]))

    plt.rcParams['font.family'] = fallback_fonts
    plt.rcParams['font.sans-serif'] = fallback_fonts + list(plt.rcParams.get('font.sans-serif', []))
    plt.rcParams['font.serif'] = fallback_fonts + list(plt.rcParams.get('font.serif', []))
    plt.rcParams['axes.unicode_minus'] = False

    # matplotlib 전역 설정
    matplotlib.rcParams['font.family'] = fallback_fonts
    matplotlib.rcParams['font.sans-serif'] = fallback_fonts + list(matplotlib.rcParams.get('font.sans-serif', []))
    matplotlib.rcParams['font.serif'] = fallback_fonts + list(matplotlib.rcParams.get('font.serif', []))
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'

    # 캐시 클리어 (중요!)
    try:
        fm._rebuild()
    except AttributeError:
        # Newer versions of matplotlib don't have _rebuild
        pass

    print(f" [FONT] 한글/한자 폰트 설정 완료: {selected_font}")
    return selected_font


def get_korean_font():
    """
    설정된 한글/한자 폰트 이름 반환
    그래프 생성 시 font_properties로 사용 가능
    """
    system = platform.system()
    font_candidates = []

    if system == 'Darwin':
        # 한자 지원 폰트를 우선으로 선택
        font_candidates = [
            'Songti SC',
            'AppleMyungjo',
            'LiSong Pro',
            'Hiragino Mincho ProN',
            'Apple SD Gothic Neo',
            'AppleGothic',
            'NanumGothic'
        ]
    elif system == 'Windows':
        font_candidates = [
            'SimSun',
            'Batang',
            'Malgun Gothic',
            'Gulim',
            'NanumGothic'
        ]
    else:
        font_candidates = [
            'Noto Serif CJK KR',
            'Noto Sans CJK KR',
            'NanumGothic'
        ]

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
