import html
import os
import re
from urllib.request import Request, urlopen
from typing import Any, Dict, List, Optional, Tuple


NAVER_SECTOR_ENABLED = str(os.getenv('ALLOW_NAVER_SECTOR', '0')).strip().lower() in {'1', 'true', 'yes', 'on'}
NAVER_SECTOR_TIMEOUT_SEC = float(os.getenv('NAVER_SECTOR_TIMEOUT_SEC', '1.5'))
NAVER_SECTOR_MAX_LOOKUPS = int(os.getenv('NAVER_SECTOR_MAX_LOOKUPS', '120'))
NAVER_SECTOR_CACHE: Dict[str, Optional[str]] = {}
NAVER_SECTOR_LOOKUPS = 0
NAVER_UPJONG_RE = re.compile(r'업종명\s*:\s*<a[^>]*>([^<]+)</a>')
FALLBACK_SECTOR = '기타'


def normalize_kiwoom_text(value: Any) -> str:
    text = str(value or '').strip()
    if not text:
        return ''

    candidates = [text]
    if all(ord(char) <= 255 for char in text):
        raw = bytes(ord(char) for char in text)
        for encoding in ('cp949', 'euc-kr', 'utf-8'):
            try:
                decoded = raw.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
            if decoded and decoded not in candidates:
                candidates.append(decoded)

    for source_encoding, target_encoding in (('cp949', 'utf-8'), ('utf-8', 'cp949')):
        try:
            decoded = text.encode(source_encoding).decode(target_encoding).strip()
        except UnicodeError:
            continue
        if decoded and decoded not in candidates:
            candidates.append(decoded)

    for candidate in candidates:
        if re.search(r'[가-힣]', candidate) and '�' not in candidate:
            return candidate

    return max(candidates, key=lambda item: len(re.findall(r'[가-힣]', item)) * 10 - len(re.findall(r'[À-ÿ�]', item)) * 4)

SECTOR_KEYWORD_RULES: List[Tuple[str, List[str]]] = [
    ('반도체', ['반도체', 'HBM', 'DRAM', 'NAND', '낸드', '메모리', '비메모리', '파운드리', '웨이퍼']),
    ('AI·로봇', ['AI', '인공지능', '로봇', '자동화', '머신비전']),
    ('2차전지', ['2차전지', '이차전지', '배터리', '전고체', '양극재', '음극재', '전해액', '리튬']),
    ('바이오·제약', ['바이오', '제약', '신약', '항암', '의료기기', '헬스케어', '백신']),
    ('자동차', ['자동차', '전기차', '자율주행', '자동차부품', '전장']),
    ('전기전자', ['전기전자', '전자', '디스플레이', 'OLED', '스마트폰', '가전', 'IT부품']),
    ('인터넷·게임', ['인터넷', '게임', '플랫폼', '콘텐츠', '엔터테인먼트', '미디어']),
    ('금융', ['은행', '증권', '보험', '금융', '카드', '지주']),
    ('조선·해운', ['조선', '선박', '해운', '운송']),
    ('방산·항공우주', ['방산', '방위산업', '우주', '항공', '드론', '위성']),
    ('화학·소재', ['화학', '소재', '정유', '석유화학', '첨단소재']),
    ('철강·금속', ['철강', '금속', '비철금속', '구리', '알루미늄']),
    ('에너지·전력', ['에너지', '원전', '태양광', '풍력', '수소', '전력', '전선']),
    ('건설·기계', ['건설', '건자재', '시멘트', '기계', '플랜트']),
    ('음식료·소비재', ['음식료', '식품', '화장품', '의류', '소비재', '유통']),
    ('통신·보안', ['통신', '5G', '네트워크', '보안', '클라우드']),
]

NAME_HINTS: List[Tuple[str, str]] = [
    ('삼성전자', '반도체'), ('SK하이닉스', '반도체'), ('한미반도체', '반도체'), ('이오테크닉스', '반도체'),
    ('현대차', '자동차'), ('기아', '자동차'), ('LG에너지솔루션', '2차전지'), ('삼성SDI', '2차전지'),
    ('카카오', '인터넷·게임'), ('셀트리온', '바이오·제약'), ('KB금융', '금융'), ('신한지주', '금융'),
    ('HD현대중공업', '조선·해운'), ('한화오션', '조선·해운'), ('한화에어로스페이스', '방산·항공우주'),
    ('POSCO', '철강·금속'), ('포스코', '철강·금속'), ('LG화학', '화학·소재'), ('한국전력', '에너지·전력'),
    ('두산에너빌리티', '에너지·전력'), ('삼성물산', '건설·기계'), ('농심', '음식료·소비재'), ('SK텔레콤', '통신·보안'),
]


def naver_sector_stats() -> Dict[str, Any]:
    return {
        'enabled': NAVER_SECTOR_ENABLED,
        'cacheCount': len(NAVER_SECTOR_CACHE),
        'lookupCount': NAVER_SECTOR_LOOKUPS,
        'lookupLimit': NAVER_SECTOR_MAX_LOOKUPS,
        'timeoutSec': NAVER_SECTOR_TIMEOUT_SEC,
    }


def fetch_naver_sector(code: Optional[str]) -> Optional[str]:
    global NAVER_SECTOR_LOOKUPS
    clean = re.sub(r'[^0-9]', '', str(code or '')).zfill(6)[-6:]
    if not NAVER_SECTOR_ENABLED or clean == '000000':
        return None
    if clean in NAVER_SECTOR_CACHE:
        return NAVER_SECTOR_CACHE[clean]
    if NAVER_SECTOR_LOOKUPS >= NAVER_SECTOR_MAX_LOOKUPS:
        NAVER_SECTOR_CACHE[clean] = None
        return None

    NAVER_SECTOR_LOOKUPS += 1
    url = f'https://finance.naver.com/item/main.naver?code={clean}'
    try:
        request = Request(url, headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.naver.com/'})
        raw = urlopen(request, timeout=NAVER_SECTOR_TIMEOUT_SEC).read()
        page = raw.decode('utf-8', errors='replace')
    except Exception:
        NAVER_SECTOR_CACHE[clean] = None
        return None

    match = NAVER_UPJONG_RE.search(page)
    sector = normalize_kiwoom_text(html.unescape(match.group(1))) if match else ''
    NAVER_SECTOR_CACHE[clean] = sector or None
    return NAVER_SECTOR_CACHE[clean]


def parse_master_info(raw: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for token in str(raw or '').split(';'):
        token = token.strip()
        if not token:
            continue
        for sep in ['|', ':', '=']:
            if sep in token:
                key, value = token.split(sep, 1)
                result[normalize_kiwoom_text(key)] = normalize_kiwoom_text(value)
                break
    return result


def parse_theme_groups(raw_groups: str) -> List[Tuple[str, str]]:
    groups: List[Tuple[str, str]] = []
    for token in re.split(r'[;\n\r]+', str(raw_groups or '')):
        item = token.strip()
        if not item:
            continue
        if '|' in item:
            theme_id, theme_name = item.split('|', 1)
        elif '\t' in item:
            theme_id, theme_name = item.split('\t', 1)
        else:
            continue
        theme_id = theme_id.strip()
        theme_name = normalize_kiwoom_text(theme_name)
        if theme_id and theme_name:
            groups.append((theme_id, theme_name))
    return groups


def parse_code_list(raw_codes: str, clean_code) -> List[str]:
    codes: List[str] = []
    for token in re.split(r'[;|,\s]+', str(raw_codes or '')):
        code = clean_code(token)
        if code and code != '000000' and code not in codes:
            codes.append(code)
    return codes


def compact_text(*values: Any) -> str:
    return ' '.join(normalize_kiwoom_text(value) for value in values if str(value or '').strip())


def sector_from_keywords(text: str) -> Optional[str]:
    upper_text = str(text or '').upper()
    for sector, keywords in SECTOR_KEYWORD_RULES:
        if any(keyword.upper() in upper_text for keyword in keywords):
            return sector
    return None


def sector_from_name_hint(name: str) -> Optional[str]:
    upper_name = str(name or '').upper()
    for hint, mapped_sector in NAME_HINTS:
        if hint.upper() in upper_name:
            return mapped_sector
    return None


def pick_sector(raw_info: str, name: str, themes: Optional[List[str]] = None, code: Optional[str] = None) -> Dict[str, Any]:
    themes = themes or []
    naver_sector = fetch_naver_sector(code)
    if naver_sector:
        return {'sector': naver_sector, 'sectorSource': 'naver-upjong', 'themes': themes}

    for sector, source in [
        (sector_from_name_hint(name), 'kiwoom-name-hint'),
        (sector_from_keywords(compact_text(*themes)), 'kiwoom-theme'),
        (sector_from_keywords(compact_text(*parse_master_info(raw_info).values(), raw_info)), 'kiwoom-master-info'),
        (sector_from_keywords(name), 'kiwoom-name-keyword'),
    ]:
        if sector:
            return {'sector': sector, 'sectorSource': source, 'themes': themes}
    return {'sector': FALLBACK_SECTOR, 'sectorSource': 'unclassified-fallback', 'themes': themes}
