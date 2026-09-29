"""
risk_routing.py
누비자 세이프루트 - 위험도 기반 경로 추천 백엔드 모듈

Phase 2에서 검증한 로직(반경 기반 부분그래프 추출, 위험도 계산,
3가지 경로 계산, folium 지도 생성)을 재사용 가능한 함수로 정리했습니다.
app.py(Streamlit)에서 이 모듈을 import해서 사용합니다.

핵심 설계 원칙 (2차 개정):
    위험도(risk_score)는 매 요청마다 잘라내는 작은 OD 부분그래프 안에서
    다시 정규화하지 않고, 창원시 전체 그래프(G_full) 위에서 딱 한 번만
    계산합니다. "상위 5%" 같은 기준이 요청마다 달라지는 국지적 분포가 아니라
    도시 전체를 기준으로 한 절대적인 값이 되도록 하기 위함입니다.
    -> prepare_g_full()을 앱 시작 시 한 번만 호출하고(st.cache_resource),
       이후 extract_od_subgraph()로 잘라낸 부분그래프는 이미 계산된
       risk_score / cost_* 속성을 그대로 물려받습니다.

전제 조건:
    changwon_G_full_with_risk.pkl 체크포인트가 이 파일과 같은 디렉토리에 있어야 합니다.
    (STEP 1 ~ STEP 2-B에서 만든, OSM 태그 + 자전거도로 라벨 + 사고 위험 피처가
     모두 붙어있는 창원시 전체 도로 그래프. 단, 이 체크포인트에는 아직
     accident_component 등 정규화 이전의 원시 피처만 있고, risk_score는
     이 모듈의 compute_risk_scores()가 최초로 계산합니다.)
"""

import os
import re
import difflib
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely import offset_curve
import folium

# 지도 위 "피할 수 없는 위험 구간" 마커 아이콘.
# 사용자가 업로드한 바리케이드(🚧) 이미지를 배경 투명 처리해 내장한 base64 PNG로,
# 배포 환경(Streamlit Cloud 등)에서 별도 이미지 파일 없이도 항상 동일하게 렌더링됩니다.
# app.py의 "🚧 피할 수 없는 위험 구간" 패널과 같은 아이콘을 지도에서도 써서
# 사용자가 둘을 같은 의미로 바로 인식할 수 있게 합니다.
RISK_MARKER_ICON_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAG8AAABpCAYAAAA0nH19AAA1z0lEQVR4nO29SYwkSZam9z0RVTNzM9/3JXyLPcNzqe6anm5g"
    "hqxqstnsAVEghmDUgTwUeeg7D3OeTBLkiXPggQQPfSgQnAMRSRAgqkGwZ5oz1RgQ3dVd1ZNbRGbG6muE77vboqoijwdRNbNw"
    "96jcKpu1pAAeHm6Lqqj87/3yvydPRYVfcFPViLW1mDg2pKlndjYVkewXfZ5f5qaqBoh4/jzCTAvp2m/kOHzTvu6mqOjbb5t7"
    "d+/Zy96/d/eu1bv3rN67Z/WeWlUVVZW/635+HS2/FhOu7Z5V3jav+uy9u/esvv22UX4x1/7KE32h9vY78rMX0/ZbJy+iy97+"
    "1r/9hxGTvRG934r41qMIsMCvBXiA8DMsvd8K13d36dIxAPjW5IvoZy+mLW+/8+ty7d+0L9u+tAXkk7IFvIi4rteHUng9hmHg"
    "APhIRPa/eld/uZqqiojoZe9FQ1XS/bNrwDWgJ4WdGO6LyFH7+/fUchcDOBHxX6YPr3Txz9Es29tl6vUMwZFfRh2uRs3mP8bY"
    "m5TiR+Elfq3Ay+drAS4FL90/GwB+B/gjYFTIPkyIToEP2h/61qMI+iKYbAF/5+B56vVMFheb3S/qJ5tvOeHfaxp5S72+75UP"
    "gZ++9JlnzyosLKTdHvvr0lS1TD29lR3Xf8cl7g9QnVH1o0bkPl3gyc2bLX32TFi43AA+T3sVePL222/LgwcPBODdO3f07oMH"
    "cmf7jiyNY97lgcsH3gH8HvT899//n2/29ZXn/F9t/IOsz94cGBzmxbPlq40n27/7s9/7b3cW/9FvM/SPfov6RARTtUOoP1O9"
    "++Kls96/Y1nCw4NwQe/Cu/lb777LV25383/u3v0y374j9+8vmaWldx0/fkd+1jctd+/elf/phz8cNyduTrdOxvu2HC/+uz8v"
    "nRwf3eyZGnirMjEwMzY9zeHB3u24wT/88L/634/SLFv9P//Hf/7ov+ZP64Xh37171759545d+i6e7+bXTtHJ+wrv6GUUfSl4"
    "OXBRtVo1AH/0k59otTEh9L0wL0pT0fZ6NSEHDuA/43uTEe4PJMm+k3zw/JY3vmen9Zi9tY1y8vzgd0WYktRBkiKJx6el90ws"
    "/4Ll6u5LJ56YjXm+5kiqfhlgDM6WYXl5mWoVlpe/zKB32v0F+O7ZAsvLsPBFv1yaNksTx5blarq8sMzT/+OBVKvbmh6dXS2n"
    "+gfaSr6tzRTdP7NHf/tk+Nj5qfJYP256Al+JBmrlnn9H0ua0Uf7ijTdeP+HDP31aHLpa3Y5PuVN6fvIim16uehb6Fe4LLHD/"
    "Pn5p6Z2MSyj6CwkWvXvPLlfP4sX/5b98iSof/hd/8j1Nsz/2jfR7Y2mFs51Dtp9vku6enAFUro7VFv7932boD97idMJSmhr4"
    "kRmu/Ulcq/zopeP/62cVvvvLSad6T+3y2HK8+PsvTxPN9f3v+ePGHyebR98b2Hbs/fg+Kz9+j+aznTNEKI331ybmZ+ibHmWn"
    "kmBqpR9JKfqTa//Df/7StT/7wQ8rC/VaKu9+/3Nf+xea8/IDOwBVrQA3aTHP//az75z+m/de79nPcOvbnK1tejncTaCJoVKb"
    "OJpk6KQM9Rq9zYizlWYfe8PJheOfG5hfpibf70wT3a0cDyUcHPT3SA3EMeLLaMPzvHVYVtIzu3rWjFOtDKbKwJV+Dnrc68Nv"
    "3fgHqroPbAArIqLnHeLztK8iWKaA/4DjxneYHpjrLTGZba5TX92Cs6b04GKLWkeL9GgfNl7AwxXcQUQztk2vQ42vcO5fnvbk"
    "oMmzrSaNDPY97G0RZSf04gFbkoqPzeke9eUGkRuhPDgzyWDp94EY+CvgmC+pxj8TPFW1LC/H51Ul8Lo/bXwn3dn+Xjmrc+Ya"
    "rO6ukJ7sNk1vJbYDvbaUlfCNpFnv26scuGX6dww7icUM1FLTsz5w/lzP3v5hZWGplsr3Pz91/F21V00Zm//qT/ulkSbupMH4"
    "qXK8s8aWOcAPa2oqpQgrtn5yxOb+lsbxofTf7u1JW0dv9r7YrURTowDPgf+3+5gPHz4s33jvveyzxkHu3b1nv3+OZ9/+J29P"
    "poa5VtKcKFerJjk9iVbXN5JWq+VStVKJxfzj//Q//s616en/ZL5naLHxdJfn/9e/YfUv/qbpTg7d8HRfeXR2MKpV1fdI1qyU"
    "jRmcHqdnaoKDikGG7M96rrT+vPT69s/IAhfZCCJvY3rEYTNPy3Gw63j8uM779+v87f0Wz1YTTupgraVatUQ2Ist77vJ/rIXI"
    "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"
    "LjOWz/T9X9J2Sb8bNMppklnvPN75IGIIBVmRCXlAdVnTZ8mhCQ/COFV3cRPbz9s+E7x79+7Zhw8fli95q0eb9VTS+nH9eM+n"
    "zVOMzzDqrfjMqkvIWme06sdIVk9oHho9Ous9fxD1WlZ91dZov5xNVe2//lfPKudff/LRSv/p4Wmp1WjRrDdJWwk+yxD1GBzi"
    "EjQ5y1xjv5nVWy1AkpOD6vnj3Lt3z36eZ+peSpuqat59593ojDOzstK0B0/fi37wnR8IwMJC+EkbBxaHxyepS1uqLi1SWRLS"
    "fR51GT5LwGYel4lvnMbPfviDCsuwTHgq15+/+275vZ+sZD/4zg9yGlr4HMO3fOGVhQX4bt63r9KWl/MfXv3UsH/2T/6ZjXpH"
    "4x985wcvvX56dFIqV8rGZx6XObwPwXvICWgeUmSetOlonbhQJv40jEkxKAsLHA38xLKG12c/1PtnNb+0dDe7bP571Zyn97mf"
    "LS0tSe0+vvLtq1r/WT0FWKoDS0BPxft6hphIxEaIsUF6A4gJqSNjEWPJH5ygJiq5hVo9ZQn+Bqg9gDtXr5oDyP569a9z8B58"
    "7oHubkt12FmC736pb3fa3yyFfi0BD8Yv/8z87877ZnOA+oP6S0q9XK1lglNjDcbaoJrz+ESlyBQYIYqL9R+Fq9lCrZ7CeDgp"
    "de4f9XpmxzygS9ztZOPPtUvByx+0pwD37t3j9u1+effddx3kDyN8F1TvaWLqRjGRMVaMKdYAw7MfEGkD6HEm7GgTe/n+uy9N"
    "9Hrvnrv77W+773/F5we92/7n62/37t3jxz9eNsWYFO2f/tP/xqtXdd7nWzuarjW+4H9ijCDWEpWLm7H0/Jjcu3ePJb7rX/VM"
    "2qJ9acGSkhZ5C5GXUhhw3lB+XZ6J/kVbsVD8UlMpqkDMqz7yeduXBi8mVkFV1ecrFopK+Ml7STul9RkW9OvTziv8ToagUxuj"
    "xR1b3QP1pdpXCRVCX4I8Oe9rX/Wwv+5N6bbuL9m+MnjftP//2tcAXnf6/jdyqmu3btfqdrFf1Kh8TZ73mw3a31X7BryvtX3l"
    "ae3ntq8EXvc6qSj50spX79Svbou4YLjtm+F/8QPzjWD5WtvXa8m/MPC+XoL41Wxf93h8JfBeIoicHopbnX9jkSxqZOj81q9D"
    "avIL97yiOvQ3FbmXWzHdfV2j8Zkr6V+8vQzjb2zL2efrZKJfoGApaLMjrrR72yG+Fkv5pWpR1LXlFtp1/V+PMX/jeV9Xyye7"
    "XwnP60BW1O5/09pj8TUNxlcEL3xdtZsy8yUi/c0GMQBXOFxnHKR4PvsvoH09avM3FbFz7ev2vK8y5yngJTwd1Kmqyzvp8l5L"
    "171r+VNnvq7L+OVoqqqK+jAWOXC+w0IU4yPFNqhfrX1l8IDMiKQUe86ouPxmGQXyjXc0IxSq/lqDB6gxJhNxKYQb3MMUEtbO"
    "wy4okqnDW2u/MoBfBTwDRKqUFa0IEuoYgweqhAd4ACBIGSHubMj769lExChaAqnQJd7yTUby5IqWxUjknLNdQ/Sl2ucCb2Nj"
    "47KTxCJSAx2qVmuVgf6E5nFKIqmt9fQy0DfI0OAww8M1do72RgzS57y7cJMgv6LrRzs7lzx43toSzvUPDQ3RWxmgdZLSOEoQ"
    "Z+nr7cOLIyMZQrXXe1+ms533l2qfC7yFyytZDRCDlMvlMqVymVJcQmOI4xJxqUS5XMZUoNIs49WVInvJ3oO/ou3KlYuvee+t"
    "qi+Xy2XK5XIYhyhGjKUUl/DiydKkbMVE1habYX359rnAS9P0Mm5uqeohsHlwsN+/s7tT2TvYI21krpUkakpElb6Iph/Am3Sv"
    "XI33VPWym0p+JefBVstd6LcxpuHF72xublIrn7G1vcPO3i7iLc1mAzWe3v7apsKhqjZo7wbz5dpngnf//n29evXqZbcaHSh8"
    "DFpW5akWDwoTPOGeuvDIn1DKcoLIe6mmu5cc5xeivP6Om+7snF0YE2PtliP7qRZze5EiJK+uBUR12ah5IGI2gTMuudvo/v37"
    "evfu3c/sxGeC98477+hlJwA21Zi/EtEnBnrFSP58dpvv4GTyYkWDwSTidduWovVLjvOrqEL17t2lC7dhpcasitg/Q9P3xWso"
    "cdewDQlqwnYkyLGNzI738R5wwsViT9555x39zal1/aZ9075p37Rv2jfts9uXjjPyu1nt2hp2f/99s7W1xdYW4QHeExNMTEzw"
    "5sQETEzC1iaTb01mQPartHHAF23F5gtHR0d2ggnCmBS34YdnmvddH/b9/bNuZwf/3e/ijTFev2TK9/8DsSUT85TtSikAAAAA"
    "SUVORK5CYII="
)
RISK_MARKER_ICON_DATA_URI = "data:image/png;base64," + RISK_MARKER_ICON_B64


# --------------------------------------------------------------------------
# 0. 설정
# --------------------------------------------------------------------------

# 체크포인트 경로는 "현재 작업 디렉토리"가 아니라 "이 파일(risk_routing.py) 자신의 위치"
# 기준 절대경로로 계산합니다. Streamlit Cloud처럼 저장소 하위 폴더(예: saferoute_app/)를
# Main file path로 지정해서 실행하는 환경에서는, 실행 시 작업 디렉토리가 그 하위 폴더가
# 아니라 저장소 루트일 수 있어서 상대경로("changwon_G_full_with_risk.pkl")가 엉뚱한
# 위치(루트)를 가리키게 되는 문제가 있었습니다. __file__ 기준으로 고정하면 이 문제를
# 완전히 피할 수 있습니다 (로컬 실행이든 Streamlit Cloud든 항상 이 .py 파일과 같은
# 디렉토리에서 pkl을 찾습니다).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_G_FULL = os.path.join(_MODULE_DIR, "changwon_G_full_with_risk.pkl")

# 자주 찾는 장소는 좌표를 미리 등록해 둡니다 (좌표: OpenStreetMap Nominatim 검색 결과, 2026.9. 확인).
#  - 등록 장소는 외부 검색 서버를 거치지 않으므로 즉시 응답하고, 서버 장애·속도 제한과 무관하게 동작합니다.
#  - 오타 입력 시 "혹시 이 장소인가요?" 추천 후보로도 사용합니다.
PLACES = {
    # 성산구·의창구 (시범 구간)
    "창원시청": (35.2275036, 128.682374),
    "창원대학교": (35.2448472, 128.6951022),
    "창원중앙역": (35.242367, 128.701193),
    "창원역": (35.257585, 128.607013),
    "경상남도청": (35.237865, 128.691872),
    "창원광장": (35.226347, 128.682189),
    "용지공원(용지호수)": (35.232475, 128.680904),
    "상남시장": (35.222238, 128.683368),
    "가음정공원": (35.212232, 128.686997),
    "가음정시장": (35.207534, 128.697875),
    "반송시장": (35.236120, 128.671879),
    "창원컨벤션센터": (35.238588, 128.656693),
    "창원종합운동장": (35.235431, 128.665418),
    "창원체육관": (35.232699, 128.666269),
    "성산아트홀": (35.229582, 128.682088),
    "롯데백화점 창원점": (35.224607, 128.681842),
    "정우상가": (35.228128, 128.679455),
    "올림픽공원": (35.225321, 128.659414),
    "창원문성대학교": (35.234695, 128.660866),
    "창원과학체험관": (35.230226, 128.661921),
    "창원기계공업고등학교": (35.224961, 128.665757),
    "파티마병원": (35.237019, 128.646896),
    "창원종합버스터미널": (35.236299, 128.639438),
    "의창구청": (35.255637, 128.634964),
    "성산구청": (35.198503, 128.702513),
    # 마산·진해 (시범 구간 밖이지만 경로 계산 가능)
    "마산역": (35.236367, 128.580728),
    "마산고속버스터미널": (35.223010, 128.588128),
    "삼성창원병원": (35.242636, 128.591048),
    "경남대학교": (35.181582, 128.553198),
    "진해역": (35.153477, 128.660151),
}
# 같은 장소를 부르는 다른 이름
PLACE_ALIASES = {
    "시청": "창원시청", "창원대": "창원대학교", "도청": "경상남도청", "경남도청": "경상남도청",
    "용지호수": "용지공원(용지호수)", "용지호수공원": "용지공원(용지호수)", "용지공원": "용지공원(용지호수)",
    "창원파티마병원": "파티마병원", "창원컨벤션": "창원컨벤션센터", "세코": "창원컨벤션센터", "CECO": "창원컨벤션센터",
    "롯데백화점": "롯데백화점 창원점", "창원버스터미널": "창원종합버스터미널", "창원터미널": "창원종합버스터미널",
    "마산터미널": "마산고속버스터미널", "경남대": "경남대학교", "문성대": "창원문성대학교",
}
# 이전 버전 호환 (app.py 등에서 DEMO_LOCATIONS를 참조하던 코드용)
DEMO_LOCATIONS = PLACES

# 창원시 도로망 범위 (지오코딩 결과를 이 범위 안으로 제한)
CHANGWON_VIEWBOX = (128.35, 35.05, 128.85, 35.40)  # (서, 남, 동, 북)
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_TIMEOUT_S = 6
NOMINATIM_HEADERS = {"User-Agent": "nubija-saferoute/1.0 (+https://github.com/fasteel01/changwon-saferoute)"}

# 도로 위계 -> 위험 기여도 (사고/자전거 인프라 태그가 없을 때의 대리 지표)
HIERARCHY_WEIGHT_MAP = {
    "motorway": 1.0, "motorway_link": 1.0,
    "trunk": 0.9, "trunk_link": 0.9,
    "primary": 0.8, "primary_link": 0.8,
    "secondary": 0.6, "secondary_link": 0.6,
    "tertiary": 0.4, "tertiary_link": 0.4,
    "unclassified": 0.3,
    "residential": 0.2,
    "living_street": 0.1,
    "service": 0.1,
    "cycleway": 0.0,
    "footway": 0.0,
    "path": 0.05,
}
DEFAULT_HIERARCHY_WEIGHT = 0.3  # 태그가 없거나 매핑에 없는 도로유형 기본값

# 자전거 보호시설 유형 -> 보호 보너스 (클수록 안전, risk_score에서 차감됨)
INFRA_PROTECTION_MAP = {
    "자전거전용도로": 1.0,        # 차도와 물리적으로 분리
    "자전거전용차로": 0.6,        # 차도 위 구획된 차로
    "자전거보행자겸용도로": 0.3,   # 보도와 공유
}
DEFAULT_INFRA_PROTECTION = 0.0  # 자전거 인프라 태그가 없는 경우 (보호 없음)

W_ACCIDENT = 0.5
W_HIERARCHY = 0.3
W_INFRA = 0.2

BIKE_PENALTY = 3.0    # 자전거 인프라 없는 구간에 곱해지는 거리 페널티 (cost_bike_priority)
K_RISK = 2.0          # risk_score가 비용에 반영되는 강도 (cost_risk_aware)

CITY_RISK_QUANTILE = 0.95  # "위험구간"으로 강조 표시할 창원시 전체 기준 상위 분위
CORRIDOR_BUFFER_M = 250    # (이전 버전) 경로 주변 위험구간 표시 반경 — 현재는 경로가 실제로 지나는 구간만 표시

# 예상 소요시간 계산용 평균 주행속도 가정치.
# 실측 주행속도 데이터(예: 누비자 실제 GPS 로그)가 없어서 쓰는 근사값이며,
# 도로 유형별 정차/서행/신호대기는 반영하지 않은 단순 환산입니다.
# (국내 공영자전거 평균 주행속도로 흔히 인용되는 15km/h를 기본값으로 사용)
AVG_BIKE_SPEED_KMH = 15.0


# --------------------------------------------------------------------------
# 1. 체크포인트 로드
# --------------------------------------------------------------------------

def load_g_full(path: str = PATH_G_FULL):
    """미리 계산해둔 창원시 전체 그래프(사고 근접 피처 포함)를 불러옵니다."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' 파일을 찾을 수 없습니다. "
            "STEP 1 ~ STEP 2-B 파이프라인을 먼저 돌려서 체크포인트를 생성하고, "
            "이 파일과 같은 디렉토리에 두었는지 확인해주세요."
        )
    with open(path, "rb") as f:
        G_full = pickle.load(f)
    return G_full


def save_checkpoint(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def prepare_g_full(path: str = PATH_G_FULL, risk_quantile: float = CITY_RISK_QUANTILE):
    """
    앱 시작 시 딱 한 번만 호출하는 함수입니다 (app.py에서 st.cache_resource로 감싸서 사용).

    G_full을 불러온 뒤 위험도(risk_score)를 창원시 전체 기준으로 계산하고,
    "위험구간"으로 강조 표시할 상위 분위 임계값(city_threshold)을 함께 반환합니다.
    이후 extract_od_subgraph()로 잘라내는 모든 부분그래프는 이미 계산된
    risk_score를 그대로 물려받으므로, 요청마다 다시 정규화하지 않습니다.
    """
    G_full = load_g_full(path)
    G_full, edges_full = compute_risk_scores(G_full)
    city_threshold = float(edges_full["risk_score"].quantile(risk_quantile))
    return G_full, city_threshold


# --------------------------------------------------------------------------
# 2. 지오코딩
# --------------------------------------------------------------------------

class GeocodeError(Exception):
    """사용자에게 그대로 보여줄 수 있는 위치 검색 오류. suggestions: 비슷한 등록 장소 이름."""

    def __init__(self, message: str, suggestions=None):
        super().__init__(message)
        self.message = message
        self.suggestions = list(suggestions or [])


def _norm_name(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


_PLACE_LOOKUP = {_norm_name(n): n for n in PLACES}
_PLACE_LOOKUP.update({_norm_name(a): n for a, n in PLACE_ALIASES.items()})

# "중앙대로 151", "창이대로263번길 12"처럼 도로명 + 건물번호 형태의 입력
_ROAD_ADDRESS_RE = re.compile(r"(로|길)\s*\d+(-\d+)?\s*$")


def suggest_places(query: str, n: int = 3):
    """오타가 난 입력과 비슷한 등록 장소 이름을 찾습니다."""
    q = _norm_name(query).replace("창원시", "")
    if not q:
        return []
    keys = list(_PLACE_LOOKUP.keys())
    hits = difflib.get_close_matches(q, keys, n=n * 2, cutoff=0.5)
    # 부분 일치(예: "가음정" -> "가음정공원", "가음정시장")도 후보로
    hits += [k for k in keys if q in k or (len(k) >= 2 and k in q)]
    out = []
    for k in hits:
        name = _PLACE_LOOKUP[k]
        if name not in out:
            out.append(name)
    return out[:n]


KAKAO_KEYWORD_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"
KAKAO_ADDRESS_URL = "https://dapi.kakao.com/v2/local/search/address.json"
KAKAO_TIMEOUT_S = 5


class _KakaoUnavailable(Exception):
    """카카오 검색을 쓸 수 없는 상황(키 오류·서버 장애 등) → OSM 검색으로 넘어감."""


def _in_changwon(lat: float, lon: float) -> bool:
    west, south, east, north = CHANGWON_VIEWBOX
    return south <= lat <= north and west <= lon <= east


def _kakao_get(url: str, params: dict, key: str) -> list:
    import requests
    try:
        resp = requests.get(url, params=params, headers={"Authorization": f"KakaoAK {key}"},
                            timeout=KAKAO_TIMEOUT_S)
    except requests.RequestException as e:
        raise _KakaoUnavailable(f"network: {e}")
    if resp.status_code != 200:
        # 401: 키 오류 / 403: 카카오맵 사용 설정 꺼짐 / 429: 쿼터 초과 → 로그에 남기고 OSM으로 대체
        print(f"[kakao] HTTP {resp.status_code}: {resp.text[:200]}")
        raise _KakaoUnavailable(f"http {resp.status_code}")
    try:
        return resp.json().get("documents", []) or []
    except ValueError:
        raise _KakaoUnavailable("bad json")


def _search_kakao(q: str, key: str):
    """
    카카오 로컬 API로 검색합니다. 찾으면 dict, 창원시 안에서 못 찾으면 None.
      - 도로명/지번 주소처럼 보이면 주소 검색을 먼저, 아니면 키워드(장소명) 검색을 먼저 시도
      - 결과는 창원시 범위 안의 것만 사용
    """
    west, south, east, north = CHANGWON_VIEWBOX

    def keyword():
        docs = _kakao_get(KAKAO_KEYWORD_URL,
                          {"query": q, "rect": f"{west},{south},{east},{north}", "size": 5}, key)
        for d in docs:
            lat, lon = float(d["y"]), float(d["x"])
            if _in_changwon(lat, lon):
                addr = d.get("road_address_name") or d.get("address_name") or ""
                return {"lat": lat, "lon": lon, "label": f"{d.get('place_name', q)} · {addr}".strip(" ·"),
                        "source": "kakao", "precision": "place", "warning": None}
        return None

    def address():
        query = q if ("창원" in q or "경남" in q or "경상남도" in q) else f"창원시 {q}"
        docs = _kakao_get(KAKAO_ADDRESS_URL, {"query": query, "size": 5}, key)
        for d in docs:
            lat, lon = float(d["y"]), float(d["x"])
            if not _in_changwon(lat, lon):
                continue
            atype = d.get("address_type", "")
            warning, precision = None, "place"
            if atype == "REGION":
                precision = "area"
                warning = (f"'{d.get('address_name', q)}' 지역의 중심점으로 잡혔어요. "
                           "동네 이름보다는 건물·장소 이름이나 정확한 주소로 입력하면 더 정확해요.")
            elif atype == "ROAD":
                precision = "road"
                warning = (f"'{d.get('address_name', q)}' 도로 위의 한 지점으로 잡혔어요. "
                           "건물번호까지 입력하면 더 정확해요.")
            return {"lat": lat, "lon": lon, "label": d.get("address_name", q), "source": "kakao",
                    "precision": precision, "warning": warning}
        return None

    looks_like_address = bool(_ROAD_ADDRESS_RE.search(q)) or bool(re.search(r"\d+(-\d+)?\s*번지?$", q))
    order = (address, keyword) if looks_like_address else (keyword, address)
    for fn in order:
        found = fn()
        if found:
            return found
    return None


def search_place(query: str, kakao_key: str = None) -> dict:
    """
    장소 이름(또는 주소) -> 위치 정보 dict.
      {"lat", "lon", "label"(화면 표시용 이름),
       "source": "registered" | "kakao" | "osm",
       "precision": "place" | "road"(도로 위 임의 지점) | "area"(동네 중심점),
       "warning": str | None}

    1) 등록 장소(PLACES/PLACE_ALIASES)면 외부 서버 없이 즉시 반환
    2) 카카오 REST API 키가 있으면 카카오 로컬 검색 (장소명·도로명 주소 모두 건물 단위로 정확)
    3) 키가 없거나 카카오를 쓸 수 없으면 OpenStreetMap Nominatim 검색 (6초 제한, 재시도 없음)
       - 이전에는 ox.geocode()를 썼는데, 공용 서버가 요청을 제한하면 OSMnx가 재시도를
         반복하면서 화면이 무한 로딩되는 문제가 있어 직접 호출 + 짧은 제한시간으로 바꿨습니다.
    4) 어디서도 찾지 못하면 GeocodeError(비슷한 등록 장소 추천 포함)를 발생
    """
    q = str(query or "").strip()
    if not q:
        raise GeocodeError("장소를 입력해 주세요.")

    name = _PLACE_LOOKUP.get(_norm_name(q))
    if name:
        lat, lon = PLACES[name]
        return {"lat": lat, "lon": lon, "label": name, "source": "registered",
                "precision": "place", "warning": None}

    if kakao_key:
        try:
            found = _search_kakao(q, kakao_key)
        except _KakaoUnavailable:
            found = None  # 카카오 장애 시 OSM으로 대체
        else:
            if found is None:
                raise GeocodeError(
                    f"'{q}'을(를) 창원시 안에서 찾지 못했어요. 오타가 없는지 확인하거나 다른 이름으로 입력해 주세요.",
                    suggest_places(q),
                )
        if found:
            return found

    return _search_osm(q)


def _search_osm(q: str) -> dict:
    """OpenStreetMap Nominatim 검색 (카카오 키가 없거나 카카오 검색이 실패했을 때의 대체 수단)."""
    import requests  # osmnx 의존성으로 이미 설치되어 있음

    west, south, east, north = CHANGWON_VIEWBOX
    params = {
        "q": q if "창원" in q else f"{q}, 창원시",
        "format": "jsonv2", "limit": 1, "countrycodes": "kr", "accept-language": "ko",
        "viewbox": f"{west},{north},{east},{south}", "bounded": 1,
    }
    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=NOMINATIM_HEADERS,
                            timeout=NOMINATIM_TIMEOUT_S)
    except requests.RequestException:
        raise GeocodeError(
            "위치 검색 서버가 응답하지 않아요. 잠시 후 다시 시도하거나, 목록에 있는 장소를 선택해 주세요.",
            suggest_places(q),
        )
    if resp.status_code != 200:
        raise GeocodeError(
            "위치 검색 요청이 많아 잠시 제한되었어요. 잠시 후 다시 시도하거나, 목록에 있는 장소를 선택해 주세요.",
            suggest_places(q),
        )
    try:
        results = resp.json()
    except ValueError:
        results = []
    if not results:
        raise GeocodeError(
            f"'{q}'을(를) 창원시 안에서 찾지 못했어요. 오타가 없는지 확인하거나 다른 이름으로 입력해 주세요.",
            suggest_places(q),
        )

    top = results[0]
    lat, lon = float(top["lat"]), float(top["lon"])
    parts = [p.strip() for p in str(top.get("display_name", q)).split(",")]
    label = " · ".join(parts[:3])
    is_road = top.get("category") == "highway" and top.get("type") != "bus_stop"
    warning = None
    if is_road:
        road = parts[0] if parts else q
        if _ROAD_ADDRESS_RE.search(q):
            warning = (f"도로명 주소의 건물번호는 인식하지 못해, '{road}' 도로 위의 한 지점으로 잡혔어요. "
                       "실제 위치와 수백 m 이상 차이가 날 수 있으니, 정확한 위치가 필요하면 장소 이름(예: 창원시청)으로 입력해 주세요.")
        else:
            warning = (f"'{road}' 도로 위의 한 지점으로 잡혔어요. 도로가 길면 원하는 위치와 차이가 날 수 있으니, "
                       "가능하면 건물·장소 이름으로 입력해 주세요.")
    return {"lat": lat, "lon": lon, "label": label, "source": "osm",
            "precision": "road" if is_road else "place", "warning": warning}




def geocode_point(query: str, kakao_key: str = None):
    """장소 이름 -> (lat, lon). 이전 버전 호환용 래퍼 (실패 시 GeocodeError)."""
    p = search_place(query, kakao_key)
    return (p["lat"], p["lon"])


# --------------------------------------------------------------------------
# 3. OD 반경 기반 부분그래프 추출
# --------------------------------------------------------------------------

def extract_od_subgraph(G_full, orig_point, dest_point,
                         margin_ratio: float = 1.5, margin_m: float = 2000):
    """
    출발/도착 두 지점의 직선거리를 기준으로 원형 버퍼를 만들어 부분그래프를 잘라냅니다.

    행정경계(예: 성산구) 기준으로 자르면 OD가 경계를 넘을 때 엉뚱한 노드로
    스냅되는 문제가 있었기 때문에 (예: 창원대학교는 실제로는 의창구),
    반경 기반 방식을 사용합니다 -> 어떤 OD 쌍이 와도 동작하는 확장 가능한 방식.

    G_full에 이미 계산되어 있는 risk_score, cost_bike_priority, cost_risk_aware 등의
    엣지 속성은 truncate 과정에서 그대로 유지됩니다 (다시 계산하지 않습니다).
    """
    orig_pt = Point(orig_point[1], orig_point[0])   # (lon, lat)
    dest_pt = Point(dest_point[1], dest_point[0])

    od_gdf = gpd.GeoDataFrame(geometry=[orig_pt, dest_pt], crs="EPSG:4326")
    utm_crs = od_gdf.estimate_utm_crs()
    od_proj = od_gdf.to_crs(utm_crs)

    od_dist_m = od_proj.geometry.iloc[0].distance(od_proj.geometry.iloc[1])
    buffer_radius_m = od_dist_m * margin_ratio + margin_m

    mid_point_proj = Point(
        (od_proj.geometry.iloc[0].x + od_proj.geometry.iloc[1].x) / 2,
        (od_proj.geometry.iloc[0].y + od_proj.geometry.iloc[1].y) / 2,
    )
    buffer_poly_proj = mid_point_proj.buffer(buffer_radius_m)
    buffer_poly_wgs84 = gpd.GeoSeries([buffer_poly_proj], crs=utm_crs).to_crs("EPSG:4326").iloc[0]

    try:
        G_ssg = ox.truncate.truncate_graph_polygon(G_full, buffer_poly_wgs84, truncate_by_edge=True)
    except AttributeError:
        G_ssg = ox.truncate_graph_polygon(G_full, buffer_poly_wgs84, truncate_by_edge=True)

    return G_ssg, utm_crs, od_dist_m


def validate_snap(G, point, node_id, max_dist_m: float = 200):
    """지오코딩된 좌표가 그래프 노드에 너무 멀리 스냅되지 않았는지 확인합니다."""
    node = G.nodes[node_id]
    snap_dist = ox.distance.great_circle(point[0], point[1], node["y"], node["x"])
    return (snap_dist <= max_dist_m), snap_dist


def get_edges_gdf(G) -> gpd.GeoDataFrame:
    """그래프의 엣지를 GeoDataFrame으로 꺼냅니다 (지도/클러스터링에 사용)."""
    return ox.graph_to_gdfs(G, nodes=False, edges=True)


# --------------------------------------------------------------------------
# 4. 위험도 계산 (창원시 전체 그래프에 대해 딱 한 번 호출됩니다)
# --------------------------------------------------------------------------

def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def _first_if_list(val):
    if isinstance(val, list):
        return val[0] if val else None
    return val


def _highway_risk(val) -> float:
    return HIERARCHY_WEIGHT_MAP.get(_first_if_list(val), DEFAULT_HIERARCHY_WEIGHT)


def _infra_bonus(val) -> float:
    """
    한 도로명에 여러 유형이 함께 매칭된 경우(예: "자전거보행자겸용도로, 자전거전용도로")
    문자열 전체로는 매핑에 없어서 보호 없음(0)으로 처리되던 문제가 있어,
    쉼표로 나눠 각 유형을 인식하도록 고칩니다.

    도로명 단위 매칭이라 그 도로의 어느 구간이 어떤 유형인지는 구분할 수 없으므로,
    안전 서비스 특성상 보호 수준을 과대평가하지 않도록 "가장 낮은" 보호 수준을 적용합니다.
    """
    items = val if isinstance(val, list) else [val]
    levels = []
    for item in items:
        if not isinstance(item, str):
            continue
        for part in item.split(","):
            part = part.strip()
            if part in INFRA_PROTECTION_MAP:
                levels.append(INFRA_PROTECTION_MAP[part])
    return min(levels) if levels else DEFAULT_INFRA_PROTECTION


def compute_risk_scores(G,
                         w_accident: float = W_ACCIDENT,
                         w_hierarchy: float = W_HIERARCHY,
                         w_infra: float = W_INFRA,
                         bike_penalty: float = BIKE_PENALTY,
                         k_risk: float = K_RISK):
    """
    그래프의 엣지마다 risk_score(0~1)와 두 가지 라우팅 비용
    (cost_bike_priority, cost_risk_aware)을 계산해서 그래프 속성으로 다시 심어줍니다.

    prepare_g_full()을 통해 G_full 전체에 대해 딱 한 번만 호출하는 것을 전제로 합니다
    (요청마다 잘라낸 작은 부분그래프에 대해 다시 호출하면 정규화 기준이
    국지적으로 흔들리므로 호출하지 않습니다).

    주의: ox.graph_to_gdfs()가 반환하는 GeoDataFrame은 복사본이라
    여기서 직접 값을 수정해도 원본 그래프에는 반영되지 않습니다.
    그래서 계산이 끝난 뒤 ox.graph_from_gdfs()로 그래프를 다시 만들고,
    nx.set_edge_attributes로 각 속성을 명시적으로 심어줍니다.
    """
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    # --- (1) 사고 근접도 컴포넌트 ---
    # 실제 체크포인트(changwon_G_full_with_risk.pkl)의 컬럼명은 노트북 쪽 조인 단계에서
    # distance_to_accident_hotspot / nearest_hotspot_severity / hotspot_count_300m 로
    # 저장돼 있습니다. 예전 이름(acc_dist_m 등)만 찾던 아래 조건이 항상 실패해서
    # accident_component가 모든 구간에서 계속 0으로 나오고 있었습니다 (도로 위계·인프라만으로
    # risk_score가 결정 — "위험도가 경로를 바꿔도 고정되어 보인다"던 문제의 원인).
    # 두 이름 체계를 모두 지원하도록 매핑해서 고칩니다.
    _dist_col = next(
        (c for c in ("distance_to_accident_hotspot", "acc_dist_m") if c in edges.columns), None
    )
    _severity_col = next(
        (c for c in ("nearest_hotspot_severity", "acc_severity") if c in edges.columns), None
    )
    _density_col = next(
        (c for c in ("hotspot_count_300m", "hotspot_count_100m", "acc_density") if c in edges.columns),
        None,
    )
    if _dist_col and _severity_col and _density_col:
        proximity = 1 / (1 + edges[_dist_col].fillna(9999) / 100)
        severity_w = 1 + np.log1p(edges[_severity_col].fillna(0))
        density_w = 1 + np.log1p(edges[_density_col].fillna(0))
        accident_raw = proximity * severity_w * density_w
    else:
        accident_raw = pd.Series(0.0, index=edges.index)
    accident_norm = _minmax(accident_raw)

    # --- (2) 도로 위계 컴포넌트 ---
    hierarchy_risk = edges["highway"].apply(_highway_risk)

    # --- (3) 보호 인프라 보너스 ---
    infra_col = "bike_infra_type" if "bike_infra_type" in edges.columns else None
    if infra_col:
        infra_bonus = edges[infra_col].apply(_infra_bonus)
    else:
        infra_bonus = pd.Series(0.0, index=edges.index)

    raw_score = (w_accident * accident_norm
                 + w_hierarchy * hierarchy_risk
                 - w_infra * infra_bonus).clip(lower=0)
    risk_score = _minmax(raw_score)

    edges["accident_component"] = accident_norm
    edges["hierarchy_component"] = hierarchy_risk
    edges["infra_bonus_component"] = infra_bonus
    edges["risk_score"] = risk_score

    has_bike_infra = infra_bonus > 0
    edges["cost_bike_priority"] = edges["length"] * np.where(has_bike_infra, 1.0, bike_penalty)
    edges["cost_risk_aware"] = edges["length"] * (1 + k_risk * risk_score)

    G_out = ox.graph_from_gdfs(nodes, edges)

    attr_cols = ["risk_score", "accident_component", "hierarchy_component",
                 "infra_bonus_component", "cost_bike_priority", "cost_risk_aware"]
    for col in attr_cols:
        nx.set_edge_attributes(G_out, edges[col].to_dict(), col)

    return G_out, edges


# --------------------------------------------------------------------------
# 5. 경로 계산 + 통계
# --------------------------------------------------------------------------

def estimate_minutes(length_m: float, speed_kmh: float = AVG_BIKE_SPEED_KMH) -> float:
    """
    거리(m)를 평균 주행속도 가정 하에 예상 소요시간(분)으로 변환합니다.
    실측 주행속도 데이터가 없어 사용하는 근사치이며, 모든 경로에 같은 속도를
    적용하므로 경로 간 시간 차이는 순수하게 거리 차이만 반영합니다
    (특정 도로 유형이 더 빠르거나 느리다는 가정은 넣지 않았습니다).
    """
    speed_m_per_min = speed_kmh * 1000 / 60
    return length_m / speed_m_per_min if speed_m_per_min > 0 else 0.0


def _edge_data(G, u, v):
    """MultiDiGraph에서 병렬 엣지가 있을 때, 가장 짧은 엣지를 대표로 사용합니다."""
    return min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", float("inf")))


def route_stats(G, route):
    total_length = 0.0
    risk_exposure = 0.0
    max_risk = 0.0
    max_risk_edge = None

    for u, v in zip(route[:-1], route[1:]):
        data = _edge_data(G, u, v)
        length = data.get("length", 0.0)
        risk = data.get("risk_score", 0.0)
        total_length += length
        risk_exposure += length * risk
        if max_risk_edge is None or risk > max_risk:
            max_risk = risk
            max_risk_edge = {
                "u": u, "v": v, "risk_score": risk,
                "name": _first_if_list(data.get("name")),
                "highway": _first_if_list(data.get("highway")),
                "length": length,
                "accident_component": data.get("accident_component", 0.0),
                "hierarchy_component": data.get("hierarchy_component", 0.0),
                "infra_bonus_component": data.get("infra_bonus_component", 0.0),
            }

    avg_risk_per_m = risk_exposure / total_length if total_length > 0 else 0.0
    return {
        "length_m": total_length,
        "risk_exposure": risk_exposure,
        "avg_risk_per_m": avg_risk_per_m,
        "eta_min": estimate_minutes(total_length),
        "max_risk_edge": max_risk_edge,
    }


def compute_three_routes(G_ssg, orig_point, dest_point):
    orig_node = ox.distance.nearest_nodes(G_ssg, orig_point[1], orig_point[0])
    dest_node = ox.distance.nearest_nodes(G_ssg, dest_point[1], dest_point[0])

    route_shortest = nx.shortest_path(G_ssg, orig_node, dest_node, weight="length")
    route_bike = nx.shortest_path(G_ssg, orig_node, dest_node, weight="cost_bike_priority")
    route_risk = nx.shortest_path(G_ssg, orig_node, dest_node, weight="cost_risk_aware")

    routes = {
        "shortest": route_shortest,
        "bike": route_bike,
        "risk": route_risk,
    }
    stats = {key: route_stats(G_ssg, r) for key, r in routes.items()}
    return routes, stats, orig_node, dest_node


# --------------------------------------------------------------------------
# 6. 위험구간 클러스터링 (구간 단위가 아니라 "지점" 단위로 보여주기 위해)
# --------------------------------------------------------------------------

def cluster_high_risk_edges(edges_high: gpd.GeoDataFrame):
    """
    서로 끝점을 공유하는(=도로상에서 실제로 이어지는) 고위험 엣지들을
    하나의 클러스터로 묶습니다. 엣지 하나하나(보통 50~150m)를 반투명 선으로
    따로따로 그리면 여러 개가 겹쳐서 "넓은 얼룩"처럼 보이는 문제가 있어서,
    이어지는 구간을 지점(마커) 하나로 합쳐서 보여줍니다.

    반환값: [{"lat", "lon", "max_risk", "total_length", "n_edges", "highway", "names"}, ...]
    """
    if edges_high.empty:
        return []

    idx_list = list(edges_high.index)  # (u, v, key) 형태의 MultiIndex

    node_to_edges = {}
    for idx in idx_list:
        u, v = idx[0], idx[1]
        node_to_edges.setdefault(u, []).append(idx)
        node_to_edges.setdefault(v, []).append(idx)

    H = nx.Graph()
    H.add_nodes_from(idx_list)
    for shared_idxs in node_to_edges.values():
        for i in range(len(shared_idxs)):
            for j in range(i + 1, len(shared_idxs)):
                H.add_edge(shared_idxs[i], shared_idxs[j])

    clusters = []
    for component in nx.connected_components(H):
        rows = edges_high.loc[list(component)]
        max_row = rows.loc[rows["risk_score"].idxmax()]
        try:
            rep_point = max_row.geometry.interpolate(0.5, normalized=True)
            lat, lon = rep_point.y, rep_point.x
        except Exception:
            continue

        # 양방향 도로는 (u,v)와 (v,u) 두 개의 엣지로 같은 물리적 구간이 중복
        # 등록되어 있을 수 있으므로, 구간 길이를 셀 때는 무방향 쌍 기준으로
        # 한 번만 셉니다 (안 그러면 길이가 2배로 표시됨).
        rows_reset = rows.reset_index()
        rows_reset["_pair"] = rows_reset.apply(
            lambda r: tuple(sorted((r["u"], r["v"]))), axis=1
        )
        dedup = rows_reset.drop_duplicates(subset="_pair")

        names = sorted({n for n in rows["name"].apply(_first_if_list).dropna().tolist()})
        clusters.append({
            "lat": lat,
            "lon": lon,
            "max_risk": float(max_row["risk_score"]),
            "total_length": float(dedup["length"].sum()),
            "n_edges": int(len(dedup)),
            "highway": _first_if_list(max_row.get("highway")),
            "names": names,
            "edges": set(component),  # 이 지점에 묶인 구간들 (u, v, key)
        })
    return clusters


# --------------------------------------------------------------------------
# 7. 지도 시각화
# --------------------------------------------------------------------------

ROUTE_STYLE = {
    "shortest": {"label": "최단거리", "color": "#4a90d9", "offset_m": -4},
    "bike": {"label": "자전거도로 우선", "color": "#2e8b57", "offset_m": 0},
    "risk": {"label": "AI 안전경로", "color": "#d94a4a", "offset_m": 4},
}
RISK_ZONE_COLOR = "#e8622c"


def offset_route_latlon(G, route, utm_crs, offset_m):
    """경로를 좌우로 살짝 밀어서, 겹치는 구간에서도 3개 경로가 서로 가려지지 않게 합니다."""
    coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in route]
    line = LineString(coords)
    line_proj = gpd.GeoSeries([line], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
    if offset_m != 0:
        try:
            line_proj = offset_curve(line_proj, offset_m)
        except Exception:
            pass  # 오프셋 실패 시(너무 짧은 구간 등) 원본 라인 사용
    line_wgs84 = gpd.GeoSeries([line_proj], crs=utm_crs).to_crs("EPSG:4326").iloc[0]
    return [(lat, lon) for lon, lat in line_wgs84.coords]


ENDPOINT_INK = "#1f2937"  # 출발/도착 표시색: 경로색(파랑·초록·빨강)·위험 지점(주황)과 겹치지 않는 짙은 회색


def _short_label(text: str, n: int = 14) -> str:
    text = str(text or "").split(" · ")[0].strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _endpoint_marker(point, kind: str, place_label: str = ""):
    """
    출발/도착 지점을 글자가 보이는 핀으로 표시합니다.
    예전의 파란/빨간 기본 핀은 경로색(최단거리=파랑, AI 안전경로=빨강)과 겹쳐서
    어느 쪽이 출발이고 도착인지 헷갈렸기 때문에, 경로와 겹치지 않는 짙은 회색으로
    "출발"/"도착" 글자를 직접 보여줍니다 (출발=채운 핀, 도착=테두리 핀).
    """
    is_start = kind == "start"
    word = "출발" if is_start else "도착"
    name = _short_label(place_label)
    text = f"{word} · {name}" if name else word
    bg, fg = (ENDPOINT_INK, "#ffffff") if is_start else ("#ffffff", ENDPOINT_INK)
    html = f"""
    <div style="position:absolute; transform:translate(-50%, -100%); display:flex;
                flex-direction:column; align-items:center; pointer-events:auto;">
      <div style="background:{bg}; color:{fg}; border:2px solid {ENDPOINT_INK};
                  border-radius:999px; padding:3px 10px; font:700 12px/1.3 sans-serif;
                  white-space:nowrap; box-shadow:0 1px 4px rgba(0,0,0,0.35);">{text}</div>
      <div style="width:0; height:0; border-left:6px solid transparent;
                  border-right:6px solid transparent; border-top:8px solid {ENDPOINT_INK};"></div>
      <div style="width:10px; height:10px; margin-top:-2px; border-radius:50%;
                  background:{bg}; border:2px solid {ENDPOINT_INK};"></div>
    </div>"""
    return folium.Marker(
        point,
        icon=folium.DivIcon(html=html, icon_size=(1, 1), icon_anchor=(0, -5)),
        tooltip=f"{word}: {place_label}" if place_label else word,
        z_index_offset=1000,
    )


def legend_html(visible_routes=None) -> str:
    """지도 위가 아니라 지도 바로 위에 한 줄로 놓는 간단한 범례 (웹 앱용)."""
    if visible_routes is None:
        visible_routes = list(ROUTE_STYLE.keys())
    pill = ("display:inline-block; border:2px solid {ink}; border-radius:999px; padding:0 7px; "
            "font-weight:700; font-size:11px; line-height:16px; background:{bg}; color:{fg};")
    items = [
        f'<span style="{pill.format(ink=ENDPOINT_INK, bg=ENDPOINT_INK, fg="#fff")}">출발</span>'
        f'<span style="{pill.format(ink=ENDPOINT_INK, bg="#fff", fg=ENDPOINT_INK)}; margin-left:4px;">도착</span>'
    ]
    for key, st_ in ROUTE_STYLE.items():
        if key in visible_routes:
            items.append(f'<span><span style="display:inline-block; width:16px; height:4px; border-radius:2px; '
                         f'background:{st_["color"]}; vertical-align:middle; margin-right:5px;"></span>{st_["label"]}</span>')
    items.append(f'<span><img src="{RISK_MARKER_ICON_DATA_URI}" style="width:15px; height:15px; '
                 f'vertical-align:-3px; margin-right:4px;" alt="">피할 수 없는 위험 지점</span>')
    items.append(f'<span><span style="display:inline-block; width:14px; height:14px; border-radius:50%; '
                 f'background:{RISK_ZONE_COLOR}; color:#fff; font:800 10px/14px sans-serif; text-align:center; '
                 f'vertical-align:-2px; margin-right:4px;">!</span>고위험 지점(일부 경로)</span>')
    return (
        '<div style="display:flex; flex-wrap:wrap; align-items:center; gap:4px 14px; '
        'font-size:12.5px; margin:4px 0 6px 0;">' + "".join(items) + "</div>"
        '<div style="font-size:11px; color:#8a8a8a; margin-bottom:6px;">'
        f"고위험 = 창원시 전체 도로 중 위험도 상위 {(1 - CITY_RISK_QUANTILE) * 100:.0f}% · "
        "지도 위 지점을 누르면 경로별로 몇 m 지나는지 보여줘요</div>"
    )


def build_comparison_map(G_ssg, edges_ssg, routes, stats, orig_point, dest_point, utm_crs,
                          city_threshold: float, visible_routes=None,
                          orig_label: str = "", dest_label: str = "",
                          show_legend: bool = True):
    """
    city_threshold: prepare_g_full()에서 계산된, 창원시 전체 기준 위험구간 임계값.
    visible_routes: 지도에 그릴 경로 키 목록 (기본값: 3개 전부).
    """
    if visible_routes is None:
        visible_routes = list(ROUTE_STYLE.keys())

    center = [
        (orig_point[0] + dest_point[0]) / 2,
        (orig_point[1] + dest_point[1]) / 2,
    ]
    fmap = folium.Map(location=center, zoom_start=15, tiles="OpenStreetMap")
    fmap.get_root().header.add_child(folium.Element('<meta charset="utf-8">'))

    # --- 경로 (오프셋 적용, 겹침 방지, 선택된 것만 표시) ---
    for key, style in ROUTE_STYLE.items():
        if key not in visible_routes:
            continue
        route = routes[key]
        latlon = offset_route_latlon(G_ssg, route, utm_crs, style["offset_m"])
        s = stats[key]
        popup = (f"{style['label']}<br>"
                 f"거리: {s['length_m']:.0f}m<br>"
                 f"예상 소요시간: 약 {s['eta_min']:.0f}분<br>"
                 f"평균 위험도: {s['avg_risk_per_m'] * 100:.0f}점")
        folium.PolyLine(
            latlon, color=style["color"], weight=5, opacity=0.9,
            tooltip=style["label"], popup=popup,
        ).add_to(fmap)

    # --- 위험 지점: 세 경로가 "실제로 지나가는" 고위험 구간만 지점(클러스터) 단위로 표시 ---
    # 예전에는 경로 주변 250m 안의 고위험 구간을 모두 표시했는데, 그러면 AI 안전경로가
    # 큰 도로를 "가로지르기만" 해도 그 도로의 위험 지점이 경로 옆에 찍혀서, 실제로는 위험 구간을
    # 가장 적게 지나는 AI 경로 주변에 마커가 더 많아 보이는 착시가 생겼습니다.
    # 이제는 경로가 실제로 달리는 구간만 표시하고, 지점마다 어느 경로가 몇 m 지나는지 알려줍니다.
    route_weight = {"shortest": "length", "bike": "cost_bike_priority", "risk": "cost_risk_aware"}
    traversed = {}  # (u, v, key) -> {route_key: length}
    for rkey, route in routes.items():
        w = route_weight.get(rkey, "length")
        for u, v in zip(route[:-1], route[1:]):
            data = G_ssg.get_edge_data(u, v)
            k = min(data, key=lambda kk: data[kk].get(w, data[kk].get("length", 0)))
            d = data[k]
            if d.get("risk_score", 0.0) >= city_threshold:
                traversed.setdefault((u, v, k), {})[rkey] = d.get("length", 0.0)

    hi_idx = [i for i in traversed if i in edges_ssg.index]
    clusters = cluster_high_risk_edges(edges_ssg.loc[hi_idx]) if hi_idx else []

    for c in clusters:
        by_route = {}
        for e in c["edges"]:
            for rkey, length in traversed.get(e, {}).items():
                by_route[rkey] = by_route.get(rkey, 0.0) + length
        unavoidable = all(k in by_route for k in routes)
        route_lines = []
        for rkey, style in ROUTE_STYLE.items():
            if rkey not in routes:
                continue
            m = by_route.get(rkey)
            route_lines.append(f"{style['label']}: " + (f"약 {m:.0f}m 통과" if m else "지나지 않음"))
        title = "🚧 피할 수 없는 위험 지점" if unavoidable else "⚠️ 고위험 지점"
        detail = f"위험도 {c['max_risk'] * 100:.0f}점" + (f" · {', '.join(c['names'][:2])}" if c["names"] else "")
        popup_html = f"<b>{title}</b><br>{detail}<br>" + "<br>".join(route_lines)
        tooltip = f"{title} · {detail}"
        near_endpoint = min(
            ox.distance.great_circle(c["lat"], c["lon"], orig_point[0], orig_point[1]),
            ox.distance.great_circle(c["lat"], c["lon"], dest_point[0], dest_point[1]),
        ) < 120
        if unavoidable:
            # "피할 수 없는 위험 구간" 패널과 같은 바리케이드 아이콘.
            # 출발·도착 핀 바로 옆이면 핀 글자(위쪽)에 가려지지 않도록 지점의 오른쪽에 붙여 그립니다.
            icon = folium.CustomIcon(icon_image=RISK_MARKER_ICON_DATA_URI, icon_size=(34, 32),
                                     icon_anchor=(-10, 16) if near_endpoint else (17, 30))
        else:
            icon = folium.DivIcon(
                html=(f'<div style="transform:translate(-50%,-50%); width:22px; height:22px; '
                      f'border-radius:50%; background:{RISK_ZONE_COLOR}; border:2px solid #fff; '
                      f'box-shadow:0 1px 3px rgba(0,0,0,0.4); color:#fff; font:800 13px/22px sans-serif; '
                      f'text-align:center;">!</div>'),
                icon_size=(1, 1), icon_anchor=(0, 0))
        folium.Marker(location=(c["lat"], c["lon"]), icon=icon, tooltip=tooltip,
                      popup=folium.Popup(popup_html, max_width=260),
                      z_index_offset=2000 if unavoidable else 500).add_to(fmap)

    # --- 출발/도착 마커 ---
    _endpoint_marker(orig_point, "start", orig_label).add_to(fmap)
    _endpoint_marker(dest_point, "end", dest_label).add_to(fmap)

    # 출발·도착과 세 경로가 모두 한 화면에 들어오도록 지도 범위를 맞춥니다
    # (고정 줌이면 먼 거리 경로에서 출발·도착 핀이 화면 밖으로 나가 어디가 어딘지 알 수 없음).
    lats = [orig_point[0], dest_point[0]]
    lons = [orig_point[1], dest_point[1]]
    for r in routes.values():
        lats += [G_ssg.nodes[n]["y"] for n in r]
        lons += [G_ssg.nodes[n]["x"] for n in r]
    # 왼쪽 아래 범례에 경로가 가리지 않도록 아래쪽 여백을 넉넉히 둡니다.
    fmap.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]],
                    padding_top_left=(50, 50),
                    padding_bottom_right=(50, 160 if show_legend else 50))

    # --- 범례 ---
    # 웹 앱에서는 지도를 가리지 않도록 범례를 지도 밖(legend_html())에 그리므로 show_legend=False로 호출합니다.
    if not show_legend:
        return fmap
    legend_rows = "".join(
        f'<div><span style="display:inline-block;width:14px;height:4px;'
        f'background:{s["color"]};margin-right:6px;"></span>{s["label"]}</div>'
        for key, s in ROUTE_STYLE.items() if key in visible_routes
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-size: 13px;
                font-family: sans-serif; color:#333;">
        <div style="display:flex; gap:10px; margin-bottom:6px;">
            <span style="background:{ENDPOINT_INK}; color:#fff; border:2px solid {ENDPOINT_INK};
                  border-radius:999px; padding:0 8px; font-weight:700; font-size:12px;">출발</span>
            <span style="background:#fff; color:{ENDPOINT_INK}; border:2px solid {ENDPOINT_INK};
                  border-radius:999px; padding:0 8px; font-weight:700; font-size:12px;">도착</span>
        </div>
        {legend_rows}
        <div style="margin-top:6px; display:flex; align-items:center;">
            <img src="{RISK_MARKER_ICON_DATA_URI}" style="width:16px;height:16px;
            margin-right:6px;" alt="">피할 수 없는 위험 지점 (세 경로 모두 통과)
        </div>
        <div style="margin-top:3px; display:flex; align-items:center;">
            <span style="display:inline-block; width:14px; height:14px; border-radius:50%;
                  background:{RISK_ZONE_COLOR}; color:#fff; font:800 10px/14px sans-serif;
                  text-align:center; margin-right:8px;">!</span>고위험 지점 (일부 경로만 통과)
        </div>
        <div style="margin-top:3px; color:#777; font-size:11px;">
            고위험 = 창원시 전체 도로 중 위험도 상위 {(1 - CITY_RISK_QUANTILE) * 100:.0f}% · 지점을 누르면 경로별 통과 거리
        </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap
