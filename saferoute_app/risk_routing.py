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
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely import offset_curve
from shapely.ops import unary_union
import folium


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

# 학과서버 네트워크가 끊겨도 데모가 가능하도록, 자주 쓰는 지점은 좌표를 하드코딩해둡니다.
# (Nominatim 지오코딩은 네트워크 상태에 따라 실패할 수 있음)
DEMO_LOCATIONS = {
    "창원시청": (35.2275036, 128.682374),
    "창원대학교": (35.2448472, 128.6951022),
}

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
CORRIDOR_BUFFER_M = 250    # 위험구간을 후보 경로 주변 몇 m 안에서만 보여줄지

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

def geocode_point(query: str):
    """장소 이름 -> (lat, lon). DEMO_LOCATIONS에 있으면 그 값을 우선 사용합니다."""
    q = query.strip()
    if q in DEMO_LOCATIONS:
        return DEMO_LOCATIONS[q]
    # 데모 지점이 아니면 실시간 지오코딩 시도 (네트워크 필요)
    query_full = q if "창원" in q else f"{q}, 창원시"
    lat, lon = ox.geocode(query_full)
    return (lat, lon)


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
    return INFRA_PROTECTION_MAP.get(_first_if_list(val), DEFAULT_INFRA_PROTECTION)


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
    if {"acc_dist_m", "acc_severity", "acc_density"}.issubset(edges.columns):
        proximity = 1 / (1 + edges["acc_dist_m"].fillna(9999) / 100)
        severity_w = 1 + np.log1p(edges["acc_severity"].fillna(0))
        density_w = 1 + np.log1p(edges["acc_density"].fillna(0))
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


def build_comparison_map(G_ssg, edges_ssg, routes, stats, orig_point, dest_point, utm_crs,
                          city_threshold: float, visible_routes=None):
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

    # --- 위험구간: 창원시 전체 기준 상위 분위 & 후보 경로 250m 코리더 안 & 지점(클러스터) 단위 ---
    # 코리더는 토글 상태와 무관하게 항상 3개 경로 전체를 기준으로 계산합니다
    # (경로를 껐다 켰다 할 때 위험구간 표시가 같이 흔들리지 않도록).
    route_lines_proj = []
    for key in routes:
        latlon = offset_route_latlon(G_ssg, routes[key], utm_crs, 0)
        line = LineString([(lon, lat) for lat, lon in latlon])
        route_lines_proj.append(gpd.GeoSeries([line], crs="EPSG:4326").to_crs(utm_crs).iloc[0])
    corridor = unary_union(route_lines_proj).buffer(CORRIDOR_BUFFER_M)

    edges_proj = edges_ssg.to_crs(utm_crs)
    high_risk_nearby = edges_ssg[
        (edges_ssg["risk_score"] >= city_threshold) & (edges_proj.geometry.intersects(corridor))
    ]
    clusters = cluster_high_risk_edges(high_risk_nearby)

    for c in clusters:
        popup_lines = [
            f"위험도 {c['max_risk'] * 100:.0f}점",
            f"구간 길이 약 {c['total_length']:.0f}m",
        ]
        if c["names"]:
            popup_lines.append(", ".join(c["names"][:2]))
        folium.CircleMarker(
            location=(c["lat"], c["lon"]),
            radius=9,
            color=RISK_ZONE_COLOR,
            weight=2,
            fill=True,
            fill_color=RISK_ZONE_COLOR,
            fill_opacity=0.85,
            tooltip=f"위험도 {c['max_risk'] * 100:.0f}점 · 약 {c['total_length']:.0f}m 구간",
            popup="<br>".join(popup_lines),
        ).add_to(fmap)

    # --- 출발/도착 마커 ---
    folium.Marker(orig_point, tooltip="출발", icon=folium.Icon(color="blue")).add_to(fmap)
    folium.Marker(dest_point, tooltip="도착", icon=folium.Icon(color="red")).add_to(fmap)

    # --- 범례 ---
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
        {legend_rows}
        <div style="margin-top:4px;">
            <span style="display:inline-block;width:9px;height:9px;border-radius:50%;
            background:{RISK_ZONE_COLOR};margin-right:6px;"></span>위험 지점 (창원시 전체 상위
            {(1 - CITY_RISK_QUANTILE) * 100:.0f}%)
        </div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    return fmap
