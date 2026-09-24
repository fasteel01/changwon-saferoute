"""
app.py
누비자 세이프루트 - Streamlit 메인 앱 (3차 개정)

실행 방법:
    streamlit run app.py

전제 조건:
    risk_routing.py, changwon_G_full_with_risk.pkl 이 이 파일과 같은 디렉토리에 있어야 합니다.

필요 패키지 (requirements.txt 참고):
    streamlit, streamlit-folium, osmnx, networkx, geopandas, shapely, folium, pandas

이번 개정에서 바뀐 점 (risk_routing.py는 건드리지 않았습니다):
    "AI 안전경로가 왜 이렇게 나왔는지" 사용자가 납득할 수 있도록, 경로를 다시 한번
    구간별로 뜯어봐서(자전거 인프라 비율, 큰 도로 비율, 고위험구간 통과 횟수) 세 경로를
    직접 비교하는 설명 패널을 추가했습니다. 이 계산은 risk_routing.compute_risk_scores()가
    이미 G_ssg 엣지에 심어둔 속성(risk_score, hierarchy_component, infra_bonus_component)을
    그대로 읽기만 하므로 백엔드 로직 변경 없이 app.py 안에서 끝납니다.
"""

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import risk_routing as rr


st.set_page_config(page_title="누비자 안전경로 추천", layout="wide", page_icon="🚲")

st.title("🚲 누비자 세이프루트")
st.caption("성산구 · 의창구 시범 구간 · MVP")

ROUTE_LABELS = {"shortest": "최단거리", "bike": "자전거도로 우선", "risk": "AI 안전경로"}
ROUTE_DOTS = {"shortest": "🔵", "bike": "🟢", "risk": "🔴"}
MAJOR_ROAD_HIERARCHY_THRESHOLD = 0.6  # hierarchy_component 기준: secondary 이상을 "큰 도로"로 취급

# 지도 위험 지점 마커·범례와 동일한 색(RISK_ZONE_COLOR)을 그대로 써서,
# "피할 수 없는 위험 구간" 패널이 지도와 한눈에 같은 경고로 인식되게 합니다.
RISK_ACCENT = rr.RISK_ZONE_COLOR

st.markdown(
    f"""
    <style>
    .risk-alert-card {{
        border-left: 6px solid {RISK_ACCENT};
        background: linear-gradient(135deg, rgba(232,98,44,0.12), rgba(232,98,44,0.03));
        border-radius: 10px;
        padding: 20px 22px;
        margin: 4px 0 20px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.10);
    }}
    .risk-alert-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }}
    .risk-alert-icon {{ width: 30px; height: 30px; }}
    .risk-alert-title {{
        font-size: 1.35rem;
        font-weight: 800;
        color: {RISK_ACCENT};
        letter-spacing: -0.01em;
    }}
    .risk-alert-lede {{
        font-size: 1.05rem;
        margin-bottom: 12px;
        line-height: 1.55;
    }}
    .risk-alert-why {{
        background: rgba(232,98,44,0.10);
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.93rem;
        margin-bottom: 18px;
        line-height: 1.5;
    }}
    .risk-alert-body {{
        display: flex;
        gap: 28px;
        align-items: flex-start;
        flex-wrap: wrap;
    }}
    .risk-score-block {{ min-width: 130px; }}
    .risk-score-value {{
        font-size: 2.8rem;
        font-weight: 800;
        color: {RISK_ACCENT};
        line-height: 1;
    }}
    .risk-score-unit {{ font-size: 1.25rem; font-weight: 600; margin-left: 2px; }}
    .risk-score-caption {{ font-size: 0.8rem; color: #8a8a8a; margin-top: 6px; }}
    .risk-detail-block {{ flex: 1; min-width: 260px; }}
    .risk-detail-road {{ font-weight: 700; font-size: 1.05rem; }}
    .risk-detail-meta {{ font-size: 0.85rem; color: #8a8a8a; margin-bottom: 6px; }}
    .risk-bar-hint {{ font-size: 0.76rem; color: #8a8a8a; margin-bottom: 8px; font-style: italic; }}
    .risk-bar-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }}
    .risk-bar-label {{ font-size: 0.8rem; color: #8a8a8a; width: 82px; flex-shrink: 0; }}
    .risk-bar-track {{
        flex: 1; height: 9px; background: rgba(128,128,128,0.18);
        border-radius: 5px; overflow: hidden;
    }}
    .risk-bar-fill {{ height: 100%; background: {RISK_ACCENT}; border-radius: 5px; }}
    .risk-bar-pct {{ font-size: 0.78rem; color: #8a8a8a; width: 36px; text-align: right; flex-shrink: 0; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="창원시 전체 도로 그래프 불러오고 위험도 계산하는 중... (최초 1회만, 몇 분 걸릴 수 있어요)")
def get_g_full_with_risk():
    return rr.prepare_g_full()


def _edge_data(G, u, v):
    """MultiDiGraph에서 병렬 엣지가 있을 때, 가장 짧은 엣지를 대표로 사용합니다."""
    return min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", float("inf")))


def analyze_route(G, route, city_threshold, major_road_threshold=MAJOR_ROAD_HIERARCHY_THRESHOLD):
    """
    경로를 다시 한번 순회하면서 "왜 이 경로가 이렇게 나왔는지" 설명하는 데 필요한
    구성 정보를 뽑아냅니다. 이미 G_ssg 엣지에 있는 속성만 읽어서 쓰기 때문에
    risk_routing.py를 건드리지 않고 app.py 안에서 계산이 끝납니다.
    """
    total = 0.0
    bike_infra_len = 0.0
    major_road_len = 0.0
    high_risk_len = 0.0
    high_risk_segments = []

    for u, v in zip(route[:-1], route[1:]):
        data = _edge_data(G, u, v)
        length = data.get("length", 0.0)
        total += length

        if data.get("infra_bonus_component", 0.0) > 0:
            bike_infra_len += length
        if data.get("hierarchy_component", 0.0) >= major_road_threshold:
            major_road_len += length

        risk = data.get("risk_score", 0.0)
        if risk >= city_threshold:
            high_risk_len += length
            high_risk_segments.append({
                "name": data.get("name") or "이름 없는 도로",
                "highway": data.get("highway"),
                "risk_score": risk,
                "length": length,
            })

    return {
        "total_m": total,
        "bike_infra_pct": (bike_infra_len / total * 100) if total else 0.0,
        "major_road_pct": (major_road_len / total * 100) if total else 0.0,
        "high_risk_m": high_risk_len,
        "high_risk_count": len(high_risk_segments),
        "high_risk_segments": high_risk_segments,
    }


def build_recommendation_explanation(stats, profiles):
    """
    AI 안전경로가 최단거리 · 자전거도로 우선과 비교해서 왜 이렇게 나왔는지
    실제 계산된 수치를 근거로 문장을 만듭니다. AI 안전경로가 항상 더 낫다고
    가정하지 않고, 실제 수치가 반대로 나오는 경우도 있는 그대로 설명합니다.
    """
    s_short, s_bike, s_risk = stats["shortest"], stats["bike"], stats["risk"]
    p_bike, p_risk = profiles["bike"], profiles["risk"]

    parts = []

    extra_m = s_risk["length_m"] - s_short["length_m"]
    extra_min = s_risk["eta_min"] - s_short["eta_min"]
    if extra_m > 5:
        parts.append(f"최단거리보다 약 {extra_m:.0f}m(+{extra_min:.1f}분) 더 가지만,")
    else:
        parts.append("최단거리와 이동거리가 거의 같으면서,")

    risk_drop_vs_short = (s_short["avg_risk_per_m"] - s_risk["avg_risk_per_m"]) * 100
    if risk_drop_vs_short > 1:
        parts.append(f"평균 위험도는 {risk_drop_vs_short:.0f}점 더 낮습니다.")
    elif risk_drop_vs_short < -1:
        parts.append(f"평균 위험도는 오히려 {-risk_drop_vs_short:.0f}점 더 높게 나왔습니다 — 이 구간은 짧은 길이 곧 안전한 길이기도 했습니다.")
    else:
        parts.append("평균 위험도는 최단거리와 비슷합니다.")

    risk_diff_vs_bike = (s_bike["avg_risk_per_m"] - s_risk["avg_risk_per_m"]) * 100
    if risk_diff_vs_bike > 3:
        reasons = []
        if p_bike["high_risk_count"] > p_risk["high_risk_count"]:
            reasons.append(
                f"고위험구간을 {p_bike['high_risk_count']}곳 지나가는 반면, "
                f"AI 안전경로는 {p_risk['high_risk_count']}곳만 지나가고"
            )
        if p_bike["major_road_pct"] > p_risk["major_road_pct"] + 5:
            reasons.append(
                f"큰 도로 비율도 {p_bike['major_road_pct']:.0f}%로 "
                f"AI 안전경로({p_risk['major_road_pct']:.0f}%)보다 높습니다"
            )
        reason_text = " · ".join(reasons) if reasons else "전체적인 위험도가 더 높게 계산됐습니다"
        parts.append(
            f"자전거도로 우선 경로는 자전거 인프라가 있는 길({p_bike['bike_infra_pct']:.0f}%)을 "
            f"최우선으로 따라가지만, {reason_text}."
        )
        parts.append(
            f"AI 안전경로는 인프라 유무 하나만 보지 않고 사고 이력·도로 규모를 함께 반영해서, "
            f"인프라 비율은 {p_risk['bike_infra_pct']:.0f}%로 다소 낮아도 차가 적은 길을 골라 "
            f"전체 위험도를 더 낮췄습니다."
        )
    elif risk_diff_vs_bike < -3:
        parts.append(
            f"이번 구간에서는 자전거도로 우선 경로(위험도 {s_bike['avg_risk_per_m'] * 100:.0f}점)가 "
            f"AI 안전경로보다 오히려 더 안전하게 계산됐습니다 — 인프라가 갖춰진 길이 사고 이력도 "
            f"적었던 경우입니다."
        )
    else:
        parts.append("자전거도로 우선 경로와 위험도 차이는 크지 않습니다.")

    return " ".join(parts)


def _risk_bar_html(label: str, value: float) -> str:
    """'피할 수 없는 위험 구간' 카드 안의 accent색 커스텀 진행률 바 한 줄."""
    v = min(max(value, 0.0), 1.0)
    return (
        '<div class="risk-bar-row">'
        f'<div class="risk-bar-label">{label}</div>'
        '<div class="risk-bar-track">'
        f'<div class="risk-bar-fill" style="width:{v * 100:.0f}%;"></div>'
        "</div>"
        f'<div class="risk-bar-pct">{v * 100:.0f}%</div>'
        "</div>"
    )


# OSM highway 태그(영문) -> 사용자가 바로 이해할 수 있는 한글 도로 유형.
# risk_routing.HIERARCHY_WEIGHT_MAP과 같은 키 집합을 씁니다.
HIGHWAY_LABEL_KO = {
    "motorway": "고속도로", "motorway_link": "고속도로",
    "trunk": "자동차전용도로", "trunk_link": "자동차전용도로",
    "primary": "간선도로(대로)", "primary_link": "간선도로(대로)",
    "secondary": "주요도로", "secondary_link": "주요도로",
    "tertiary": "일반도로", "tertiary_link": "일반도로",
    "unclassified": "일반도로",
    "residential": "이면도로(주택가 도로)",
    "living_street": "생활도로",
    "service": "이면도로",
    "cycleway": "자전거도로",
    "footway": "보행로",
    "path": "소로(작은 길)",
}


def _highway_label_ko(value) -> str:
    v = value[0] if isinstance(value, list) else value
    return HIGHWAY_LABEL_KO.get(v, "도로")


def _road_description(e: dict) -> str:
    """도로명 + 한글 도로유형을 사용자가 바로 이해할 수 있는 한 구절로 합칩니다."""
    label = _highway_label_ko(e.get("highway"))
    name = e.get("name")
    if isinstance(name, list):
        name = name[0] if name else None
    return f"{name} ({label})" if name else f"이름 없는 {label} 구간"


col_input, col_result = st.columns([1, 2.2], gap="large")

with col_input:
    st.subheader("출발지 · 도착지")
    origin_text = st.text_input("출발지", value="창원시청")
    dest_text = st.text_input("도착지", value="창원대학교")
    run = st.button("경로 비교하기", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "이동거리 우선, 자전거도로 우선, AI 안전경로 3가지 전략으로 "
        "경로를 계산해서 비교합니다. 위험도는 사고 이력 · 도로 규모 · "
        "자전거 보호시설 유무를 반영한 값이에요 (0~100점, 낮을수록 안전, "
        "창원시 전체 도로를 기준으로 계산됩니다). 예상 소요시간은 평균 주행속도 "
        f"{rr.AVG_BIKE_SPEED_KMH:.0f}km/h 가정 기준 추정치입니다."
    )

    st.divider()
    st.markdown("**지도에 표시할 경로**")
    visible_routes = []
    for key in ["shortest", "bike", "risk"]:
        checked = st.checkbox(
            f"{ROUTE_DOTS[key]} {ROUTE_LABELS[key]}", value=True, key=f"vis_{key}"
        )
        if checked:
            visible_routes.append(key)

with col_result:
    if run:
        try:
            G_full, city_threshold = get_g_full_with_risk()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        try:
            with st.spinner("출발 · 도착지 확인 중..."):
                orig_point = rr.geocode_point(origin_text)
                dest_point = rr.geocode_point(dest_text)

            with st.spinner("경로 주변 도로 그래프 추출 중..."):
                G_ssg, utm_crs, od_dist_m = rr.extract_od_subgraph(G_full, orig_point, dest_point)
                edges_ssg = rr.get_edges_gdf(G_ssg)

            with st.spinner("3가지 경로 계산 중..."):
                routes, stats, orig_node, dest_node = rr.compute_three_routes(
                    G_ssg, orig_point, dest_point
                )
        except Exception as e:
            st.error(f"경로 계산 중 오류가 발생했습니다: {e}")
            st.stop()

        st.session_state["last_result"] = {
            "routes": routes,
            "stats": stats,
            "G_ssg": G_ssg,
            "edges_ssg": edges_ssg,
            "orig_point": orig_point,
            "dest_point": dest_point,
            "utm_crs": utm_crs,
            "city_threshold": city_threshold,
        }

    result = st.session_state.get("last_result")

    if result is None:
        st.info("왼쪽에서 출발지와 도착지를 입력하고 '경로 비교하기'를 눌러주세요.")
    else:
        stats = result["stats"]
        G_ssg = result["G_ssg"]
        city_threshold = result["city_threshold"]

        metric_cols = st.columns(3)
        for col, key in zip(metric_cols, ["shortest", "bike", "risk"]):
            s = stats[key]
            with col:
                st.metric(
                    f"{ROUTE_DOTS[key]} {ROUTE_LABELS[key]}",
                    f"약 {s['eta_min']:.0f}분",
                    f"위험도 {s['avg_risk_per_m'] * 100:.0f}점",
                    delta_color="inverse",
                )
                st.caption(f"{s['length_m']:.0f}m")

        # --- 왜 이 경로를 추천했는지 (경로별 구성 비교) ---
        try:
            profiles = {
                key: analyze_route(G_ssg, result["routes"][key], city_threshold)
                for key in ["shortest", "bike", "risk"]
            }
            with st.container(border=True):
                st.markdown("#### 🤔 왜 이 경로를 추천했나요?")
                st.write(build_recommendation_explanation(stats, profiles))

                comparison_df = pd.DataFrame(
                    {
                        "자전거 인프라 비율": [
                            f"{profiles[k]['bike_infra_pct']:.0f}%" for k in ["shortest", "bike", "risk"]
                        ],
                        "큰 도로 비율": [
                            f"{profiles[k]['major_road_pct']:.0f}%" for k in ["shortest", "bike", "risk"]
                        ],
                        "고위험구간 통과": [
                            f"{profiles[k]['high_risk_count']}곳" for k in ["shortest", "bike", "risk"]
                        ],
                        "평균 위험도": [
                            f"{stats[k]['avg_risk_per_m'] * 100:.0f}점" for k in ["shortest", "bike", "risk"]
                        ],
                    },
                    index=[f"{ROUTE_DOTS[k]} {ROUTE_LABELS[k]}" for k in ["shortest", "bike", "risk"]],
                )
                st.dataframe(comparison_df, use_container_width=True)
                st.caption(
                    "큰 도로 비율 = 2차로 이상 간선도로(secondary 이상)를 지나는 구간 비율 · "
                    f"고위험구간 = 창원시 전체 도로 중 위험도 상위 {(1 - rr.CITY_RISK_QUANTILE) * 100:.0f}% 이내 구간"
                )
        except Exception as e:
            st.warning(f"경로 비교 설명을 만드는 중 문제가 발생했습니다: {e}")

        # --- 피할 수 없는 위험 구간 ---
        edges_by_route = {k: stats[k]["max_risk_edge"] for k in stats}
        ref = next(iter(edges_by_route.values()))
        shared = all(
            e["u"] == ref["u"] and e["v"] == ref["v"] for e in edges_by_route.values()
        )

        if shared:
            e = ref
            has_infra = e.get("infra_bonus_component", 0) > 0
            why = (
                "자전거 보호시설은 있지만, "
                if has_infra
                else "자전거를 보호할 시설이 없고, "
            )
            road_desc = _road_description(e)
            pct = (1 - rr.CITY_RISK_QUANTILE) * 100

            # "보호시설" 항목은 원래 값이 높을수록(=보호시설이 잘 갖춰질수록) 위험도를
            # "낮추는" 방향이라, 사고 이력·도로 규모(높을수록 위험도를 "높이는" 방향)와
            # 막대 하나로 나란히 보여주면 방향이 반대로 읽혀 헷갈립니다.
            # 그래서 "보호시설 부족" = 1 - 보호시설 로 뒤집어서, 세 막대 모두
            # "길수록 위험도를 더 많이 끌어올린 요인"으로 방향을 통일했습니다.
            infra_lack = 1.0 - min(max(e.get("infra_bonus_component", 0.0), 0.0), 1.0)
            bars_html = (
                _risk_bar_html("사고 이력", e.get("accident_component", 0.0))
                + _risk_bar_html("도로 규모", e.get("hierarchy_component", 0.0))
                + _risk_bar_html("보호시설 부족", infra_lack)
            )

            st.markdown(
                f"""
                <div class="risk-alert-card">
                    <div class="risk-alert-header">
                        <img src="{rr.RISK_MARKER_ICON_DATA_URI}" class="risk-alert-icon" alt="위험 구간 아이콘">
                        <span class="risk-alert-title">피할 수 없는 위험 구간</span>
                    </div>
                    <div class="risk-alert-lede">
                        <strong>어떤 경로를 선택해도</strong> 이 구간은 반드시 지나가게 됩니다.
                        대신할 다른 길이 없기 때문이에요.
                    </div>
                    <div class="risk-alert-why">
                        ⚠️ 왜 위험할까요? {why}사고 이력과 도로 규모를 함께 반영했을 때 위험도가 높게 나온 구간이에요.
                    </div>
                    <div class="risk-alert-body">
                        <div class="risk-score-block">
                            <div class="risk-score-value">{e['risk_score'] * 100:.0f}<span class="risk-score-unit">점</span></div>
                            <div class="risk-score-caption">위험도 · 창원시 전체 상위 {pct:.0f}% 이내</div>
                        </div>
                        <div class="risk-detail-block">
                            <div class="risk-detail-road">{road_desc}</div>
                            <div class="risk-detail-meta">길이 약 {e['length']:.0f}m · 창원시 전체 도로 중 위험도 상위 {pct:.0f}% 이내</div>
                            <div class="risk-bar-hint">막대가 길수록 위험도를 더 많이 끌어올린 요인이에요</div>
                            {bars_html}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.expander("경로별 최고 위험 지점 보기"):
                for key in ["shortest", "bike", "risk"]:
                    e = edges_by_route[key]
                    st.write(
                        f"{ROUTE_DOTS[key]} **{ROUTE_LABELS[key]}**: "
                        f"위험도 {e['risk_score'] * 100:.0f}점 · {_road_description(e)}"
                    )

        fmap = rr.build_comparison_map(
            result["G_ssg"], result["edges_ssg"], result["routes"], result["stats"],
            result["orig_point"], result["dest_point"], result["utm_crs"],
            city_threshold=result["city_threshold"],
            visible_routes=visible_routes,
        )
        st_folium(fmap, width=None, height=600, use_container_width=True)
