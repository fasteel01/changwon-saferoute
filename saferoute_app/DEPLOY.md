# Streamlit Cloud 배포 가이드

## 0. 배포 전 체크 — pkl 파일 용량

```bash
ls -lh changwon_G_full_with_risk.pkl
```

- **100MB 미만**: `.gitignore`에서 `changwon_G_full_with_risk.pkl` 줄을 지우고 그냥 같이 커밋하면 됩니다.
- **100MB 이상**: GitHub 기본 업로드가 막힙니다. 아래 둘 중 하나를 선택하세요.
  - **Git LFS 사용**: `git lfs install && git lfs track "*.pkl"` 후 커밋 (Streamlit Cloud도 LFS를 지원합니다)
  - **외부 스토리지에서 받아오기**: 구글드라이브 등에 올려두고, `risk_routing.load_g_full()`이 파일이 없으면 다운로드하도록 코드를 조금 손보는 방식 (필요하면 말씀해주세요, 코드 수정해드릴게요)

## 1. 저장소 준비 (본인이 직접 진행)

```bash
cd saferoute_app   # app.py, risk_routing.py, requirements.txt, packages.txt, .gitignore 있는 폴더
git init
git add app.py risk_routing.py requirements.txt packages.txt .gitignore DEPLOY.md
# pkl을 같이 올리기로 했다면:
# git add changwon_G_full_with_risk.pkl
git commit -m "Initial commit: 누비자 세이프루트 MVP"
```

GitHub에서 새 저장소를 만든 뒤 (Public 또는 Private 둘 다 Streamlit Cloud 무료 티어에서 배포 가능합니다):

```bash
git remote add origin https://github.com/<본인계정>/<저장소이름>.git
git branch -M main
git push -u origin main
```

## 2. Streamlit Cloud 연결

1. [share.streamlit.io](https://share.streamlit.io) 접속 → GitHub 계정으로 로그인
2. **"New app"** 클릭
3. Repository: 방금 만든 저장소 선택
4. Branch: `main`
5. Main file path: `app.py`
6. **Advanced settings**에서 Python 버전 3.11 정도로 지정 (선택)
7. **Deploy** 클릭

빌드 로그가 뜨는데, `packages.txt`의 apt 패키지들을 먼저 설치하고 `requirements.txt`의 pip 패키지들을 설치합니다. geopandas/osmnx 설치는 꽤 오래 걸릴 수 있어요 (5~10분).

## 3. API 키가 있다면

만약 TAAS/data.go.kr API 키를 코드 어딘가에서 쓰고 있다면 (`risk_routing.py`나 노트북 어디에도 하드코딩되어 있으면 안 됩니다):

1. Streamlit Cloud 앱 화면 → **Settings → Secrets**
2. TOML 형식으로 입력:
   ```toml
   TAAS_API_KEY = "실제키값"
   ```
3. 코드에서는 `st.secrets["TAAS_API_KEY"]`로 불러오면 됩니다.

(지금 드린 `risk_routing.py`/`app.py`는 배포 시점에 API를 새로 호출하지 않고, 미리 만들어둔 pkl 체크포인트만 읽으니 이 단계는 필요 없을 가능성이 높아요 — 혹시 지오코딩(`geocode_point`)에서 Nominatim을 호출하는 부분만 네트워크를 쓰는데, 이건 키가 필요 없습니다.)

## 4. 흔한 에러

- **"No module named 'fiona'" 또는 GDAL 관련 에러**: `packages.txt`가 제대로 인식됐는지 확인 (저장소 최상위, `requirements.txt`와 같은 위치에 있어야 함). Manage app → Reboot으로 재시도.
- **메모리 초과 / 앱이 죽음**: 무료 티어는 리소스가 1GB 정도로 제한적입니다. 창원시 전체 그래프(2.4만 노드)가 너무 크면 감당이 안 될 수 있어요. 이 경우 `st.cache_resource`가 이미 최초 1회만 로드하도록 해뒀으니, 그래도 안 되면 유료 티어나 별도 서버 배포를 고려해야 합니다.
- **빌드가 너무 오래 걸림**: geopandas/osmnx 계열은 원래 느립니다. 10분 넘게 멈춰있지 않다면 기다려보세요.

## 5. 업데이트하는 법

로컬에서 파일 수정 → `git add . && git commit -m "..." && git push` 하면 Streamlit Cloud가 자동으로 감지해서 재배포합니다.
