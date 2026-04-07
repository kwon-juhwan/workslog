import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO

# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="업무일지 / 생산 대시보드",
    page_icon="📊",
    layout="wide"
)

# GitHub Raw 파일 주소
GITHUB_XLSX_URL = "https://raw.githubusercontent.com/kwon-juhwan/workslog/main/worklogs.xlsx"

# 고정 컬럼 후보
BASE_FIXED_COLS = [
    "업무일자",
    "이름",
    "부서",
    "작성일시",
    "입고수량",
    "출고수량",
    "반품수량",
    "재고확인사항",
    "배송이슈",
    "오전 업무내용",
    "오후 업무내용",
    "오전업무",
    "오후업무",
    "미처리내역",
    "예정사항",
    "특이사항",
    "Comment",
    "코멘트",
]

TEXT_COL_CANDIDATES = [
    "재고확인사항",
    "배송이슈",
    "오전 업무내용",
    "오후 업무내용",
    "오전업무",
    "오후업무",
    "미처리내역",
    "예정사항",
    "특이사항",
    "Comment",
    "코멘트",
]


# =========================
# 공통 유틸
# =========================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def unify_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    서로 다른 컬럼명을 하나로 정리
    """
    df = df.copy()

    alias_map = {
        "오전업무": "오전 업무내용",
        "오후업무": "오후 업무내용",
        "코멘트": "Comment",
    }

    for old_col, new_col in alias_map.items():
        if old_col in df.columns:
            if new_col in df.columns:
                df[new_col] = df[new_col].where(
                    df[new_col].notna() & (df[new_col].astype(str).str.strip() != ""),
                    df[old_col]
                )
                df.drop(columns=[old_col], inplace=True)
            else:
                df.rename(columns={old_col: new_col}, inplace=True)

    return df


def get_fixed_cols(df: pd.DataFrame) -> list:
    return [c for c in BASE_FIXED_COLS if c in df.columns]


def get_text_cols(df: pd.DataFrame) -> list:
    return [c for c in TEXT_COL_CANDIDATES if c in df.columns]


def ensure_fixed_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_cols = [
        "업무일자", "이름", "부서", "작성일시",
        "입고수량", "출고수량", "반품수량",
        "재고확인사항", "배송이슈",
        "오전 업무내용", "오후 업무내용",
        "미처리내역", "예정사항", "특이사항", "Comment"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "업무일자" in df.columns:
        dt = pd.to_datetime(df["업무일자"], errors="coerce")
        df["업무일자"] = dt.dt.date
        df["업무분기"] = dt.dt.to_period("Q").astype(str)

    if "작성일시" in df.columns:
        df["작성일시"] = pd.to_datetime(df["작성일시"], errors="coerce")

    return df


def detect_product_cols(df: pd.DataFrame) -> list:
    fixed_cols = set(get_fixed_cols(df)) | {"업무분기", "총생산수량"}
    product_cols = []

    for col in df.columns:
        if col in fixed_cols:
            continue
        if df[col].isna().all():
            continue
        product_cols.append(col)

    return product_cols


def preprocess_numeric_cols(df: pd.DataFrame, product_cols: list) -> pd.DataFrame:
    df = df.copy()

    qty_cols = ["입고수량", "출고수량", "반품수량"] + product_cols
    qty_cols = [c for c in qty_cols if c in df.columns]

    for col in qty_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def make_long_dataframe(df: pd.DataFrame, product_cols: list) -> pd.DataFrame:
    if not product_cols:
        return pd.DataFrame(columns=get_fixed_cols(df) + ["업무분기", "품목명", "수량"])

    id_vars = get_fixed_cols(df)
    if "업무분기" in df.columns:
        id_vars = id_vars + ["업무분기"]

    long_df = df.melt(
        id_vars=id_vars,
        value_vars=product_cols,
        var_name="품목명",
        value_name="수량"
    )
    long_df["수량"] = pd.to_numeric(long_df["수량"], errors="coerce").fillna(0)
    long_df = long_df[long_df["수량"] > 0].copy()
    return long_df


def add_total_qty(df: pd.DataFrame, product_cols: list) -> pd.DataFrame:
    df = df.copy()
    if product_cols:
        df["총생산수량"] = df[product_cols].sum(axis=1)
    else:
        df["총생산수량"] = 0
    return df


def safe_contains(series: pd.Series, keyword: str) -> pd.Series:
    if keyword.strip() == "":
        return pd.Series([True] * len(series), index=series.index)
    return series.fillna("").astype(str).str.contains(keyword, case=False, na=False)


def build_keyword_mask(df: pd.DataFrame, keyword: str) -> pd.Series:
    text_cols = get_text_cols(df)
    if not text_cols:
        return pd.Series([True] * len(df), index=df.index)

    mask = pd.Series([False] * len(df), index=df.index)
    for col in text_cols:
        mask = mask | safe_contains(df[col], keyword)
    return mask


def build_issue_mask(df: pd.DataFrame) -> pd.Series:
    issue_cols = [c for c in ["재고확인사항", "배송이슈", "미처리내역", "특이사항", "Comment"] if c in df.columns]
    if not issue_cols:
        return pd.Series([False] * len(df), index=df.index)

    mask = pd.Series([False] * len(df), index=df.index)
    for col in issue_cols:
        mask = mask | df[col].fillna("").astype(str).str.strip().ne("")
    return mask


def sort_quarter_strings(values):
    def quarter_key(q):
        try:
            y, qq = q.split("Q")
            return (int(y), int(qq))
        except:
            return (9999, 9)
    return sorted(values, key=quarter_key)


@st.cache_data(ttl=60, show_spinner=False)
def fetch_github_excel(url: str) -> bytes:
    headers = {"Cache-Control": "no-cache"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response.content


@st.cache_data(ttl=60, show_spinner=False)
def load_excel(file_bytes: bytes, sheet_name=0):
    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
    df = clean_columns(df)
    df = unify_column_aliases(df)
    df = ensure_fixed_cols(df)
    df = parse_dates(df)

    product_cols = detect_product_cols(df)
    df = preprocess_numeric_cols(df, product_cols)
    df = add_total_qty(df, product_cols)
    long_df = make_long_dataframe(df, product_cols)

    return df, long_df, product_cols


def apply_filters(df: pd.DataFrame, long_df: pd.DataFrame):
    st.sidebar.header("필터")

    min_date = None
    max_date = None
    if "업무일자" in df.columns and df["업무일자"].notna().any():
        min_date = df["업무일자"].min()
        max_date = df["업무일자"].max()

    if min_date and max_date:
        date_range = st.sidebar.date_input(
            "업무일자 범위",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date
    else:
        start_date, end_date = None, None

    names = sorted([x for x in df["이름"].dropna().astype(str).unique().tolist() if x.strip()])
    selected_names = st.sidebar.multiselect("이름", names, default=names)

    depts = sorted([x for x in df["부서"].dropna().astype(str).unique().tolist() if x.strip()])
    selected_depts = st.sidebar.multiselect("부서", depts, default=depts)

    products = sorted([x for x in long_df["품목명"].dropna().astype(str).unique().tolist() if x.strip()])
    selected_products = st.sidebar.multiselect("품목", products, default=products)

    keyword = st.sidebar.text_input("업무/이슈 키워드 검색", "")
    issue_only = st.sidebar.checkbox("이슈 있는 건만 보기", value=False)

    filtered_df = df.copy()

    if start_date and end_date and "업무일자" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["업무일자"].between(start_date, end_date, inclusive="both")
        ]

    if selected_names:
        filtered_df = filtered_df[filtered_df["이름"].astype(str).isin(selected_names)]

    if selected_depts:
        filtered_df = filtered_df[filtered_df["부서"].astype(str).isin(selected_depts)]

    if keyword.strip():
        filtered_df = filtered_df[build_keyword_mask(filtered_df, keyword)]

    if issue_only:
        filtered_df = filtered_df[build_issue_mask(filtered_df)]

    filtered_long_df = long_df.copy()

    if start_date and end_date and "업무일자" in filtered_long_df.columns:
        filtered_long_df = filtered_long_df[
            filtered_long_df["업무일자"].between(start_date, end_date, inclusive="both")
        ]

    if selected_names:
        filtered_long_df = filtered_long_df[filtered_long_df["이름"].astype(str).isin(selected_names)]

    if selected_depts:
        filtered_long_df = filtered_long_df[filtered_long_df["부서"].astype(str).isin(selected_depts)]

    if selected_products:
        filtered_long_df = filtered_long_df[filtered_long_df["품목명"].astype(str).isin(selected_products)]

    if keyword.strip():
        filtered_long_df = filtered_long_df[build_keyword_mask(filtered_long_df, keyword)]

    if issue_only:
        filtered_long_df = filtered_long_df[build_issue_mask(filtered_long_df)]

    # 선택 품목 기준 총생산수량 재계산
    if selected_products:
        group_cols = [c for c in get_fixed_cols(filtered_long_df) if c in filtered_long_df.columns]
        if "업무분기" in filtered_long_df.columns:
            group_cols = group_cols + ["업무분기"]

        selected_sum_df = (
            filtered_long_df.groupby(group_cols, dropna=False)["수량"]
            .sum()
            .reset_index()
        )

        filtered_df = filtered_df.drop(columns=["총생산수량"], errors="ignore")
        filtered_df = filtered_df.merge(
            selected_sum_df,
            on=group_cols,
            how="left"
        )
        filtered_df.rename(columns={"수량": "총생산수량"}, inplace=True)
        filtered_df["총생산수량"] = filtered_df["총생산수량"].fillna(0)

    return filtered_df, filtered_long_df


def make_download_file(df: pd.DataFrame) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    output.seek(0)
    return output.getvalue()


def compute_basic_kpis(filtered_df: pd.DataFrame, filtered_long_df: pd.DataFrame):
    if filtered_df.empty:
        return {
            "총 생산수량": 0,
            "작업일수": 0,
            "투입 인원수": 0,
            "1인당 일평균 생산량": 0,
            "worker_days": 0,
        }

    total_qty = int(filtered_long_df["수량"].sum()) if not filtered_long_df.empty else 0
    work_days = filtered_df["업무일자"].nunique() if "업무일자" in filtered_df.columns else 0
    active_people = filtered_df["이름"].dropna().astype(str).nunique() if "이름" in filtered_df.columns else 0

    if "업무일자" in filtered_df.columns and "이름" in filtered_df.columns:
        worker_days = filtered_df.groupby("업무일자")["이름"].nunique().sum()
    else:
        worker_days = 0

    per_person_day_avg = round(total_qty / worker_days, 2) if worker_days > 0 else 0

    return {
        "총 생산수량": total_qty,
        "작업일수": work_days,
        "투입 인원수": active_people,
        "1인당 일평균 생산량": per_person_day_avg,
        "worker_days": worker_days,
    }


def compute_quarter_staffing_kpis(filtered_df: pd.DataFrame, filtered_long_df: pd.DataFrame):
    """
    방법 B:
    추가 필요 인원 = 전분기 대비 증가분 / 전분기 1인당 분기 처리량
    """
    empty_result = {
        "quarter_summary": pd.DataFrame(),
        "current_quarter": None,
        "prev_quarter": None,
        "current_qty": 0,
        "prev_qty": 0,
        "increase_qty": 0,
        "growth_rate": 0,
        "prev_avg_headcount": 0,
        "prev_per_person_quarter_qty": 0,
        "extra_headcount_needed": 0,
        "current_avg_headcount": 0,
    }

    if filtered_df.empty or filtered_long_df.empty:
        return empty_result

    if "업무분기" not in filtered_df.columns or "업무분기" not in filtered_long_df.columns:
        return empty_result

    quarter_qty = (
        filtered_long_df.groupby("업무분기", dropna=False)["수량"]
        .sum()
        .reset_index()
        .rename(columns={"수량": "분기총생산량"})
    )

    daily_headcount = (
        filtered_df.groupby(["업무분기", "업무일자"], dropna=False)["이름"]
        .nunique()
        .reset_index()
        .rename(columns={"이름": "일투입인원"})
    )

    quarter_headcount = (
        daily_headcount.groupby("업무분기", dropna=False)["일투입인원"]
        .mean()
        .reset_index()
        .rename(columns={"일투입인원": "평균투입인원"})
    )

    quarter_summary = quarter_qty.merge(quarter_headcount, on="업무분기", how="left")
    quarter_summary["평균투입인원"] = quarter_summary["평균투입인원"].fillna(0)
    quarter_summary["1인당분기처리량"] = quarter_summary.apply(
        lambda x: round(x["분기총생산량"] / x["평균투입인원"], 2) if x["평균투입인원"] > 0 else 0,
        axis=1
    )

    quarter_summary["정렬키"] = quarter_summary["업무분기"].apply(
        lambda x: pd.Period(x, freq="Q") if pd.notna(x) and str(x) != "nan" else pd.NaT
    )
    quarter_summary = quarter_summary.sort_values("정렬키").reset_index(drop=True)

    if len(quarter_summary) < 2:
        quarter_summary = quarter_summary.drop(columns=["정렬키"], errors="ignore")
        return {
            **empty_result,
            "quarter_summary": quarter_summary
        }

    prev_row = quarter_summary.iloc[-2]
    curr_row = quarter_summary.iloc[-1]

    prev_qty = float(prev_row["분기총생산량"])
    current_qty = float(curr_row["분기총생산량"])
    increase_qty = current_qty - prev_qty
    growth_rate = round((increase_qty / prev_qty) * 100, 2) if prev_qty > 0 else 0

    prev_avg_headcount = float(prev_row["평균투입인원"])
    current_avg_headcount = float(curr_row["평균투입인원"])
    prev_per_person_quarter_qty = float(prev_row["1인당분기처리량"])

    if increase_qty > 0 and prev_per_person_quarter_qty > 0:
        extra_headcount_needed = int(np.ceil(increase_qty / prev_per_person_quarter_qty))
    else:
        extra_headcount_needed = 0

    quarter_summary = quarter_summary.drop(columns=["정렬키"], errors="ignore")

    return {
        "quarter_summary": quarter_summary,
        "current_quarter": curr_row["업무분기"],
        "prev_quarter": prev_row["업무분기"],
        "current_qty": int(current_qty),
        "prev_qty": int(prev_qty),
        "increase_qty": int(increase_qty),
        "growth_rate": growth_rate,
        "prev_avg_headcount": round(prev_avg_headcount, 2),
        "prev_per_person_quarter_qty": round(prev_per_person_quarter_qty, 2),
        "extra_headcount_needed": extra_headcount_needed,
        "current_avg_headcount": round(current_avg_headcount, 2),
    }


# =========================
# UI
# =========================
st.title("📊 업무일지 / 생산 대시보드")
st.caption("GitHub의 worklogs.xlsx를 자동으로 읽습니다. 고정 컬럼 외 나머지 컬럼은 모두 자동으로 생산 품목으로 인식합니다.")

with st.sidebar:
    st.header("데이터 불러오기 방식")
    github_url = st.text_input("GitHub Raw Excel URL", value=GITHUB_XLSX_URL)
    st.caption("반드시 raw.githubusercontent.com 주소를 넣어야 합니다.")
    col_a, col_b = st.columns(2)
    with col_a:
        refresh_clicked = st.button("새로고침")
    with col_b:
        show_manual_upload = st.checkbox("수동 업로드 사용", value=False)

uploaded_file = None

if refresh_clicked:
    fetch_github_excel.clear()
    load_excel.clear()
    st.cache_data.clear()

if show_manual_upload:
    uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx", "xls"])
    if uploaded_file is None:
        st.info("엑셀 파일을 업로드하면 대시보드가 표시됩니다.")
        st.stop()
    file_bytes = uploaded_file.read()
    data_source_label = f"수동 업로드 파일: {uploaded_file.name}"
else:
    if "raw.githubusercontent.com" not in github_url:
        st.error("GitHub 일반 주소가 아니라 raw 주소를 넣어야 합니다.")
        st.stop()

    try:
        file_bytes = fetch_github_excel(github_url)
        data_source_label = "GitHub worklogs.xlsx"
    except Exception as e:
        st.error(f"GitHub에서 엑셀을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()

try:
    df, long_df, product_cols = load_excel(file_bytes)
except Exception as e:
    st.error(f"엑셀 파싱 중 오류가 발생했습니다: {e}")
    st.stop()

if df.empty:
    st.warning("데이터가 비어 있습니다.")
    st.stop()

st.success(f"데이터 원본: {data_source_label} / 불러온 데이터: {len(df):,}건 / 자동 인식된 품목 수: {len(product_cols):,}개")
st.caption("GitHub 파일 변경 후 최대 60초 이내에 자동 반영됩니다. 바로 반영이 필요하면 '새로고침' 버튼을 누르세요.")

with st.expander("자동 인식된 품목 컬럼 보기"):
    if product_cols:
        st.write(product_cols)
    else:
        st.warning("인식된 품목 컬럼이 없습니다.")

filtered_df, filtered_long_df = apply_filters(df, long_df)

# =========================
# KPI
# =========================
basic_kpis = compute_basic_kpis(filtered_df, filtered_long_df)
quarter_kpis = compute_quarter_staffing_kpis(filtered_df, filtered_long_df)
issue_count = int(build_issue_mask(filtered_df).sum()) if not filtered_df.empty else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("총 생산수량", f"{basic_kpis['총 생산수량']:,}")
c2.metric("작업일수", f"{basic_kpis['작업일수']:,}")
c3.metric("투입 인원수", f"{basic_kpis['투입 인원수']:,}")
c4.metric("1인당 일평균 생산량", f"{basic_kpis['1인당 일평균 생산량']:,}")

if quarter_kpis["prev_quarter"] and quarter_kpis["current_quarter"]:
    c5.metric(
        "전분기 대비 증감률",
        f"{quarter_kpis['growth_rate']:,}%"
    )
    c6.metric(
        "추가 필요 인원",
        f"{quarter_kpis['extra_headcount_needed']:,}명"
    )
else:
    c5.metric("전분기 대비 증감률", "-")
    c6.metric("추가 필요 인원", "-")

st.caption(
    f"인원-일수(worker-days): {basic_kpis['worker_days']} / 이슈 건수: {issue_count}건"
)

# =========================
# 탭
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "생산현황", "인원판단", "업무일지", "이슈관리", "원본데이터"
])

# =========================
# 1) Overview
# =========================
with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("날짜별 총 생산수량")
        if not filtered_long_df.empty and "업무일자" in filtered_long_df.columns:
            trend = (
                filtered_long_df.groupby("업무일자", dropna=False)["수량"]
                .sum()
                .reset_index()
                .sort_values("업무일자")
            )
            st.line_chart(trend.set_index("업무일자"))
        else:
            st.info("표시할 생산 데이터가 없습니다.")

    with right:
        st.subheader("품목별 총 생산수량 TOP 10")
        if not filtered_long_df.empty:
            product_sum = (
                filtered_long_df.groupby("품목명", dropna=False)["수량"]
                .sum()
                .reset_index()
                .sort_values("수량", ascending=False)
                .head(10)
            )
            st.bar_chart(product_sum.set_index("품목명"))
        else:
            st.info("표시할 품목 데이터가 없습니다.")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("작성자별 생산량 TOP 10")
        if not filtered_long_df.empty and "이름" in filtered_long_df.columns:
            by_person = (
                filtered_long_df.groupby("이름", dropna=False)["수량"]
                .sum()
                .reset_index()
                .sort_values("수량", ascending=False)
                .head(10)
            )
            st.dataframe(by_person, use_container_width=True, hide_index=True)
        else:
            st.info("표시할 데이터가 없습니다.")

    with col2:
        st.subheader("부서별 생산량")
        if not filtered_long_df.empty and "부서" in filtered_long_df.columns:
            by_dept = (
                filtered_long_df.groupby("부서", dropna=False)["수량"]
                .sum()
                .reset_index()
                .sort_values("수량", ascending=False)
            )
            st.dataframe(by_dept, use_container_width=True, hide_index=True)
        else:
            st.info("표시할 데이터가 없습니다.")


# =========================
# 2) 생산현황
# =========================
with tab2:
    st.subheader("품목별 생산량 상세")

    if filtered_long_df.empty:
        st.info("조건에 맞는 생산 데이터가 없습니다.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            product_daily = (
                filtered_long_df.groupby(["업무일자", "품목명"], dropna=False)["수량"]
                .sum()
                .reset_index()
                .sort_values(["업무일자", "수량"], ascending=[True, False])
            )
            st.write("**날짜 × 품목 생산량**")
            st.dataframe(product_daily, use_container_width=True, hide_index=True, height=450)

        with col2:
            person_product = (
                filtered_long_df.groupby(["이름", "품목명"], dropna=False)["수량"]
                .sum()
                .reset_index()
                .sort_values("수량", ascending=False)
            )
            st.write("**작성자 × 품목 생산량**")
            st.dataframe(person_product, use_container_width=True, hide_index=True, height=450)

        st.markdown("---")

        st.write("**품목별 총 생산량**")
        product_sum_full = (
            filtered_long_df.groupby("품목명", dropna=False)["수량"]
            .sum()
            .reset_index()
            .sort_values("수량", ascending=False)
        )
        st.dataframe(product_sum_full, use_container_width=True, hide_index=True)

        download_bytes = make_download_file(product_sum_full)
        st.download_button(
            "품목별 집계 다운로드",
            data=download_bytes,
            file_name="품목별_집계.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# =========================
# 3) 인원판단
# =========================
with tab3:
    st.subheader("인원 충원 판단 (방법 B: 증가분 ÷ 전분기 1인당 분기 처리량)")

    quarter_summary = quarter_kpis["quarter_summary"]

    if quarter_summary.empty:
        st.info("분기 비교를 위한 데이터가 부족합니다.")
    else:
        st.write("**분기별 생산량 / 평균투입인원 / 1인당 분기 처리량**")
        st.dataframe(
            quarter_summary.sort_values("업무분기", key=lambda s: s.map(lambda x: pd.Period(x, freq="Q"))),
            use_container_width=True,
            hide_index=True,
            height=350
        )

        st.markdown("---")

        if quarter_kpis["prev_quarter"] and quarter_kpis["current_quarter"]:
            col1, col2 = st.columns(2)

            with col1:
                compare_df = pd.DataFrame({
                    "구분": ["전분기", "현재분기", "증가분", "증감률"],
                    "값": [
                        f"{quarter_kpis['prev_quarter']} / {quarter_kpis['prev_qty']:,}개",
                        f"{quarter_kpis['current_quarter']} / {quarter_kpis['current_qty']:,}개",
                        f"{quarter_kpis['increase_qty']:,}개",
                        f"{quarter_kpis['growth_rate']:,}%"
                    ]
                })
                st.write("**분기 비교**")
                st.dataframe(compare_df, use_container_width=True, hide_index=True)

            with col2:
                staffing_basis_df = pd.DataFrame({
                    "구분": ["전분기 평균투입인원", "전분기 1인당 분기 처리량", "추가 필요 인원"],
                    "값": [
                        quarter_kpis["prev_avg_headcount"],
                        quarter_kpis["prev_per_person_quarter_qty"],
                        quarter_kpis["extra_headcount_needed"]
                    ]
                })
                st.write("**충원 산정 근거**")
                st.dataframe(staffing_basis_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.write("**판단 가이드**")

            increase_qty = quarter_kpis["increase_qty"]
            extra_headcount_needed = quarter_kpis["extra_headcount_needed"]

            if increase_qty <= 0:
                st.info(
                    f"{quarter_kpis['prev_quarter']} 대비 {quarter_kpis['current_quarter']} 생산량이 증가하지 않아 추가 충원 필요 인원은 0명으로 계산됩니다."
                )
            elif quarter_kpis["prev_per_person_quarter_qty"] <= 0:
                st.warning("전분기 1인당 분기 처리량을 계산할 수 없어 추가 필요 인원을 산정하지 못했습니다.")
            else:
                st.warning(
                    f"{quarter_kpis['prev_quarter']} 대비 {quarter_kpis['current_quarter']} 생산량이 "
                    f"{increase_qty:,}개 증가했습니다. "
                    f"전분기 1인당 분기 처리량 {quarter_kpis['prev_per_person_quarter_qty']:,}개 기준으로 "
                    f"약 {extra_headcount_needed}명 추가 충원 검토가 필요합니다."
                )

            formula_df = pd.DataFrame({
                "항목": ["증가분", "전분기 1인당 분기 처리량", "추가 필요 인원 계산식"],
                "내용": [
                    f"{quarter_kpis['current_qty']:,} - {quarter_kpis['prev_qty']:,} = {quarter_kpis['increase_qty']:,}",
                    f"{quarter_kpis['prev_qty']:,} ÷ {quarter_kpis['prev_avg_headcount']:,}" if quarter_kpis["prev_avg_headcount"] > 0 else "-",
                    f"{quarter_kpis['increase_qty']:,} ÷ {quarter_kpis['prev_per_person_quarter_qty']:,} → 올림 = {quarter_kpis['extra_headcount_needed']}명" if quarter_kpis["prev_per_person_quarter_qty"] > 0 else "-"
                ]
            })
            st.write("**계산식 요약**")
            st.dataframe(formula_df, use_container_width=True, hide_index=True)

        else:
            st.info("최소 2개 분기의 데이터가 있어야 충원 판단이 가능합니다.")


# =========================
# 4) 업무일지
# =========================
with tab4:
    st.subheader("업무일지 상세")

    display_cols = [
        c for c in [
            "업무일자", "업무분기", "이름", "부서", "작성일시",
            "입고수량", "출고수량", "반품수량",
            "총생산수량",
            "재고확인사항", "배송이슈",
            "오전 업무내용", "오후 업무내용",
            "미처리내역", "예정사항", "특이사항", "Comment"
        ] if c in filtered_df.columns
    ]

    worklog_view = filtered_df[display_cols].copy()

    if "작성일시" in worklog_view.columns:
        worklog_view = worklog_view.sort_values("작성일시", ascending=False, na_position="last")
    elif "업무일자" in worklog_view.columns:
        worklog_view = worklog_view.sort_values("업무일자", ascending=False, na_position="last")

    st.dataframe(worklog_view, use_container_width=True, hide_index=True, height=600)

    download_bytes = make_download_file(worklog_view)
    st.download_button(
        "업무일지 다운로드",
        data=download_bytes,
        file_name="업무일지_필터결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =========================
# 5) 이슈관리
# =========================
with tab5:
    st.subheader("이슈 관리")

    issues = filtered_df.copy()
    issues = issues[build_issue_mask(issues)].copy()

    issue_cols = [
        c for c in [
            "업무일자", "업무분기", "이름", "부서",
            "재고확인사항", "배송이슈", "미처리내역",
            "특이사항", "Comment",
            "오전 업무내용", "오후 업무내용",
            "총생산수량"
        ] if c in issues.columns
    ]

    if issues.empty:
        st.info("등록된 이슈가 없습니다.")
    else:
        if "업무일자" in issues.columns:
            issues = issues.sort_values("업무일자", ascending=False, na_position="last")

        st.dataframe(
            issues[issue_cols],
            use_container_width=True,
            hide_index=True,
            height=600
        )

        multi_issue_count = 0
        issue_check_cols = [c for c in ["재고확인사항", "배송이슈", "미처리내역", "특이사항", "Comment"] if c in issues.columns]
        if issue_check_cols:
            multi_issue_count = int(
                issues[issue_check_cols]
                .fillna("")
                .astype(str)
                .apply(lambda row: sum(x.strip() != "" for x in row), axis=1)
                .ge(2)
                .sum()
            )

        issue_summary = pd.DataFrame({
            "구분": ["재고확인사항", "배송이슈", "미처리내역", "특이사항", "Comment", "2개 이상 이슈"],
            "건수": [
                int(issues["재고확인사항"].fillna("").astype(str).str.strip().ne("").sum()) if "재고확인사항" in issues.columns else 0,
                int(issues["배송이슈"].fillna("").astype(str).str.strip().ne("").sum()) if "배송이슈" in issues.columns else 0,
                int(issues["미처리내역"].fillna("").astype(str).str.strip().ne("").sum()) if "미처리내역" in issues.columns else 0,
                int(issues["특이사항"].fillna("").astype(str).str.strip().ne("").sum()) if "특이사항" in issues.columns else 0,
                int(issues["Comment"].fillna("").astype(str).str.strip().ne("").sum()) if "Comment" in issues.columns else 0,
                multi_issue_count
            ]
        })

        st.write("**이슈 요약**")
        st.dataframe(issue_summary, use_container_width=True, hide_index=True)


# =========================
# 6) 원본데이터
# =========================
with tab6:
    st.subheader("원본 데이터")

    st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=600)

    st.markdown("---")
    st.write("**자동 인식 품목 컬럼**")
    st.write(product_cols if product_cols else "없음")

    raw_download = make_download_file(filtered_df)
    st.download_button(
        "원본 데이터 다운로드",
        data=raw_download,
        file_name="원본데이터_필터결과.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
