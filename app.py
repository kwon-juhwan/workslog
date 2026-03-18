import streamlit as st

import pandas as pd

import numpy as np

from io import BytesIO



# =========================

# 기본 설정

# =========================

st.set_page_config(

    page_title="업무일지 / 생산 대시보드",

    page_icon="📊",

    layout="wide"

)



# 고정 컬럼: 이 컬럼을 제외한 나머지는 모두 '신규 품목 자동 인식'

FIXED_COLS = [

    "업무일자",

    "이름",

    "부서",

    "작성일시",

    "오전업무",

    "오후업무",

    "특이사항",

    "코멘트",

]



TEXT_COLS = ["오전업무", "오후업무", "특이사항", "코멘트"]

DATE_COLS = ["업무일자", "작성일시"]





# =========================

# 공통 함수

# =========================

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df.columns = [str(c).strip() for c in df.columns]

    return df





def ensure_fixed_cols(df: pd.DataFrame) -> pd.DataFrame:

    """

    고정 컬럼이 없더라도 앱이 깨지지 않게 빈 컬럼 생성

    """

    df = df.copy()

    for col in FIXED_COLS:

        if col not in df.columns:

            df[col] = np.nan

    return df





def parse_dates(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()



    if "업무일자" in df.columns:

        df["업무일자"] = pd.to_datetime(df["업무일자"], errors="coerce").dt.date



    if "작성일시" in df.columns:

        df["작성일시"] = pd.to_datetime(df["작성일시"], errors="coerce")



    return df





def detect_product_cols(df: pd.DataFrame) -> list:

    """

    고정 컬럼을 제외한 나머지를 모두 품목 컬럼으로 간주

    단, 완전히 빈 컬럼은 제외

    """

    product_cols = []

    for col in df.columns:

        if col in FIXED_COLS:

            continue

        # 전부 NaN인 컬럼은 제외

        if df[col].isna().all():

            continue

        product_cols.append(col)

    return product_cols





def preprocess_product_values(df: pd.DataFrame, product_cols: list) -> pd.DataFrame:

    """

    품목 컬럼은 숫자로 변환

    문자/공란/이상값은 0 처리

    """

    df = df.copy()

    for col in product_cols:

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df





def make_long_dataframe(df: pd.DataFrame, product_cols: list) -> pd.DataFrame:

    """

    wide -> long 변환

    품목명 / 수량 구조로 변환

    """

    if not product_cols:

        return pd.DataFrame(columns=FIXED_COLS + ["품목명", "수량"])



    long_df = df.melt(

        id_vars=[c for c in FIXED_COLS if c in df.columns],

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





@st.cache_data

def load_excel(file_bytes: bytes, sheet_name=0):

    df = pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)

    df = clean_columns(df)

    df = ensure_fixed_cols(df)

    df = parse_dates(df)



    product_cols = detect_product_cols(df)

    df = preprocess_product_values(df, product_cols)

    df = add_total_qty(df, product_cols)

    long_df = make_long_dataframe(df, product_cols)



    return df, long_df, product_cols





def apply_filters(df: pd.DataFrame, long_df: pd.DataFrame):

    st.sidebar.header("필터")



    # 날짜 범위

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



    # 이름

    names = sorted([x for x in df["이름"].dropna().astype(str).unique().tolist() if x.strip()])

    selected_names = st.sidebar.multiselect("이름", names, default=names)



    # 부서

    depts = sorted([x for x in df["부서"].dropna().astype(str).unique().tolist() if x.strip()])

    selected_depts = st.sidebar.multiselect("부서", depts, default=depts)



    # 품목

    products = sorted([x for x in long_df["품목명"].dropna().astype(str).unique().tolist() if x.strip()])

    selected_products = st.sidebar.multiselect("품목", products, default=products)



    # 키워드

    keyword = st.sidebar.text_input("업무/특이사항 키워드 검색", "")



    # 이슈만 보기

    issue_only = st.sidebar.checkbox("특이사항/코멘트 있는 건만 보기", value=False)



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

        keyword_mask = (

            safe_contains(filtered_df["오전업무"], keyword) |

            safe_contains(filtered_df["오후업무"], keyword) |

            safe_contains(filtered_df["특이사항"], keyword) |

            safe_contains(filtered_df["코멘트"], keyword)

        )

        filtered_df = filtered_df[keyword_mask]



    if issue_only:

        issue_mask = (

            filtered_df["특이사항"].fillna("").astype(str).str.strip().ne("") |

            filtered_df["코멘트"].fillna("").astype(str).str.strip().ne("")

        )

        filtered_df = filtered_df[issue_mask]



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

        keyword_mask_long = (

            safe_contains(filtered_long_df["오전업무"], keyword) |

            safe_contains(filtered_long_df["오후업무"], keyword) |

            safe_contains(filtered_long_df["특이사항"], keyword) |

            safe_contains(filtered_long_df["코멘트"], keyword)

        )

        filtered_long_df = filtered_long_df[keyword_mask_long]



    if issue_only:

        issue_mask_long = (

            filtered_long_df["특이사항"].fillna("").astype(str).str.strip().ne("") |

            filtered_long_df["코멘트"].fillna("").astype(str).str.strip().ne("")

        )

        filtered_long_df = filtered_long_df[issue_mask_long]



    # filtered_df의 총생산수량도 선택된 품목 기준으로 다시 계산

    if selected_products:

        pivot_selected = (

            filtered_long_df.groupby(filtered_long_df.index)["수량"].sum()

        )

        # long_df는 melt 후 index가 원본 행 인덱스를 유지하지 않으므로 재계산 방식 변경

        selected_sum_df = (

            filtered_long_df.groupby(

                ["업무일자", "이름", "부서", "작성일시", "오전업무", "오후업무", "특이사항", "코멘트"],

                dropna=False

            )["수량"].sum().reset_index()

        )



        filtered_df = filtered_df.drop(columns=["총생산수량"], errors="ignore")

        filtered_df = filtered_df.merge(

            selected_sum_df,

            on=["업무일자", "이름", "부서", "작성일시", "오전업무", "오후업무", "특이사항", "코멘트"],

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





# =========================

# UI

# =========================

st.title("📊 업무일지 / 생산 대시보드")

st.caption("고정 컬럼 외 나머지 컬럼은 모두 자동으로 생산 품목으로 인식합니다.")



uploaded_file = st.file_uploader("엑셀 파일 업로드", type=["xlsx", "xls"])



if uploaded_file is None:

    st.info("엑셀 파일을 업로드하면 대시보드가 표시됩니다.")

    st.stop()



try:

    file_bytes = uploaded_file.read()

    df, long_df, product_cols = load_excel(file_bytes)

except Exception as e:

    st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

    st.stop()



if df.empty:

    st.warning("데이터가 비어 있습니다.")

    st.stop()



st.success(f"불러온 데이터: {len(df):,}건 / 자동 인식된 품목 수: {len(product_cols):,}개")



with st.expander("자동 인식된 품목 컬럼 보기"):

    if product_cols:

        st.write(product_cols)

    else:

        st.warning("인식된 품목 컬럼이 없습니다.")



filtered_df, filtered_long_df = apply_filters(df, long_df)



# =========================

# KPI

# =========================

total_logs = len(filtered_df)

total_people = filtered_df["이름"].nunique() if "이름" in filtered_df.columns else 0

total_products = filtered_long_df["품목명"].nunique() if not filtered_long_df.empty else 0

total_qty = int(filtered_long_df["수량"].sum()) if not filtered_long_df.empty else 0

issue_count = 0

if "특이사항" in filtered_df.columns and "코멘트" in filtered_df.columns:

    issue_count = int(

        (

            filtered_df["특이사항"].fillna("").astype(str).str.strip().ne("") |

            filtered_df["코멘트"].fillna("").astype(str).str.strip().ne("")

        ).sum()

    )



c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("업무일지 건수", f"{total_logs:,}")

c2.metric("작성자 수", f"{total_people:,}")

c3.metric("품목 수", f"{total_products:,}")

c4.metric("총 생산수량", f"{total_qty:,}")

c5.metric("이슈 건수", f"{issue_count:,}")



# =========================

# 탭

# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "Overview", "생산현황", "업무일지", "이슈관리", "원본데이터"

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

# 3) 업무일지

# =========================

with tab3:

    st.subheader("업무일지 상세")



    display_cols = [

        c for c in ["업무일자", "이름", "부서", "작성일시", "총생산수량", "오전업무", "오후업무", "특이사항", "코멘트"]

        if c in filtered_df.columns

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

# 4) 이슈관리

# =========================

with tab4:

    st.subheader("특이사항 / 코멘트 관리")



    issues = filtered_df.copy()

    issue_mask = (

        issues["특이사항"].fillna("").astype(str).str.strip().ne("") |

        issues["코멘트"].fillna("").astype(str).str.strip().ne("")

    )

    issues = issues[issue_mask].copy()



    issue_cols = [c for c in ["업무일자", "이름", "부서", "특이사항", "코멘트", "오전업무", "오후업무", "총생산수량"] if c in issues.columns]



    if issues.empty:

        st.info("등록된 이슈가 없습니다.")

    else:

        st.dataframe(

            issues[issue_cols].sort_values("업무일자", ascending=False, na_position="last"),

            use_container_width=True,

            hide_index=True,

            height=600

        )



        issue_summary = pd.DataFrame({

            "구분": ["특이사항 있음", "코멘트 있음", "둘 다 있음"],

            "건수": [

                int(issues["특이사항"].fillna("").astype(str).str.strip().ne("").sum()),

                int(issues["코멘트"].fillna("").astype(str).str.strip().ne("").sum()),

                int((

                    issues["특이사항"].fillna("").astype(str).str.strip().ne("") &

                    issues["코멘트"].fillna("").astype(str).str.strip().ne("")

                ).sum())

            ]

        })

        st.write("**이슈 요약**")

        st.dataframe(issue_summary, use_container_width=True, hide_index=True)





# =========================

# 5) 원본데이터

# =========================

with tab5:

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
