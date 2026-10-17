# OncoFin — онкологиялық скрининг экономикалық терминалы
# VS Code:  streamlit run oncofin_app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="OncoFin — Экономикалық терминал",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- CSS ----------
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Outfit:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Outfit', system-ui, sans-serif; }
  .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; max-width: 1100px; }
  h1, h2, h3, .display { font-family: 'Fraunces', Georgia, serif !important; }
  #MainMenu, footer, header { visibility: hidden; }
  div[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 700; }
  .hero {
    background: linear-gradient(135deg, #0f766e 0%, #0e7490 55%, #0369a1 100%);
    border-radius: 16px; padding: 1.4rem 1.5rem; color: #fff; margin-bottom: 1rem;
  }
  .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; }
  .hero p { margin: 0.4rem 0 0; opacity: 0.95; font-size: 1rem; line-height: 1.45; }
  .step-box {
    background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 12px;
    padding: 0.85rem 1rem; height: 100%;
  }
  .step-n {
    display: inline-flex; width: 1.6rem; height: 1.6rem; align-items: center; justify-content: center;
    background: #0f766e; color: #fff; border-radius: 8px; font-weight: 700; font-size: 0.85rem; margin-right: 0.4rem;
  }
  .bar-box {
    background: linear-gradient(135deg, #fef2f2, #fff7ed);
    border-left: 4px solid #e11d48; border-radius: 0 12px 12px 0;
    padding: 1rem 1.15rem; margin: 0.75rem 0;
  }
  .ok-box {
    background: linear-gradient(135deg, #ecfdf5, #f0fdf4);
    border-left: 4px solid #10b981; border-radius: 0 12px 12px 0;
    padding: 1rem 1.15rem; margin: 0.75rem 0;
  }
  .info-box {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1rem 1.15rem; margin: 0.5rem 0;
  }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- MODEL ----------
CAPEX0 = 20_500_000
OPEX0 = 7_600_000
COST_EARLY = 800_000
COST_LATE = 5_500_000
SAVE = COST_LATE - COST_EARLY  # 4_700_000


def npv_calc(cf: float, rate: float, years: int, capex: float) -> float:
    total = -capex
    for t in range(1, years + 1):
        total += cf / ((1 + rate) ** t)
    return total


def monte_carlo(cases: float, case_std: float, opex: float, n: int = 10_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    c = np.clip(rng.normal(cases, case_std, n), 3, 55)
    s = np.clip(rng.normal(SAVE, 300_000, n), 3_200_000, 6_500_000)
    net = c * s - opex
    net.sort()
    return {
        "net": net,
        "p5": float(np.percentile(net, 5)),
        "p1": float(np.percentile(net, 1)),
        "median": float(np.median(net)),
        "mean": float(np.mean(net)),
        "positive": float((net > 0).mean()),
    }


def ml_forecast(opex: float):
    years_h = np.array([2021, 2022, 2023, 2024, 2025]).reshape(-1, 1)
    cases_h = np.array([9, 12, 15, 18, 20])
    model = make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0))
    model.fit(years_h, cases_h)
    future = np.arange(2026, 2031).reshape(-1, 1)
    pred = np.clip(model.predict(future), 5, 50).round().astype(int)
    r2 = r2_score(cases_h, model.predict(years_h))
    rows = []
    for y, p in zip(future.flatten(), pred):
        rows.append(
            {
                "Жыл": int(y),
                "Ерте анықтау": int(p),
                "Benefit (₸)": int(p * SAVE),
                "Таза CF (₸)": int(p * SAVE - opex),
            }
        )
    return cases_h, years_h.flatten(), pred, future.flatten(), r2, pd.DataFrame(rows)


def mln(x: float, d: int = 1) -> str:
    if not np.isfinite(x):
        return "—"
    return f"{x / 1e6:,.{d}f}".replace(",", " ").replace(".", ",")


# ---------- HEADER ----------
st.markdown(
    """
<div class="hero">
  <h1>🏥 OncoFin</h1>
  <p>Скринингті ерте өткізсек, мемлекет кеш сатыдағы қымбат емге қанша аз ақша жұмсайды.
  Бұл калькулятор сол үнемді, өтелу мерзімін және тәуекелді көрсетеді.</p>
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="step-box"><span class="step-n">1</span><b>Сценарий</b><br/>Нашар / База / Жақсы таңдаңыз</div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="step-box"><span class="step-n">2</span><b>Сандар</b><br/>Үнем, NPV, өтелу — жоғарыда</div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="step-box"><span class="step-n">3</span><b>Тәуекел</b><br/>Monte-Carlo · BaR 95%</div>',
        unsafe_allow_html=True,
    )

st.write("")

# ---------- CONTROLS ----------
left, right = st.columns([1.1, 1.4])
with left:
    scenario = st.radio(
        "Сценарий",
        ["Базалық", "Пессимистік", "Оптимистік", "Қолмен"],
        horizontal=True,
        index=0,
    )
with right:
    years = st.slider("Болжам мерзімі (жыл)", 1, 5, 3)
    discount = st.slider("Дисконт (%)", 5.0, 20.0, 10.0, 0.5) / 100

capex = CAPEX0
opex = OPEX0
cases = 20

if scenario == "Пессимистік":
    cases, opex = 5, int(OPEX0 * 1.20)
elif scenario == "Оптимистік":
    cases = 35
elif scenario == "Қолмен":
    a, b, c = st.columns(3)
    with a:
        capex = int(st.number_input("CapEx (млн ₸)", value=20.5, step=0.5) * 1_000_000)
    with b:
        opex = int(st.number_input("OpEx (млн ₸)", value=7.6, step=0.1) * 1_000_000)
    with c:
        cases = int(st.number_input("Ерте анықтау (адам/жыл)", value=20, min_value=3, max_value=60, step=1))

benefit = cases * SAVE
cf = benefit - opex
payback = (capex / cf * 12) if cf > 0 else float("inf")
npv = npv_calc(cf, discount, years, capex)
icer = capex / cases if cases else 0
ratio = COST_LATE / icer if icer else 0

# ---------- KPI ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Жылдық үнем", f"{mln(benefit)} млн ₸", f"{cases} адам")
k2.metric("Таза ағын", f"{mln(cf)} млн ₸", "Benefit − OpEx")
k3.metric(f"NPV · {years} жыл", f"{mln(npv)} млн ₸", "оң" if npv > 0 else "теріс")
k4.metric("Өтелу", f"{payback:.1f} ай" if payback < 200 else "—", f"ICER {mln(icer, 2)} млн")

st.markdown("---")

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Талдау", "🎲 Тәуекел (Monte-Carlo)", "🤖 Болжам", "🔀 Сценарийлер", "ℹ️ Әдістеме"]
)

# ===== TAB 1 =====
with tab1:
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.subheader("NPV құрылымы")
        disc_vals = [-capex]
        labels = ["CapEx"]
        for t in range(1, years + 1):
            disc_vals.append(cf / ((1 + discount) ** t))
            labels.append(f"Жыл {t}")
        disc_vals.append(npv)
        labels.append("NPV")
        colors = ["#e11d48"] + ["#10b981"] * years + ["#0ea5e9"]
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=[v / 1e6 for v in disc_vals],
                marker_color=colors,
                text=[f"{v/1e6:.1f}" for v in disc_vals],
                textposition="outside",
            )
        )
        fig.update_layout(
            yaxis_title="млн ₸",
            height=340,
            margin=dict(t=20, b=30),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("Бір абзацта")
        st.markdown(
            f"""
<div class="info-box">
Кеш сатыны емдеу <b>{mln(COST_LATE, 1)} млн</b>, ерте саты <b>{mln(COST_EARLY, 1)} млн</b>.
Айырма — <b>{mln(SAVE)} млн ₸</b> бір адамға.<br/><br/>
{cases} адам ерте табылса, жылына <b>{mln(benefit)} млн</b> үнем.
Жүйе жылына {mln(opex)} млн тұрады → қалтада <b>{mln(cf)} млн</b>.<br/><br/>
Бастапқы {mln(capex)} млн салым
{"{:.1f} айда".format(payback) if payback < 200 else "өтелмейді"},
{years} жылда NPV = <b>{mln(npv)} млн ₸</b>.
Бір адамды ерте табу кеш емдеуден <b>{ratio:.1f}×</b> арзан.
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="ok-box">
<b>Кабинет:</b> 15 → 22 пациент/күн (+46%). Бос тұру −68%.
Бір тексеру 12 500 → 8 523 ₸.
</div>
""",
            unsafe_allow_html=True,
        )

# ===== TAB 2 MONTE CARLO =====
with tab2:
    st.subheader("Monte-Carlo · Budget-at-Risk")
    st.caption("Науқас саны мен үнем қалыпты заңмен ауытқиды · 10 000 итерация · 95% сенімділік")

    s1, s2 = st.columns(2)
    with s1:
        case_std = st.slider("Науқас санының ауытқуы (σ)", 1.0, 10.0, 4.0, 0.5)
    with s2:
        st.write("")
        st.write(f"Орташа N = **{cases}** адам · σ = **{case_std:.1f}**")

    risk = monte_carlo(cases, case_std, opex, n=10_000)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("BaR 95%", f"{mln(risk['p5'])} млн ₸")
    r2.metric("Медиана", f"{mln(risk['median'])} млн ₸")
    r3.metric("Орташа", f"{mln(risk['mean'])} млн ₸")
    r4.metric("P(плюс)", f"{risk['positive']*100:.0f}%")

    st.markdown(
        f"""
<div class="bar-box">
<b>Budget-at-Risk (95%):</b> 100 жағдайдың 95-інде бюджет жылына кемінде
<b>{mln(risk['p5'])} млн ₸</b> үнемдейді — тіпті науқас саны мен ем бағасы ауытқыса да.<br/>
Ең нашар 1% сценарий: <b>{mln(risk['p1'])} млн ₸</b>.
Оң нәтиже ықтималдығы: <b>{risk['positive']*100:.0f}%</b>.
</div>
""",
        unsafe_allow_html=True,
    )

    fig_mc = go.Figure()
    fig_mc.add_trace(
        go.Histogram(
            x=risk["net"] / 1e6,
            nbinsx=50,
            marker_color="#10b981",
            opacity=0.85,
            name="Симуляция",
        )
    )
    fig_mc.add_vline(
        x=risk["p5"] / 1e6,
        line_dash="dash",
        line_color="#e11d48",
        line_width=2.5,
        annotation_text=f"BaR 95%: {mln(risk['p5'])} млн",
        annotation_position="top right",
    )
    fig_mc.add_vline(
        x=risk["median"] / 1e6,
        line_dash="dot",
        line_color="#0ea5e9",
        line_width=2,
        annotation_text="Медиана",
        annotation_position="top left",
    )
    fig_mc.update_layout(
        title="Таза жылдық ақша ағынының үлестірімі (10 000 сынақ)",
        xaxis_title="Таза үнем (млн ₸)",
        yaxis_title="Жиілік",
        height=400,
        showlegend=False,
        bargap=0.03,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    with st.expander("Толық статистика"):
        st.dataframe(
            pd.DataFrame(
                {
                    "Көрсеткіш": [
                        "Орташа",
                        "Медиана",
                        "σ",
                        "BaR 95% (5%)",
                        "BaR 99% (1%)",
                        "Min",
                        "Max",
                        "P(CF>0)",
                    ],
                    "Мәні": [
                        f"{mln(risk['mean'])} млн",
                        f"{mln(risk['median'])} млн",
                        f"{mln(float(np.std(risk['net'])))} млн",
                        f"{mln(risk['p5'])} млн",
                        f"{mln(risk['p1'])} млн",
                        f"{mln(float(risk['net'][0]))} млн",
                        f"{mln(float(risk['net'][-1]))} млн",
                        f"{risk['positive']*100:.1f}%",
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

# ===== TAB 3 ML =====
with tab3:
    st.subheader("Машиналық оқыту · 5 жылдық болжам")
    hist_c, hist_y, pred, fut_y, r2, df_ml = ml_forecast(opex)

    fig_ml = go.Figure()
    fig_ml.add_trace(
        go.Scatter(
            x=hist_y,
            y=hist_c,
            mode="lines+markers",
            name="Тарихи",
            line=dict(color="#0ea5e9", width=3),
            marker=dict(size=10),
        )
    )
    fig_ml.add_trace(
        go.Scatter(
            x=fut_y,
            y=pred,
            mode="lines+markers",
            name="ML болжам",
            line=dict(color="#e11d48", width=3, dash="dash"),
            marker=dict(size=10, symbol="diamond"),
        )
    )
    fig_ml.update_layout(
        title="Ерте анықтау саны: тарихи + болжам",
        xaxis_title="Жыл",
        yaxis_title="Адам",
        height=380,
        legend=dict(orientation="h", y=1.12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_ml, use_container_width=True)

    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("R² (модель сапасы)", f"{r2:.3f}")
    with m2:
        st.dataframe(
            df_ml.style.format({"Benefit (₸)": "{:,.0f}", "Таза CF (₸)": "{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
    st.info(
        "ML модель 2021–2025 үрдісіне квадраттық регрессия. Нақты диспансер деректері қосылса дәлдік артады."
    )

# ===== TAB 4 SCENARIOS =====
with tab4:
    st.subheader("Үш сценарий · 10% дисконт · 3 жыл")
    rows = []
    for name, n_case, o_mult in [
        ("Пессимистік", 5, 1.20),
        ("Базалық", 20, 1.00),
        ("Оптимистік", 35, 1.00),
    ]:
        o = int(OPEX0 * o_mult)
        net = n_case * SAVE - o
        n3 = npv_calc(net, 0.10, 3, CAPEX0)
        pb = (CAPEX0 / net * 12) if net > 0 else float("inf")
        rows.append(
            {
                "Сценарий": name,
                "Адам/жыл": n_case,
                "Таза CF (млн)": round(net / 1e6, 1),
                "NPV 3ж (млн)": round(n3 / 1e6, 1),
                "Өтелу (ай)": round(pb, 1) if pb < 200 else "—",
            }
        )
    df_s = pd.DataFrame(rows)
    st.dataframe(df_s, use_container_width=True, hide_index=True)

    fig_s = go.Figure(
        go.Bar(
            x=df_s["Сценарий"],
            y=df_s["NPV 3ж (млн)"],
            marker_color=["#e11d48", "#0ea5e9", "#10b981"],
            text=[f"{v:.0f} млн" for v in df_s["NPV 3ж (млн)"]],
            textposition="outside",
        )
    )
    fig_s.update_layout(
        yaxis_title="NPV (млн ₸)",
        height=360,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_s, use_container_width=True)
    st.markdown(
        '<div class="ok-box">Тіпті <b>пессимистік</b> нұсқада да NPV оң — жоба тәуекелге төзімді.</div>',
        unsafe_allow_html=True,
    )

# ===== TAB 5 METHOD =====
with tab5:
    st.subheader("Әдістеме және жоба")
    st.markdown(
        f"""
**Формулалар**

- Үнем бір адамға: $S = 5{','}500{','}000 - 800{','}000 = 4{','}700{','}000$ ₸ (ҚР ДСМ КШТ/КЗГ)
- Жылдық Benefit: $B = N \\times S$
- Таза ағын: $CF = B - OpEx$
- NPV: $NPV = -CapEx + \\sum_{{t=1}}^{{T}} CF / (1+r)^t$
- Budget-at-Risk 95%: Monte-Carlo үлестірімінің 5-процентилі

**Базалық параметрлер**

| Көрсеткіш | Мәні |
|-----------|------|
| CapEx | 20,5 млн ₸ |
| OpEx | 7,6 млн ₸/жыл |
| 1–2 / 3–4 стадия | 0,8 / 5,5 млн ₸ |
| Базалық N | 20 адам/жыл |
| Дисконт | 10% |
| 3 жылдық NPV | ≈ +194,4 млн ₸ |
| Өтелу | ≈ 2,8 ай |

**Авторлар:** Сарсенбиғалиқызы Зере, Назарбай Аружан · НЗМ Ақтау · Жетекші: Қонырбаева Ж.М.
"""
    )

st.markdown("---")
st.caption("OncoFin · Ақтау 2026 · Monte-Carlo Risk Engine · ML Forecast")
