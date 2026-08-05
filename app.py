# ==========================================================================
#  PAUTA DE TECHO CORTO (metodología José Luis Cava) — Streamlit
# ==========================================================================
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st

try:
    from ta.volatility import AverageTrueRange
    _TA_OK = True
except Exception:
    _TA_OK = False

# --------------------------------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------------------------------
st.set_page_config(page_title="Pauta de Techo Corto", page_icon="📉", layout="wide")

LEFT = RIGHT = 3
TOL_DOBLE_TECHO = 0.020
TOL_RESISTENCIA = 0.030
SEP_MIN_PICOS = 5
BARRAS_LATERAL = 10
FACTOR_ATR_LATERAL = 2.0
PROX_EMA = 0.015
MARGEN_STOP_ATR = 0.5

CSS = """
<style>
.card {border-radius:10px; padding:16px 18px; margin-bottom:10px; border:1px solid #2b3038;}
.card-ok {background:#0e2418; border-color:#1f6f45;}
.card-ko {background:#26130f; border-color:#7a2b21;}
.card h4 {margin:0 0 6px 0; font-size:15px;}
.card p {margin:0; font-size:12.5px; color:#b9c1cc; line-height:1.55;}
.badge {font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#8b93a1;}
.metric-box {border:1px solid #2b3038; border-radius:10px; padding:12px 14px; background:#12151a;}
.metric-box .lbl {font-size:11px; color:#8b93a1; text-transform:uppercase; letter-spacing:.06em;}
.metric-box .val {font-size:20px; font-weight:700; margin-top:2px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# DATOS Y CÁLCULOS
# --------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def descargar(ticker, intervalo):
    periodo = "2y" if intervalo == "1d" else "180d"
    df = yf.Ticker(ticker).history(period=periodo, interval=intervalo)
    if df is None or df.empty:
        return None
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols].dropna()
    df = df.tail(400 if intervalo == "1d" else 500)
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "Fecha"}, inplace=True)
    return df


def calcular_atr(df, n=14):
    if _TA_OK:
        try:
            return AverageTrueRange(high=df["High"], low=df["Low"],
                                    close=df["Close"], window=n).average_true_range()
        except Exception:
            pass
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(n).mean()


def encontrar_pivotes(serie, left=LEFT, right=RIGHT, tipo="max"):
    v = serie.values
    idx = []
    for i in range(left, len(v) - right):
        vent = v[i - left:i + right + 1]
        if tipo == "max" and v[i] == vent.max() and v[i] >= v[i - 1]:
            idx.append(i)
        elif tipo == "min" and v[i] == vent.min() and v[i] <= v[i - 1]:
            idx.append(i)
    return idx


def recta(x1, y1, x2, y2, x):
    if x2 == x1:
        return y1
    return y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)


def analizar(df):
    res = {"c1": False, "c2": False, "c3": False, "detalle": {}, "lineas": {}}

    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ATR"] = calcular_atr(df)

    n = len(df)
    cierre = float(df["Close"].iloc[-1])
    atr = df["ATR"].iloc[-1]
    atr = float(atr) if not pd.isna(atr) else cierre * 0.01

    piv_max = encontrar_pivotes(df["High"], tipo="max")
    piv_min = encontrar_pivotes(df["Low"], tipo="min")

    # ---- CIRCUNSTANCIA 2 : doble techo + valle + directriz ----
    picos_rec = [i for i in piv_max if i > n - 150]
    pico1 = pico2 = None
    for a in range(len(picos_rec) - 1, 0, -1):
        i2 = picos_rec[a]
        for b in range(a - 1, -1, -1):
            i1 = picos_rec[b]
            if i2 - i1 < SEP_MIN_PICOS:
                continue
            p1, p2 = df["High"].iloc[i1], df["High"].iloc[i2]
            if abs(p2 - p1) / p1 <= TOL_DOBLE_TECHO:
                pico1, pico2 = i1, i2
                break
        if pico1 is not None:
            break

    valle_roto = directriz_rota = False
    if pico1 is not None:
        tramo = df["Low"].iloc[pico1:pico2 + 1]
        valle_val = float(tramo.min())
        idx_valle = int(tramo.idxmin())
        post = df["Close"].iloc[pico2 + 1:]
        valle_roto = bool((post < valle_val).any()) if len(post) else False

        min_prev = [i for i in piv_min if i < pico2]
        if len(min_prev) >= 2:
            m1, m2 = min_prev[-2], min_prev[-1]
            y1, y2 = float(df["Low"].iloc[m1]), float(df["Low"].iloc[m2])
            if y2 > y1:
                for j in range(pico2 + 1, n):
                    if df["Close"].iloc[j] < recta(m1, y1, m2, y2, j):
                        directriz_rota = True
                        break
                res["lineas"]["directriz"] = (m1, y1, m2, y2)

        res["lineas"]["valle"] = (idx_valle, valle_val)
        res["lineas"]["picos"] = (pico1, float(df["High"].iloc[pico1]),
                                  pico2, float(df["High"].iloc[pico2]))

    res["c2"] = bool(pico1 is not None and valle_roto and directriz_rota)
    res["detalle"]["c2"] = (
        f"Doble techo: <b>{'sí' if pico1 is not None else 'no detectado'}</b> · "
        f"Valle perforado: <b>{'sí' if valle_roto else 'no'}</b> · "
        f"Directriz alcista perforada: <b>{'sí' if directriz_rota else 'no'}</b>")

    # ---- CIRCUNSTANCIA 1 : tendencia previa + resistencia ----
    ref = pico1 if pico1 is not None else max(0, n - 30)
    ini = max(0, ref - 60)
    tendencia = False
    if ref - ini > 10:
        tendencia = bool(
            df["Close"].iloc[ref] > df["Close"].iloc[ini]
            and df["EMA21"].iloc[ref] > df["EMA50"].iloc[ref]
            and df["EMA50"].iloc[ref] > df["EMA50"].iloc[max(0, ref - 20)])

    resist_prev, testeo = None, False
    zona = df["High"].iloc[:max(0, ref - 15)]
    if len(zona) > 20:
        resist_prev = float(zona.max())
        testeo = abs(float(df["High"].iloc[ref]) - resist_prev) / resist_prev <= TOL_RESISTENCIA
        res["lineas"]["resistencia"] = resist_prev

    res["c1"] = bool(tendencia and testeo)
    res["detalle"]["c1"] = (
        f"Tendencia previa alcista: <b>{'sí' if tendencia else 'no'}</b> · "
        f"Resistencia previa: <b>{('%.4f' % resist_prev) if resist_prev else 'no localizada'}</b> · "
        f"Testeo/dilatación: <b>{'sí' if testeo else 'no'}</b>")

    # ---- CIRCUNSTANCIA 3 : lateral plano junto a las EMAs ----
    t = df.iloc[-BARRAS_LATERAL:]
    rango = float(t["High"].max() - t["Low"].min())
    lateral = bool(rango <= FACTOR_ATR_LATERAL * atr)
    e21, e50 = float(df["EMA21"].iloc[-1]), float(df["EMA50"].iloc[-1])
    cerca = bool(min(abs(cierre - e21), abs(cierre - e50)) / cierre <= PROX_EMA)
    bajo = bool(cierre < e21 or cierre < e50)

    res["c3"] = bool(lateral and cerca)
    res["detalle"]["c3"] = (
        f"Rango últimas {BARRAS_LATERAL} barras: <b>{rango:.4f}</b> (ATR {atr:.4f}) · "
        f"Consolidación plana: <b>{'sí' if lateral else 'no'}</b> · "
        f"Pegado a EMA21/50: <b>{'sí' if cerca else 'no'}</b> · "
        f"Bajo las EMAs: <b>{'sí' if bajo else 'no'}</b>")

    # ---- NIVELES OPERATIVOS (corto) ----
    vmax = df["High"].iloc[-BARRAS_LATERAL:].max()
    if pico2 is not None:
        vmax = max(vmax, df["High"].iloc[pico2])
    stop = float(vmax) + MARGEN_STOP_ATR * atr
    riesgo = stop - cierre
    res["niveles"] = {"entrada": cierre, "stop": stop, "riesgo": riesgo,
                      "obj2": cierre - 2 * riesgo, "obj3": cierre - 3 * riesgo, "atr": atr}
    return res, df


def graficar(df, res, ticker, intervalo):
    x = df["Fecha"]
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=x, open=df["Open"], high=df["High"],
                                 low=df["Low"], close=df["Close"], name="Precio"))
    fig.add_trace(go.Scatter(x=x, y=df["EMA21"], line=dict(color="royalblue", width=1.6), name="EMA 21"))
    fig.add_trace(go.Scatter(x=x, y=df["EMA50"], line=dict(color="orange", width=1.6), name="EMA 50"))

    L = res["lineas"]
    if "resistencia" in L:
        fig.add_hline(y=L["resistencia"], line=dict(color="red", dash="dash", width=1.2),
                      annotation_text="Resistencia previa", annotation_position="top left")
    if "valle" in L:
        fig.add_hline(y=L["valle"][1], line=dict(color="magenta", dash="dot", width=1.2),
                      annotation_text="Valle", annotation_position="bottom left")
    if "picos" in L:
        i1, y1, i2, y2 = L["picos"]
        fig.add_trace(go.Scatter(x=[x.iloc[i1], x.iloc[i2]], y=[y1, y2], mode="markers+lines",
                                 marker=dict(color="red", size=11, symbol="triangle-down"),
                                 line=dict(color="red", width=1, dash="dot"), name="Doble techo"))
    if "directriz" in L:
        m1, ya, m2, yb = L["directriz"]
        fin = len(df) - 1
        fig.add_trace(go.Scatter(x=[x.iloc[m1], x.iloc[fin]], y=[ya, recta(m1, ya, m2, yb, fin)],
                                 line=dict(color="lime", width=1.6), name="Directriz alcista"))

    nv = res["niveles"]
    fig.add_hline(y=nv["stop"], line=dict(color="crimson", width=1),
                  annotation_text="STOP", annotation_position="right")
    fig.add_hline(y=nv["obj2"], line=dict(color="seagreen", width=1),
                  annotation_text="Obj 1:2", annotation_position="right")
    fig.add_hline(y=nv["obj3"], line=dict(color="darkgreen", width=1),
                  annotation_text="Obj 1:3", annotation_position="right")

    fig.update_layout(title=f"{ticker} — {intervalo}", xaxis_rangeslider_visible=False,
                      template="plotly_dark", height=640,
                      margin=dict(l=30, r=90, t=50, b=30),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom"))
    return fig


def tarjeta(ok, titulo, texto):
    clase = "card-ok" if ok else "card-ko"
    icono = "🟢" if ok else "🔴"
    st.markdown(
        f'<div class="card {clase}"><div class="badge">{icono} {"cumple" if ok else "no cumple"}</div>'
        f'<h4>{titulo}</h4><p>{texto}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# INTERFAZ
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuración")
    ticker = st.text_input("Ticker (formato Yahoo Finance)", value="USDJPY=X")
    st.caption("Ej.: USDJPY=X · NQ=F · ES=F · AAPL · EURUSD=X · SAN.MC")
    intervalo = st.selectbox("Temporalidad", options=["1d", "1h"],
                             format_func=lambda v: "Diaria (1d)" if v == "1d" else "Intradía (1h)")
    analizar_btn = st.button("Analizar gráfico", type="primary", use_container_width=True)
    st.divider()
    st.caption("Detección automática aproximada. La pauta de Cava es discrecional: "
               "valida siempre el gráfico manualmente antes de operar.")

st.title("Pauta de Techo Corto")
st.caption("Metodología José Luis Cava — evaluación automática de las 3 circunstancias")

if analizar_btn or "ultimo" in st.session_state:
    if analizar_btn:
        st.session_state["ultimo"] = (ticker.strip(), intervalo)
    tk, iv = st.session_state["ultimo"]

    with st.spinner(f"Descargando {tk} ({iv})..."):
        df = descargar(tk, iv)

    if df is None or len(df) < 80:
        st.error("No se han obtenido datos suficientes. Revisa el símbolo o prueba otra temporalidad.")
        st.stop()

    res, df = analizar(df)
    nv = res["niveles"]
    cumplidas = sum([res["c1"], res["c2"], res["c3"]])

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in [
        (c1, "Circunstancias", f"{cumplidas}/3"),
        (c2, "Cierre actual", f"{nv['entrada']:.4f}"),
        (c3, "Stop sugerido", f"{nv['stop']:.4f}"),
        (c4, "ATR (14)", f"{nv['atr']:.4f}"),
    ]:
        col.markdown(f'<div class="metric-box"><div class="lbl">{lbl}</div>'
                     f'<div class="val">{val}</div></div>', unsafe_allow_html=True)

    st.write("")
    if cumplidas == 3:
        st.success("Pauta completa. Vigilar el giro a la baja desde la consolidación.")
    elif cumplidas == 2:
        st.warning("Pauta en formación. Falta una circunstancia por confirmar.")
    else:
        st.info("Sin pauta según estos criterios. No operar por este motivo.")

    a, b, c = st.columns(3)
    with a:
        tarjeta(res["c1"], "1 · Tendencia alcista previa y testeo de resistencia", res["detalle"]["c1"])
    with b:
        tarjeta(res["c2"], "2 · Doble techo, valle roto y directriz perforada", res["detalle"]["c2"])
    with c:
        tarjeta(res["c3"], "3 · Consolidación plana junto a EMA21 / EMA50", res["detalle"]["c3"])

    st.subheader("Niveles orientativos (posición corta)")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Entrada", f"{nv['entrada']:.4f}")
    n2.metric("Stop Loss", f"{nv['stop']:.4f}", f"riesgo {nv['riesgo']:.4f}")
    n3.metric("Objetivo 1:2", f"{nv['obj2']:.4f}")
    n4.metric("Objetivo 1:3", f"{nv['obj3']:.4f}")

    st.plotly_chart(graficar(df, res, tk, iv), use_container_width=True)
else:
    st.info("Introduce un ticker en el panel lateral y pulsa **Analizar gráfico**.")
