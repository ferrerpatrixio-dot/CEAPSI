import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json
import calendar
import io
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="CEAPSI — Predicción de Ventas",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constantes ──────────────────────────────────────────────────────────────
ESCALA_MM   = 1_000_000
FACTOR      = 1.063
TIPOS       = ["Adultos", "Infantil", "Teleconsulta"]
TIPO_COD    = {"Adultos": 0, "Infantil": 1, "Teleconsulta": 2}
COLOR_TIPOS = {"Adultos": "#3b82f6", "Infantil": "#10b981", "Teleconsulta": "#f59e0b"}
MESES_ES    = {1:"Enero",2:"Febrero",3:"Marzo",4:"Abril",5:"Mayo",6:"Junio",
               7:"Julio",8:"Agosto",9:"Septiembre",10:"Octubre",11:"Noviembre",12:"Diciembre"}

# ════════════════════════════════════════════════════════════════════
# AUTENTICACIÓN
# ════════════════════════════════════════════════════════════════════
def check_auth() -> bool:
    if st.session_state.get("auth_ok"):
        return True

    st.markdown("""
    <style>
    .login-wrap { max-width:380px; margin:80px auto; padding:40px;
                  background:#f8fafc; border-radius:16px;
                  box-shadow:0 4px 24px rgba(0,0,0,.10); }
    .login-title { text-align:center; font-size:2rem; margin-bottom:4px; }
    .login-sub   { text-align:center; color:#64748b; margin-bottom:28px; }
    </style>
    """, unsafe_allow_html=True)

    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown('<div class="login-wrap">', unsafe_allow_html=True)
        st.markdown('<p class="login-title">🏥 CEAPSI</p>', unsafe_allow_html=True)
        st.markdown('<p class="login-sub">Predicción de Ventas · Acceso restringido</p>',
                    unsafe_allow_html=True)

        pwd = st.text_input("Contraseña", type="password", placeholder="Ingresa la clave")
        if st.button("Ingresar", type="primary", use_container_width=True):
            if pwd == st.secrets.get("password", ""):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                st.error("Contraseña incorrecta")
        st.markdown('</div>', unsafe_allow_html=True)
    return False

if not check_auth():
    st.stop()

# ── Logout en sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**🏥 CEAPSI**")
    if st.button("Cerrar sesión"):
        st.session_state["auth_ok"] = False
        st.rerun()

# ── Carga de artefactos ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    mdl = xgb.XGBRegressor()
    mdl.load_model("xgb_ventas_clinica_v5.json")
    ph   = pickle.load(open("prom_hist_dict.pkl",      "rb"))
    pc   = pickle.load(open("prom_cant_dict.pkl",      "rb"))
    fer  = pickle.load(open("feriados_set.pkl",        "rb"))
    fi   = pickle.load(open("fecha_inicio.pkl",        "rb"))
    vac  = pickle.load(open("vacaciones_invierno.pkl", "rb"))
    feat = pickle.load(open("features_v5.pkl",         "rb"))
    return mdl, ph, pc, fer, fi, vac, feat

model, PROM_HIST, PROM_CANT, FERIADOS, FECHA_INICIO, VAC_INV, FEATURES = load_artifacts()

@st.cache_data
def load_metrics():
    try:
        with open("model_metrics.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_training_data():
    try:
        df = pd.read_csv("entrenamiento.csv", encoding="utf-8-sig")
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_validation_data():
    try:
        df = pd.read_csv("validacion.csv", encoding="utf-8-sig")
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        return df
    except FileNotFoundError:
        return None

metrics   = load_metrics()
df_train  = load_training_data()
df_valid  = load_validation_data()

# ── Lógica de predicción ────────────────────────────────────────────────────
def calc_vacaciones(mes: int, dia: int, año: int) -> int:
    if mes == 12 and dia >= 26: return 1
    if mes in (1, 2):           return 1
    if mes == 3  and dia == 1:  return 1
    if año in VAC_INV:
        fi, ff = VAC_INV[año]
        if fi <= date(año, mes, dia) <= ff: return 1
    return 0

def predecir_mes(mes: int, año: int) -> pd.DataFrame:
    n_dias  = calendar.monthrange(año, mes)[1]
    td      = timedelta
    cache_v = {}
    cache_c = {}

    def _v(fd, tc):
        return cache_v.get((fd, tc), PROM_HIST.get((fd.month, tc), 0.0) / ESCALA_MM)

    def _c(fd, tc):
        return cache_c.get((fd, tc), PROM_CANT.get((fd.month, tc), 0.0))

    filas = []
    for dia in range(1, n_dias + 1):
        fd      = date(año, mes, dia)
        diasem  = fd.weekday()
        cerrado = diasem == 6 or fd in FERIADOS

        for tipo in TIPOS:
            tc = TIPO_COD[tipo]
            if cerrado:
                cache_v[(fd, tc)] = 0.0
                cache_c[(fd, tc)] = 0.0
                filas.append({"Fecha": fd, "DIA": dia, "Tipo Consulta": tipo,
                               "VENTAS_DIA": 0.0, "cerrado": True})
                continue

            prom_h  = PROM_HIST.get((mes, tc), 0.0) / ESCALA_MM
            l7      = _v(fd - td(7),  tc)
            l14     = _v(fd - td(14), tc)
            l21     = _v(fd - td(21), tc)
            l28     = _v(fd - td(28), tc)
            mov4s   = (l7 + l14 + l21 + l28) / 4.0
            c_l7    = _c(fd - td(7), tc)
            c_mov4s = (_c(fd-td(7),tc)+_c(fd-td(14),tc)+
                       _c(fd-td(21),tc)+_c(fd-td(28),tc)) / 4.0

            X = pd.DataFrame([{
                "DIASEM":      diasem,
                "tipo_cod":    tc,
                "A_FERIADO":   0,
                "TENDENCIA":   (pd.Timestamp(fd) - FECHA_INICIO).days,
                "PROM_HIST":   prom_h,
                "LAG7":        l7,
                "MEDIA_MOV4S": mov4s,
                "VACACIONES":  calc_vacaciones(mes, dia, año),
                "CANT_LAG7":   c_l7,
                "CANT_MOV4S":  c_mov4s,
            }])

            pred_mm = max(float(model.predict(X)[0]), 0.0)
            cache_v[(fd, tc)] = pred_mm
            cache_c[(fd, tc)] = c_l7
            filas.append({"Fecha": fd, "DIA": dia, "Tipo Consulta": tipo,
                          "VENTAS_DIA": pred_mm * ESCALA_MM, "cerrado": False})

    return pd.DataFrame(filas)

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    return buf.getvalue()

# ── Estilos ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-box { background:#f0f4ff; border-radius:10px; padding:14px 18px;
           text-align:center; margin-bottom:8px; }
.kpi-val { font-size:2rem; font-weight:700; }
.kpi-lbl { font-size:.82rem; color:#64748b; margin-top:4px; }
</style>""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏥 CEAPSI — Predicción de Ventas")
st.caption("Modelo XGBoost v5 · Consultas Adultos · Infantil · Teleconsulta · Las Condes")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Modelo", "📊 Datos", "📈 Métricas", "🔮 Predecir mes"])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — MODELO
# ════════════════════════════════════════════════════════════════════
with tab1:
    info = (metrics or {}).get("training_info", {})
    c_a, c_b = st.columns(2)

    with c_a:
        st.subheader("Datos de entrenamiento")
        st.markdown(f"""
| Ítem | Valor |
|---|---|
| Período entrenamiento | {info.get('fecha_min_train','Abr 2023')} → {info.get('fecha_max_train','Abr 2026')} |
| Registros usados | {info.get('n_train','2 761')} filas |
| Validación out-of-sample | {info.get('fecha_min_test','May 2')} → {info.get('fecha_max_test','May 15 2026')} |
| Registros validación | {info.get('n_test', 36)} filas |
| Tipos de consulta | Adultos · Infantil · Teleconsulta |
| Target | VENTAS diarias por tipo ($) |
| Factor de corrección mensual | ×1.063 |
""")

    with c_b:
        st.subheader("Parámetros del modelo")
        st.markdown(f"""
| Parámetro | Valor |
|---|---|
| Algoritmo | XGBoost (reg:squarederror) |
| Árboles (n_estimators) | {info.get('n_opt', 172)} (early stopping) |
| Learning rate | 0.03 |
| Max depth | 5 |
| Min child weight | 5 |
| Regularización α / λ | 10 / 10 |
| Subsample / colsample | 0.80 / 0.80 |
| Holdout early stopping | Mar–Abr 2026 |
| Escala target | ÷ 1 000 000 (MM$) |
""")

    st.divider()
    st.subheader("Features del modelo — 10 variables autónomas")
    st.caption("*Autónomo*: predice sin necesitar datos del día actual; usa solo historial pasado.")

    st.dataframe(pd.DataFrame([
        ("DIASEM",      "Calendario",  "Día de la semana (0=Lun…6=Dom). Captura el patrón semanal de demanda.", "Alta"),
        ("tipo_cod",    "Categórica",  "Tipo de consulta: Adultos=0, Infantil=1, Teleconsulta=2.", "Alta"),
        ("A_FERIADO",   "Binaria",     "1 si la fecha es feriado nacional → VENTAS=0 ese día.", "Alta"),
        ("TENDENCIA",   "Temporal",    "Días desde Abr 2023. Captura crecimiento de demanda a largo plazo.", "Media"),
        ("PROM_HIST",   "Histórica",   "Promedio histórico de VENTAS para ese mes y tipo (MM$). Ancla estacional fuerte.", "Alta"),
        ("LAG7",        "Lag ventas",  "VENTAS de hace 7 días exactos (mismo día de semana). Mejor predictor de corto plazo.", "Muy alta"),
        ("MEDIA_MOV4S", "Lag ventas",  "Promedio de lags -7/-14/-21/-28 días. Suaviza ruido puntual de LAG7.", "Alta"),
        ("VACACIONES",  "Binaria",     "1 en vacaciones de verano (dic 26–mar 1) o invierno (MINEDUC).", "Media"),
        ("CANT_LAG7",   "Lag conteo",  "Cantidad de pacientes atendidos hace 7 días. Volumen sin efecto precio.", "Media"),
        ("CANT_MOV4S",  "Lag conteo",  "Promedio de cantidad de pacientes en las últimas 4 semanas.", "Media"),
    ], columns=["Feature", "Tipo", "Descripción", "Importancia relativa"]),
    use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════
# TAB 2 — DATOS
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Datos utilizados en el modelo")

    sub_train, sub_valid = st.tabs(["📂 Entrenamiento", "🧪 Validación"])

    # ── Entrenamiento ────────────────────────────────────────────────
    with sub_train:
        if df_train is None:
            st.info("entrenamiento.csv no disponible. Ejecuta el notebook (Paso 2) para generarlo.")
        else:
            df_t = df_train.copy()
            df_t["Mes"]  = df_t["Fecha"].dt.to_period("M").astype(str)
            df_t["VENTAS_MM"] = df_t["VENTAS"] / ESCALA_MM

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Registros", f"{len(df_t):,}")
            c2.metric("Período",
                      f"{df_t['Fecha'].min().strftime('%b %Y')} → {df_t['Fecha'].max().strftime('%b %Y')}")
            c3.metric("Tipos de consulta", df_t["Tipo Consulta"].nunique())
            c4.metric("VENTAS total ($)", f"${df_t['VENTAS'].sum():,.0f}")

            st.divider()

            # Pivot mensual por tipo
            st.markdown("**Ventas mensuales por tipo de consulta (MM$)**")
            pivot_m = (df_t.pivot_table(index="Mes", columns="Tipo Consulta",
                                         values="VENTAS_MM", aggfunc="sum")
                           .reindex(columns=TIPOS).fillna(0))
            pivot_m["TOTAL"] = pivot_m.sum(axis=1)
            st.dataframe(pivot_m.style.format("${:.2f}"), use_container_width=True)

            # Gráfico evolución mensual
            df_chart = pivot_m.drop(columns="TOTAL").reset_index().melt(
                id_vars="Mes", var_name="Tipo", value_name="VENTAS MM$")
            fig = px.line(df_chart, x="Mes", y="VENTAS MM$", color="Tipo",
                          color_discrete_map=COLOR_TIPOS,
                          title="Evolución mensual de ventas — conjunto de entrenamiento")
            fig.update_xaxes(tickangle=45, nticks=20)
            fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Descarga
            st.download_button(
                "⬇️ Descargar entrenamiento.csv",
                data=df_train.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="entrenamiento.csv", mime="text/csv"
            )

    # ── Validación ───────────────────────────────────────────────────
    with sub_valid:
        if df_valid is None:
            st.info("validacion.csv no disponible. Ejecuta el notebook (Paso 2) para generarlo.")
        else:
            st.markdown(f"**{len(df_valid)} registros · Mayo 2–15 2026 · conjunto out-of-sample "
                        f"(nunca visto durante el entrenamiento)**")

            df_v = df_valid.copy()
            df_v["Fecha_str"] = df_v["Fecha"].dt.strftime("%a %d %b")
            df_v["VENTAS_fmt"] = df_v["VENTAS"].apply(lambda x: f"${x:,.0f}")

            # Pivot por fecha y tipo
            pivot_v = (df_v.pivot_table(index="Fecha_str", columns="Tipo Consulta",
                                         values="VENTAS", aggfunc="sum")
                           .reindex(columns=TIPOS).fillna(0))
            pivot_v["TOTAL DÍA"] = pivot_v.sum(axis=1)
            st.dataframe(pivot_v.style.format("${:,.0f}"), use_container_width=True)

            # Resumen por tipo
            st.markdown("**Total período Mayo 2–15 por tipo**")
            tot_v = df_v.groupby("Tipo Consulta")["VENTAS"].sum().reindex(TIPOS)
            c1, c2, c3, c4 = st.columns(4)
            for col, tipo in zip([c1, c2, c3], TIPOS):
                col.metric(tipo, f"${tot_v[tipo]:,.0f}")
            c4.metric("TOTAL", f"${tot_v.sum():,.0f}")

            # Gráfico barras
            fig_v = px.bar(df_v, x="Fecha_str", y="VENTAS", color="Tipo Consulta",
                           color_discrete_map=COLOR_TIPOS,
                           labels={"Fecha_str":"Fecha","VENTAS":"Ventas ($)"},
                           title="Ventas reales — Validación Mayo 2–15 2026")
            fig_v.update_xaxes(tickangle=45)
            fig_v.update_layout(margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_v, use_container_width=True)

            st.download_button(
                "⬇️ Descargar validacion.csv",
                data=df_valid.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name="validacion.csv", mime="text/csv"
            )

# ════════════════════════════════════════════════════════════════════
# TAB 3 — MÉTRICAS
# ════════════════════════════════════════════════════════════════════
with tab3:
    if metrics is None:
        st.info("Ejecuta `python generate_metrics.py` después de correr el notebook para ver las métricas aquí.")
    else:
        r2    = metrics.get("r2")
        mape  = metrics.get("mape")
        rmse  = metrics.get("rmse")
        mae   = metrics.get("mae")
        sesgo = metrics.get("sesgo_medio")

        kpis = [
            (r2,   "R²",    "{:.4f}",   lambda v: v >= 0.75, "≥ 0.75 aprueba"),
            (mape, "MAPE",  "{:.1f}%",  lambda v: v <= 18.0, "≤ 18% aprueba"),
            (rmse, "RMSE",  "${:,.0f}", lambda v: True,      "Error cuadrático medio"),
            (mae,  "MAE",   "${:,.0f}", lambda v: True,      "Error absoluto medio"),
        ]
        cols = st.columns(4)
        for col, (val, lbl, fmt, ok_fn, hint) in zip(cols, kpis):
            if val is not None:
                color = "#16a34a" if ok_fn(val) else "#dc2626"
                col.markdown(f"""
<div class="kpi-box">
  <div class="kpi-val" style="color:{color}">{fmt.format(val)}</div>
  <div class="kpi-lbl">{lbl}<br><span style="font-size:.75rem">{hint}</span></div>
</div>""", unsafe_allow_html=True)

        if sesgo is not None:
            dir_ = "sobreestima" if sesgo > 0 else "subestima"
            st.info(f"**Sesgo medio:** ${sesgo:,.0f} ({dir_}) — factor ×1.063 corrige la subestimación sistemática.")

        st.divider()
        st.subheader("Importancia de features (gain)")
        scores = model.get_booster().get_score(importance_type="gain")
        df_imp = pd.DataFrame(scores.items(), columns=["Feature","Gain"]).sort_values("Gain")
        fig_imp = px.bar(df_imp, x="Gain", y="Feature", orientation="h",
                         color="Gain", color_continuous_scale="Blues")
        fig_imp.update_layout(showlegend=False, coloraxis_showscale=False,
                              margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_imp, use_container_width=True)

        if "pred_vs_real" in metrics and metrics["pred_vs_real"]:
            pvr = pd.DataFrame(metrics["pred_vs_real"])
            fig_sc = px.scatter(pvr, x="real", y="pred", color="tipo",
                                color_discrete_map=COLOR_TIPOS,
                                labels={"real":"Real ($)","pred":"Predicción ($)","tipo":"Tipo"},
                                title="Predicho vs Real — Validación Mayo 2–15 2026")
            lim = max(pvr["real"].max(), pvr["pred"].max()) * 1.05
            fig_sc.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim,
                             line=dict(color="red", dash="dash", width=1))
            st.plotly_chart(fig_sc, use_container_width=True)

        st.divider()
        st.subheader("Decisión ciclo de mantención")
        ok_r2, ok_mape = r2 is not None and r2 >= 0.75, mape is not None and mape <= 18.0
        st.markdown(f"""
| Métrica | Valor | Umbral | Estado |
|---|---|---|---|
| R² | {f"{r2:.4f}" if r2 is not None else "N/D"} | ≥ 0.75 | {"✅" if ok_r2 else "❌"} |
| MAPE | {f"{mape:.1f}%" if mape is not None else "N/D"} | ≤ 18% | {"✅" if ok_mape else "❌"} |
""")
        if ok_r2 and ok_mape:
            st.success("**✅ APROBADO** — el modelo puede pasar a producción.")
        else:
            st.error("**❌ RECHAZADO** — conservar modelo anterior. Revisar datos y código.")

# ════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICCIÓN
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Predicción mensual de ventas")
    st.caption(f"Factor de corrección ×{FACTOR} aplicado al total mensual.")

    c_mes, c_año, c_btn = st.columns([2, 2, 3])
    with c_mes:
        mes_sel = st.selectbox("Mes", list(MESES_ES.keys()), index=6,
                               format_func=lambda m: MESES_ES[m])
    with c_año:
        año_sel = st.number_input("Año", min_value=2024, max_value=2030, value=2026, step=1)
    with c_btn:
        st.markdown("&nbsp;")
        run = st.button("🔮 Predecir", type="primary", use_container_width=True)

    if run:
        with st.spinner(f"Calculando {MESES_ES[mes_sel]} {año_sel}…"):
            df_pred = predecir_mes(int(mes_sel), int(año_sel))

        df_open = df_pred[~df_pred["cerrado"]].copy()

        st.subheader(f"Predicción diaria — {MESES_ES[mes_sel]} {año_sel}")
        pivot = (df_open.pivot_table(index="Fecha", columns="Tipo Consulta",
                                      values="VENTAS_DIA", aggfunc="sum")
                        .reindex(columns=TIPOS).fillna(0))
        pivot["TOTAL DÍA"] = pivot.sum(axis=1)
        pivot.index = [d.strftime("%a %d") for d in pivot.index]
        st.dataframe(pivot.style.format("${:,.0f}"), use_container_width=True)

        df_open["Fecha_str"] = df_open["Fecha"].astype(str)
        fig = px.bar(df_open, x="Fecha_str", y="VENTAS_DIA", color="Tipo Consulta",
                     color_discrete_map=COLOR_TIPOS,
                     labels={"Fecha_str":"Fecha","VENTAS_DIA":"Ventas ($)"},
                     title=f"Ventas diarias — {MESES_ES[mes_sel]} {año_sel}")
        fig.update_xaxes(tickangle=45)
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Resumen mensual")
        totales   = df_open.groupby("Tipo Consulta")["VENTAS_DIA"].sum().reindex(TIPOS)
        total_sin = totales.sum()
        total_con = total_sin * FACTOR

        df_res = pd.concat([
            totales.reset_index().rename(columns={"VENTAS_DIA":"VENTAS MES ($)"}),
            pd.DataFrame([{"Tipo Consulta":"TOTAL (sin factor)", "VENTAS MES ($)": total_sin}]),
            pd.DataFrame([{"Tipo Consulta":f"TOTAL ×{FACTOR}",  "VENTAS MES ($)": total_con}]),
        ], ignore_index=True)
        st.dataframe(df_res.style.format({"VENTAS MES ($)": "${:,.0f}"}),
                     use_container_width=True, hide_index=True)

        st.success(f"**Estimación {MESES_ES[mes_sel]} {año_sel}** con factor ×{FACTOR}: "
                   f"**${total_con:,.0f}**")

        st.download_button(
            "⬇️ Descargar predicción Excel",
            data=df_to_excel_bytes(df_res),
            file_name=f"prediccion_{MESES_ES[mes_sel]}_{año_sel}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
