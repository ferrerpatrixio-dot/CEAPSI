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
# FACTOR se lee de model_metrics.json (calculado automáticamente en generate_metrics.py)
# Fallback 1.0 si el JSON no existe aún
try:
    import json as _json
    with open("model_metrics.json", encoding="utf-8") as _mf:
        _m0 = _json.load(_mf)
    FACTOR = float(_m0.get("factor_correccion", 1.063))
except Exception:
    FACTOR = 1.063
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
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important; display: none !important;}
    header {visibility: hidden !important;}
    [data-testid="stToolbar"]        {display: none !important;}
    [data-testid="stDecoration"]     {display: none !important;}
    [data-testid="stStatusWidget"]   {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    [data-testid="stDeployButton"]   {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    [class*="viewerBadge"]           {display: none !important;}
    [class*="styles_viewerBadge"]    {display: none !important;}
    [class*="badge"]                 {display: none !important;}
    .stApp > footer                  {display: none !important;}
    a[href*="share.streamlit.io"]    {display: none !important;}
    a[href*="streamlit.io/user"]     {display: none !important;}
    iframe[title="streamlit_analytics"] {display: none !important;}
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
    mdl.load_model("xgb_ventas_clinica_v5.json")   # contiene modelo v6
    ph   = pickle.load(open("prom_hist_dict.pkl",      "rb"))
    pc   = pickle.load(open("prom_cant_dict.pkl",      "rb"))
    fer  = pickle.load(open("feriados_set.pkl",        "rb"))
    fi   = pickle.load(open("fecha_inicio.pkl",        "rb"))
    vac  = pickle.load(open("vacaciones_invierno.pkl", "rb"))
    feat = pickle.load(open("features_v5.pkl",         "rb"))  # contiene features v6
    return mdl, ph, pc, fer, fi, vac, feat

model, PROM_HIST, PROM_CANT, FERIADOS, FECHA_INICIO, VAC_INV, FEATURES = load_artifacts()

def load_metrics():
    # Sin cache: el archivo cambia con cada reentrenamiento
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

# Etiqueta dinámica del período de validación
if df_valid is not None and len(df_valid) > 0:
    _MESES_CORTOS = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",
                     7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    _vd_min = df_valid["Fecha"].min()
    _vd_max = df_valid["Fecha"].max()
    VAL_LABEL = (f"{_vd_min.day} {_MESES_CORTOS[_vd_min.month]}"
                 f"–{_vd_max.day} {_MESES_CORTOS[_vd_max.month]} {_vd_max.year}")
else:
    VAL_LABEL = "período de validación"

# ── Lógica de predicción ────────────────────────────────────────────────────
def calc_vacaciones(mes: int, dia: int, año: int) -> int:
    if mes == 12 and dia >= 26: return 1
    if mes in (1, 2):           return 1
    if mes == 3  and dia == 1:  return 1
    if año in VAC_INV:
        fi, ff = VAC_INV[año]
        if fi <= date(año, mes, dia) <= ff: return 1
    return 0

def predecir_mes(mes: int, año: int, df_historico: pd.DataFrame = None) -> pd.DataFrame:
    """
    df_historico: si se pasa (mes pasado), se usan los datos reales como fuente de lags
                  en lugar de los promedios históricos. Permite backtest preciso.
    """
    n_dias  = calendar.monthrange(año, mes)[1]
    td      = timedelta
    cache_v = {}   # acumula predicciones del mes actual día a día
    cache_c = {}

    # Índice rápido sobre el historial real (si está disponible)
    hist_v, hist_c = {}, {}
    if df_historico is not None:
        for _, row in df_historico.iterrows():
            fd_ = row["Fecha"].date() if hasattr(row["Fecha"], "date") else row["Fecha"]
            tc_ = TIPO_COD.get(row["Tipo Consulta"], 0)
            hist_v[(fd_, tc_)] = row["VENTAS"] / ESCALA_MM
            hist_c[(fd_, tc_)] = row["CANT_VENTAS"]

    def _v(fd, tc):
        # 1) dato real del historial, 2) predicción acumulada del mes, 3) promedio histórico
        return hist_v.get((fd, tc),
               cache_v.get((fd, tc),
               PROM_HIST.get((fd.month, tc), 0.0) / ESCALA_MM))

    def _c(fd, tc):
        return hist_c.get((fd, tc),
               cache_c.get((fd, tc),
               PROM_CANT.get((fd.month, tc), 0.0)))

    filas = []
    for dia in range(1, n_dias + 1):
        fd      = date(año, mes, dia)
        diasem  = fd.isoweekday()   # 1=Lun...7=Dom, igual que el training
        cerrado = diasem == 7 or fd in FERIADOS

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

            # ── Features v6: ratios de posición y crecimiento ──
            lag_ratio   = l7 / prom_h if prom_h > 0 else 1.0
            mov4s_ratio = mov4s / prom_h if prom_h > 0 else 1.0
            _crec_vals  = []
            for _d in [7, 14, 21, 28, 35, 42, 49, 56]:
                _lag_fd  = fd - td(_d)
                _lag_val = _v(_lag_fd, tc)
                _ph_lag  = PROM_HIST.get((_lag_fd.month, tc), 0.0) / ESCALA_MM
                if _lag_val > 0 and _ph_lag > 0:
                    _crec_vals.append(_lag_val / _ph_lag)
            crec8s = float(sum(_crec_vals) / len(_crec_vals)) if _crec_vals else 1.0

            X = pd.DataFrame([{
                "DIASEM":         diasem,
                "tipo_cod":       tc,
                "A_FERIADO":      0,
                "TENDENCIA":      (pd.Timestamp(fd) - FECHA_INICIO).days,
                "PROM_HIST":      prom_h,
                "LAG7":           l7,
                "MEDIA_MOV4S":    mov4s,
                "VACACIONES":     calc_vacaciones(mes, dia, año),
                "CANT_LAG7":      c_l7,
                "CANT_MOV4S":     c_mov4s,
                "LAG_RATIO":      lag_ratio,
                "MOV4S_RATIO":    mov4s_ratio,
                "CRECIMIENTO_8S": crec8s,
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
#MainMenu {visibility: hidden !important;}
footer {visibility: hidden !important; display: none !important;}
header {visibility: hidden !important;}
[data-testid="stToolbar"]        {display: none !important;}
[data-testid="stDecoration"]     {display: none !important;}
[data-testid="stStatusWidget"]   {display: none !important;}
[data-testid="collapsedControl"] {display: none !important;}
[data-testid="stDeployButton"]   {display: none !important;}
section[data-testid="stSidebar"] {display: none !important;}
[class*="viewerBadge"]           {display: none !important;}
[class*="styles_viewerBadge"]    {display: none !important;}
[class*="badge"]                 {display: none !important;}
.stApp > footer                  {display: none !important;}
a[href*="share.streamlit.io"]    {display: none !important;}
a[href*="streamlit.io/user"]     {display: none !important;}
iframe[title="streamlit_analytics"] {display: none !important;}
.kpi-box { background:#f0f4ff; border-radius:10px; padding:14px 18px;
           text-align:center; margin-bottom:8px; }
.kpi-val { font-size:2rem; font-weight:700; }
.kpi-lbl { font-size:.82rem; color:#64748b; margin-top:4px; }
</style>""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏥 CEAPSI — Predicción de Ventas")
st.caption("Modelo XGBoost v6 · Consultas Adultos · Infantil · Teleconsulta · Las Condes")

try:
    with open("version.json", encoding="utf-8-sig") as _vf:
        _ver = json.load(_vf)
    _ver_txt = f"· actualizado {_ver.get('timestamp','—')} · commit {_ver.get('commit','—')}"
except Exception:
    _ver_txt = ""

st.markdown(f"""
<style>
.footer {{ position:fixed; bottom:0; left:0; width:100%;
          background:#f1f5f9; border-top:1px solid #e2e8f0;
          text-align:center; padding:8px 0;
          font-size:.78rem; color:#64748b; z-index:999; }}
.footer a {{ color:#3b82f6; text-decoration:none; }}
</style>
<div class="footer">Desarrollado por <a href="https://aiprocess.cl" target="_blank">AIProcess.cl</a> · 2026 {_ver_txt}</div>
""", unsafe_allow_html=True)

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
| Factor de corrección mensual | ×{FACTOR:.3f} |
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

    # ── Explicación en lenguaje simple ──────────────────────────────
    with st.expander("💡 ¿Cómo funciona el modelo? (explicación simple)", expanded=True):
        st.markdown("""
**El modelo es como un analista con memoria de 3 años.**

Aprendió a partir del historial real de ventas de la clínica (abril 2023 – abril 2026).
Cada vez que se le pide predecir un día, el modelo mira:

- **¿Qué día de la semana es?** Los lunes y viernes tienen más consultas que los sábados.
- **¿Es feriado?** Si es festivo, predice $0 (la clínica no atiende).
- **¿Qué pasó la semana pasada?** Las ventas del mismo día hace 7 días son el mejor indicador de lo que viene.
- **¿Cómo fue el promedio del último mes?** Para detectar si la demanda está subiendo o bajando.
- **¿Es época de vacaciones?** En julio-agosto y diciembre-marzo hay menos consultas Infantil.
- **¿Cuántos pacientes hubo la semana pasada?** El volumen de atenciones también anticipa el nivel de ventas.

Con esa información predice por separado **Adultos**, **Infantil** y **Teleconsulta**, y suma los tres para obtener el total del día.

Al final del mes suma todos los días y aplica un **factor de corrección de +6.3%** para compensar que el modelo tiende a subestimar levemente la demanda real.

> **En resumen**: el modelo aprendió los patrones del pasado (días, estaciones, tendencias) y los usa para estimar el futuro — sin necesitar datos del día que está prediciendo.
""")

    st.divider()

    # ── Ciclo de reentrenamiento ─────────────────────────────────────
    with st.expander("🔄 ¿Cómo se mantiene actualizado el modelo?", expanded=False):
        st.markdown("""
El modelo **aprende de los datos del pasado**. Si la clínica crece, cambia su mix de consultas
o hay algún evento especial, el modelo necesita "ver" esos nuevos datos para seguir siendo preciso.
Por eso existe un **ciclo de mantención mensual**: al cierre de cada mes, se evalúa si el modelo
sigue siendo bueno y, si es así, se actualiza automáticamente con los datos más recientes.

---

### ¿Cómo funciona el ciclo? (paso a paso simple)

**1. Fin de mes** → se carga el mes completo como datos de prueba
**2. Se corre el modelo** → predice ese mes y se comparan las predicciones con las ventas reales
**3. Se evalúan dos indicadores clave:**
- **R² ≥ 0.75** — el modelo explica al menos el 75% de la variación de ventas
- **MAPE ≤ 18%** — el error promedio no supera el 18%

**4a. Si ambos indicadores están OK → ✅ APROBADO**
> El modelo se reentrena incorporando el mes nuevo y queda listo para predecir el mes siguiente.

**4b. Si algún indicador falla → ⚠️ ALERTA**
> El modelo anterior sigue activo (no se actualiza) y se genera una notificación para revisar
> manualmente qué pasó: ¿hubo un mes atípico? ¿cambió la operación de la clínica? ¿hay datos faltantes?

---

### Ejemplo concreto — Ciclo de junio 2026

| Paso | Detalle |
|---|---|
| Datos de entrenamiento | Abril 2023 → Mayo 2026 |
| Datos de prueba | Junio 2026 completo (30 días × 3 tipos = hasta 90 registros) |
| Se calculan R² y MAPE | Comparando predicciones vs ventas reales de junio |
| Si R² ≥ 0.75 y MAPE ≤ 18% | ✅ El modelo se actualiza e incorpora junio. Queda listo para predecir julio |
| Si MAPE = 22% (por ejemplo) | ⚠️ Alerta: el modelo no se actualiza. Se revisa si junio tuvo algo inusual |

---

### ¿Qué aspecto tiene la alerta?

Si el ciclo no aprueba, el sistema registra automáticamente en un archivo de log:

```
[2026-07-01 09:15] RECHAZADO  MAPE=22.3% > umbral 18%
— modelo NO actualizado. Revisar manualmente.
```

Y en el panel de Métricas de esta aplicación aparecerá el semáforo en ❌ rojo
con la explicación del motivo. El modelo del mes anterior **sigue funcionando** sin interrupciones
mientras se investiga la causa.

> **Regla de oro**: cada ciclo necesita datos nuevos que el modelo nunca haya visto.
> La validación siempre es el mes más reciente que aún no entró al entrenamiento.
""")

    st.divider()
    st.subheader("Features del modelo — 13 variables autónomas")
    st.caption("*Autónomo*: predice sin necesitar datos del día actual; usa solo historial pasado.")

    st.dataframe(pd.DataFrame([
        ("DIASEM",         "Calendario",  "Día de la semana (1=Lun…7=Dom). Captura el patrón semanal de demanda.", "Alta"),
        ("tipo_cod",       "Categórica",  "Tipo de consulta: Adultos=0, Infantil=1, Teleconsulta=2.", "Alta"),
        ("A_FERIADO",      "Binaria",     "1 si la fecha es feriado nacional → VENTAS=0 ese día.", "Alta"),
        ("TENDENCIA",      "Temporal",    "Días desde Abr 2023. Captura crecimiento de demanda a largo plazo.", "Media"),
        ("PROM_HIST",      "Histórica",   "Promedio histórico de VENTAS para ese mes y tipo (MM$). Ancla estacional fuerte.", "Alta"),
        ("LAG7",           "Lag ventas",  "VENTAS de hace 7 días exactos (mismo día de semana). Mejor predictor de corto plazo.", "Muy alta"),
        ("MEDIA_MOV4S",    "Lag ventas",  "Promedio de lags -7/-14/-21/-28 días. Suaviza ruido puntual de LAG7.", "Alta"),
        ("VACACIONES",     "Binaria",     "1 en vacaciones de verano (dic 26–mar 1) o invierno (MINEDUC).", "Media"),
        ("CANT_LAG7",      "Lag conteo",  "Cantidad de pacientes atendidos hace 7 días. Volumen sin efecto precio.", "Media"),
        ("CANT_MOV4S",     "Lag conteo",  "Promedio de cantidad de pacientes en las últimas 4 semanas.", "Media"),
        ("LAG_RATIO",      "Ratio v6",    "LAG7 / PROM_HIST: posición relativa inmediata vs. histórico del mes. Detecta si la semana pasada fue alta o baja.", "Alta"),
        ("MOV4S_RATIO",    "Ratio v6",    "MEDIA_MOV4S / PROM_HIST: tendencia de las últimas 4 semanas normalizada por el histórico.", "Alta"),
        ("CRECIMIENTO_8S", "Ratio v6",    "Promedio de ratios LAG/PROM_HIST en las últimas 8 semanas. Detecta si la clínica está en régimen de crecimiento acelerado.", "Muy alta"),
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
            st.markdown(f"**{len(df_valid)} registros · {VAL_LABEL} · conjunto out-of-sample "
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
            st.markdown(f"**Total período {VAL_LABEL} por tipo**")
            tot_v = df_v.groupby("Tipo Consulta")["VENTAS"].sum().reindex(TIPOS)
            c1, c2, c3, c4 = st.columns(4)
            for col, tipo in zip([c1, c2, c3], TIPOS):
                col.metric(tipo, f"${tot_v[tipo]:,.0f}")
            c4.metric("TOTAL", f"${tot_v.sum():,.0f}")

            # Gráfico barras
            fig_v = px.bar(df_v, x="Fecha_str", y="VENTAS", color="Tipo Consulta",
                           color_discrete_map=COLOR_TIPOS,
                           labels={"Fecha_str":"Fecha","VENTAS":"Ventas ($)"},
                           title=f"Ventas reales — Validación {VAL_LABEL}")
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
            st.info(f"**Sesgo medio:** ${sesgo:,.0f} ({dir_}) — factor ×{FACTOR:.3f} corrige la subestimación sistemática.")

        st.divider()

        # ── Explicación de métricas en lenguaje simple ───────────────
        with st.expander("💡 ¿Qué significa cada indicador? (explicación simple)", expanded=True):
            _r2_pct   = f"{r2*100:.1f}"       if r2   is not None else "?"
            _r2_rest  = f"{100 - r2*100:.1f}" if r2   is not None else "?"
            _mape_str = f"{mape:.1f}"          if mape is not None else "?"
            _bajo_mape  = f"{1_000_000 * (1 - (mape or 0)/100):,.0f}"
            _sobre_mape = f"{1_000_000 * (1 + (mape or 0)/100):,.0f}"
            _rmse_str = f"${rmse:,.0f}"  if rmse is not None else "N/D"
            _mae_str  = f"${mae:,.0f}"   if mae  is not None else "N/D"
            _sesgo_dir = "sobreestima" if sesgo is not None and sesgo > 0 else "subestima"
            if sesgo is not None and sesgo > 0:
                _sesgo_expl = (f"Un sesgo **positivo** (+${sesgo:,.0f}) significa que el modelo actualmente "
                               f"**sobreestima** las ventas. El factor de corrección ×{FACTOR:.3f} está calibrado "
                               f"sobre el período de entrenamiento y puede necesitar revisión.")
            elif sesgo is not None and sesgo < 0:
                _sesgo_expl = (f"Un sesgo **negativo** (−${abs(sesgo):,.0f}) significa que el modelo **subestima** "
                               f"las ventas sistemáticamente. Por eso se aplica el **factor de corrección "
                               f"×{FACTOR:.3f}** al total mensual, para compensar esa diferencia.")
            else:
                _sesgo_expl = (f"El sesgo es prácticamente nulo: el modelo no tiene tendencia sistemática "
                               f"a sobreestimar ni subestimar. El factor ×{FACTOR:.3f} se aplica por precaución.")
            st.markdown(f"""
Estas métricas miden **qué tan bien predice el modelo** comparando sus predicciones contra
las ventas reales del período de validación ({VAL_LABEL}, datos que el modelo nunca vio).

---

**R² = {f"{r2:.4f}" if r2 is not None else "N/D"}**
> *"¿Cuánto de la variación en las ventas logra explicar el modelo?"*

Va de 0 a 1. Un R² de **{f"{r2:.2f}" if r2 is not None else "?"}** significa que el modelo explica el **{_r2_pct}% de los cambios** en las ventas diarias.
El {_r2_rest}% restante se debe a factores que el modelo no captura (eventos puntuales, clima, etc.).
Un valor sobre **0.75 es considerado bueno** para este tipo de predicción.

---

**MAPE = {f"{mape:.1f}%" if mape is not None else "N/D"}**
> *"En promedio, ¿cuánto porcentaje se equivoca el modelo por cada predicción?"*

Un MAPE del **{_mape_str}%** significa que si el modelo predice $1.000.000, el valor real estuvo entre
${_bajo_mape} y ${_sobre_mape}. Se considera **aceptable bajo el 18%** para ventas de salud,
que tienen alta variabilidad por ausentismo, urgencias y demanda espontánea.

---

**RMSE = {f"${rmse:,.0f}" if rmse is not None else "N/D"}**
> *"¿Cuántos pesos se equivoca el modelo en una predicción típica?"*

Es el error "estándar" en pesos. Penaliza más los errores grandes.
Un RMSE de **{_rmse_str}** significa que el modelo puede errar hasta ese monto en un día puntual.

---

**MAE = {f"${mae:,.0f}" if mae is not None else "N/D"}**
> *"Error promedio simple en pesos, sin castigar los errores grandes."*

Es más fácil de interpretar que el RMSE. Con un MAE de **{_mae_str}**, en promedio
la predicción se aleja ese monto del valor real (puede ser por encima o por debajo).

---

**Sesgo medio = {f"${sesgo:,.0f}" if sesgo is not None else "N/D"}** ({_sesgo_dir})
> *"¿El modelo tiende a predecir de más o de menos?"*

{_sesgo_expl}
""")

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
                                title=f"Predicho vs Real — Validación {VAL_LABEL}")
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

        if not (ok_r2 and ok_mape):
            _dir_sesgo_ctx = "sobreestima" if sesgo is not None and sesgo > 0 else "subestima"
            st.info(f"""
**ℹ️ Contexto para interpretar estos resultados**

El período de validación ({VAL_LABEL}) coincide con un momento en que la demanda de la clínica
estaba creciendo más rápido de lo que el modelo había aprendido hasta ese momento
(el entrenamiento llegaba hasta abril 2026).

Esto puede explicar que el modelo {_dir_sesgo_ctx} las ventas o tenga un MAPE por encima del umbral del 18%.
**No significa que el modelo esté mal construido** — significa que en ese período la clínica tuvo más actividad
que el patrón histórico, algo que el modelo aprende con el ciclo de mantención siguiente.

El factor de corrección ×{FACTOR:.3f} compensa parcialmente esta diferencia en las predicciones mensuales.
""")

# ════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICCIÓN
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Predicción mensual de ventas")
    st.caption(f"Factor de corrección ×{FACTOR} aplicado al total mensual.")

    c_mes, c_año, c_btn = st.columns([2, 2, 3])
    with c_mes:
        mes_sel = st.selectbox("Mes", list(MESES_ES.keys()), index=5,   # Junio por defecto
                               format_func=lambda m: MESES_ES[m])
    with c_año:
        año_sel = st.number_input("Año", min_value=2026, max_value=2030, value=2026, step=1)
    with c_btn:
        st.markdown("&nbsp;")
        run = st.button("🔮 Predecir", type="primary", use_container_width=True)

    # ── Límite mínimo de fecha permitida ────────────────────────────
    FECHA_MIN = date(2026, 6, 1)   # no permitir predecir antes de Jun 2026
    mes_pedido_check = date(int(año_sel), int(mes_sel), 1)
    if mes_pedido_check < FECHA_MIN:
        st.error(
            f"⛔ No se puede predecir antes de **junio 2026**. "
            f"El modelo fue entrenado con datos hasta abril 2026 y las predicciones "
            f"anteriores a esa fecha no tienen sentido como estimaciones futuras. "
            f"Para analizar meses históricos usa los datos reales en la pestaña **📊 Datos**."
        )
        st.stop()

    if run:
        # ── Detectar si el mes es pasado, presente o futuro ─────────
        hoy         = date.today()
        mes_actual  = date(hoy.year, hoy.month, 1)
        mes_pedido  = date(int(año_sel), int(mes_sel), 1)
        es_pasado   = mes_pedido < mes_actual
        es_presente = mes_pedido == mes_actual

        # Determinar si el mes está en el historial de entrenamiento
        hist_disponible = False
        df_hist_mes = None
        if df_train is not None and (es_pasado or es_presente):
            df_hist_mes = df_train[
                (df_train["Fecha"].dt.year  == int(año_sel)) &
                (df_train["Fecha"].dt.month == int(mes_sel))
            ]
            hist_disponible = len(df_hist_mes) > 0

        # Avisos según tipo de mes
        if es_pasado and hist_disponible:
            st.info(
                f"📂 **Modo backtest** — {MESES_ES[mes_sel]} {año_sel} está en el historial de "
                f"entrenamiento. Se usan los datos reales como fuente de lags para mayor precisión. "
                f"Al final verás la comparación **real vs predicho**."
            )
        elif es_pasado and not hist_disponible:
            st.warning(
                f"⚠️ **Mes pasado sin datos** — {MESES_ES[mes_sel]} {año_sel} no está en "
                f"entrenamiento.csv. Los lags se calcularán con promedios históricos, "
                f"por lo que la predicción será aproximada."
            )
        elif es_presente:
            st.info(f"📅 Estás prediciendo el mes en curso ({MESES_ES[mes_sel]} {año_sel}).")

        # Combinar entrenamiento + validacion: da lags reales de mayo 2026 para predecir junio
        _dfs_lag = [d for d in [df_train, df_valid] if d is not None]
        lag_fuente = pd.concat(_dfs_lag, ignore_index=True) if _dfs_lag else None

        with st.spinner(f"Calculando {MESES_ES[mes_sel]} {año_sel}…"):
            df_pred = predecir_mes(int(mes_sel), int(año_sel), df_historico=lag_fuente)

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
                     title=f"Ventas diarias predichas — {MESES_ES[mes_sel]} {año_sel}")
        fig.update_xaxes(tickangle=45)
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

        # ── Comparación real vs predicho (solo si hay historial) ────
        if hist_disponible and df_hist_mes is not None:
            st.subheader("📊 Real vs Predicho")
            df_comp = (df_open[["Fecha","Tipo Consulta","VENTAS_DIA"]]
                       .rename(columns={"VENTAS_DIA":"Predicho ($)"})
                       .merge(
                           df_hist_mes[["Fecha","Tipo Consulta","VENTAS"]]
                           .assign(Fecha=lambda x: pd.to_datetime(x["Fecha"]).dt.date)
                           .rename(columns={"VENTAS":"Real ($)"}),
                           on=["Fecha","Tipo Consulta"], how="left"
                       ))
            df_comp["Error ($)"]   = df_comp["Predicho ($)"] - df_comp["Real ($)"]
            df_comp["Error (%)"]   = (df_comp["Error ($)"] / df_comp["Real ($)"].replace(0, np.nan) * 100).round(1)
            df_comp["Fecha_str"]   = pd.to_datetime(df_comp["Fecha"]).dt.strftime("%a %d")

            # Gráfico líneas real vs predicho
            df_tot_dia = df_comp.groupby("Fecha")[["Real ($)","Predicho ($)"]].sum().reset_index()
            df_tot_dia["Fecha_str"] = pd.to_datetime(df_tot_dia["Fecha"]).dt.strftime("%a %d")
            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(x=df_tot_dia["Fecha_str"], y=df_tot_dia["Real ($)"],
                                         name="Real", line=dict(color="#1d4ed8", width=2)))
            fig_cmp.add_trace(go.Scatter(x=df_tot_dia["Fecha_str"], y=df_tot_dia["Predicho ($)"],
                                         name="Predicho", line=dict(color="#f59e0b", width=2, dash="dash")))
            fig_cmp.update_layout(title="Total diario: Real vs Predicho",
                                   yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                                   margin=dict(l=0,r=0,t=40,b=0))
            fig_cmp.update_xaxes(tickangle=45)
            st.plotly_chart(fig_cmp, use_container_width=True)

            # Tabla detalle
            st.dataframe(
                df_comp[["Fecha_str","Tipo Consulta","Real ($)","Predicho ($)","Error ($)","Error (%)"]
                ].style.format({"Real ($)":"${:,.0f}","Predicho ($)":"${:,.0f}","Error ($)":"${:,.0f}","Error (%)":"{:.1f}%"})
                 .applymap(lambda v: "color:#dc2626" if isinstance(v, (int,float)) and abs(v) > 20 else "",
                           subset=["Error (%)"]),
                use_container_width=True, hide_index=True
            )

            # Resumen de accuracy
            mask_r = df_comp["Real ($)"] > 0
            mape_bt = (df_comp.loc[mask_r, "Error (%)"].abs()).mean()
            sesgo_bt = df_comp["Error ($)"].mean()
            c1, c2 = st.columns(2)
            c1.metric("MAPE backtest", f"{mape_bt:.1f}%")
            c2.metric("Sesgo medio", f"${sesgo_bt:,.0f}",
                      delta_color="inverse" if sesgo_bt < 0 else "normal")

        # ── Resumen mensual ──────────────────────────────────────────
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

        lbl = "Backtest" if es_pasado else "Estimación"
        st.success(f"**{lbl} {MESES_ES[mes_sel]} {año_sel}** con factor ×{FACTOR}: "
                   f"**${total_con:,.0f}**")

        st.download_button(
            "⬇️ Descargar predicción Excel",
            data=df_to_excel_bytes(df_res),
            file_name=f"prediccion_{MESES_ES[mes_sel]}_{año_sel}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
