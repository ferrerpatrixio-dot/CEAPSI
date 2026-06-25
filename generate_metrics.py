"""
Ejecutar después de correr el notebook para generar model_metrics.json.
Calcula R², MAPE, RMSE, MAE y sesgo directamente desde validacion.csv
usando entrenamiento.csv como fuente de lags.
"""
import json, pickle, sys, subprocess
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.stdout.reconfigure(encoding='utf-8')

ESCALA_MM  = 1_000_000
TIPOS      = ['Adultos', 'Infantil', 'Teleconsulta']
TIPO_COD   = {'Adultos': 0, 'Infantil': 1, 'Teleconsulta': 2}

# ── Cargar artefactos ────────────────────────────────────────────────────────
model = xgb.XGBRegressor()
model.load_model('xgb_ventas_clinica_v5.json')
FEATURES_V5  = pickle.load(open('features_v5.pkl', 'rb'))  # contiene features v6 (13)
PROM_HIST    = pickle.load(open('prom_hist_dict.pkl',       'rb'))
PROM_CANT    = pickle.load(open('prom_cant_dict.pkl',       'rb'))
FERIADOS     = pickle.load(open('feriados_set.pkl',         'rb'))
FECHA_INICIO = pickle.load(open('fecha_inicio.pkl',         'rb'))
VAC_INV      = pickle.load(open('vacaciones_invierno.pkl',  'rb'))

# ── Cargar datos ─────────────────────────────────────────────────────────────
try:
    df_val = pd.read_csv('validacion.csv', encoding='utf-8-sig')
    df_val['Fecha'] = pd.to_datetime(df_val['Fecha'])
    print(f'Validacion   : {len(df_val)} filas '
          f'({df_val["Fecha"].min().date()} → {df_val["Fecha"].max().date()})')
except FileNotFoundError:
    print('ERROR: validacion.csv no encontrado.')
    sys.exit(1)

try:
    df_train = pd.read_csv('entrenamiento.csv', encoding='utf-8-sig')
    df_train['Fecha'] = pd.to_datetime(df_train['Fecha'])
    print(f'Entrenamiento: {len(df_train)} filas '
          f'({df_train["Fecha"].min().date()} → {df_train["Fecha"].max().date()})')
except FileNotFoundError:
    df_train = None
    print('WARN: entrenamiento.csv no encontrado — lags usarán promedios históricos')

# ── Índice rápido de lags (train + validacion real acumulada) ─────────────────
# hist_v empieza con entrenamiento y se enriquece con datos reales de validacion
# a medida que se procesan en orden — igual que predecir_mes con cache_v.
hist_v, hist_c = {}, {}
if df_train is not None:
    for _, row in df_train.iterrows():
        fd = row['Fecha'].date()
        tc = TIPO_COD.get(row['Tipo Consulta'], 0)
        hist_v[(fd, tc)] = row['VENTAS']       / ESCALA_MM
        hist_c[(fd, tc)] = row['CANT_VENTAS']

def _v(fd, tc):
    return hist_v.get((fd, tc), PROM_HIST.get((fd.month, tc), 0.0) / ESCALA_MM)

def _c(fd, tc):
    return hist_c.get((fd, tc), PROM_CANT.get((fd.month, tc), 0.0))

def calc_vacaciones(mes, dia, año):
    if mes == 12 and dia >= 26: return 1
    if mes in (1, 2):           return 1
    if mes == 3 and dia == 1:   return 1
    if año in VAC_INV:
        fi, ff = VAC_INV[año]
        if fi <= date(año, mes, dia) <= ff: return 1
    return 0

# ── Construir features procesando dias en orden (acumulando lags reales) ─────
# Ordenar por fecha para que los lags del día anterior sean siempre correctos
df_val = df_val.sort_values(['Fecha', 'Tipo Consulta']).reset_index(drop=True)

filas_X, y_real = [], []
pvr = []

for _, row in df_val.iterrows():
    fd    = row['Fecha'].date()
    tipo  = row['Tipo Consulta']
    tc    = TIPO_COD.get(tipo, 0)
    mes_  = fd.month
    dia_  = fd.day
    año_  = fd.year
    td    = timedelta

    l7    = _v(fd - td(7),  tc)
    l14   = _v(fd - td(14), tc)
    l21   = _v(fd - td(21), tc)
    l28   = _v(fd - td(28), tc)
    mov4s = (l7 + l14 + l21 + l28) / 4.0
    prom_h = PROM_HIST.get((mes_, tc), 0.0) / ESCALA_MM

    # CRECIMIENTO_8S
    _crec_vals = []
    for _d in [7, 14, 21, 28, 35, 42, 49, 56]:
        _lag_fd  = fd - td(_d)
        _lag_val = _v(_lag_fd, tc)
        _ph_lag  = PROM_HIST.get((_lag_fd.month, tc), 0.0) / ESCALA_MM
        if _lag_val > 0 and _ph_lag > 0:
            _crec_vals.append(_lag_val / _ph_lag)
    crec8s = float(sum(_crec_vals) / len(_crec_vals)) if _crec_vals else 1.0

    filas_X.append({
        'DIASEM':         fd.isoweekday(),
        'tipo_cod':       tc,
        'A_FERIADO':      1 if fd in FERIADOS else 0,
        'TENDENCIA':      (pd.Timestamp(fd) - FECHA_INICIO).days,
        'PROM_HIST':      prom_h,
        'LAG7':           l7,
        'MEDIA_MOV4S':    mov4s,
        'VACACIONES':     calc_vacaciones(mes_, dia_, año_),
        'CANT_LAG7':      _c(fd - td(7),  tc),
        'CANT_MOV4S':     (_c(fd-td(7),tc)+_c(fd-td(14),tc)+
                          _c(fd-td(21),tc)+_c(fd-td(28),tc)) / 4.0,
        'LAG_RATIO':      l7 / prom_h if prom_h > 0 else 1.0,
        'MOV4S_RATIO':    mov4s / prom_h if prom_h > 0 else 1.0,
        'CRECIMIENTO_8S': crec8s,
    })
    y_real.append(row['VENTAS'] / ESCALA_MM)

    # Agregar dato real al índice para que los días siguientes lo usen como lag
    hist_v[(fd, tc)] = row['VENTAS'] / ESCALA_MM
    hist_c[(fd, tc)] = row['CANT_VENTAS']

X_val   = pd.DataFrame(filas_X)[FEATURES_V5]   # 13 features v6
y_real  = np.array(y_real)
y_pred  = model.predict(X_val)

# Convertir a pesos para métricas interpretables
y_real_pesos = y_real * ESCALA_MM
y_pred_pesos = y_pred * ESCALA_MM

mask   = y_real_pesos > 500_000   # excluye dias con ventas anomalamente bajas (<500K)
r2_v   = float(r2_score(y_real_pesos, y_pred_pesos))
rmse_v = float(np.sqrt(mean_squared_error(y_real_pesos, y_pred_pesos)))
mae_v  = float(mean_absolute_error(y_real_pesos, y_pred_pesos))
mape_v = float(np.mean(np.abs((y_real_pesos[mask] - y_pred_pesos[mask]) / y_real_pesos[mask])) * 100)
# Sesgo sobre misma máscara que MAPE/factor para consistencia (excluye outliers extremos)
sesgo  = float((y_pred_pesos[mask] - y_real_pesos[mask]).mean())

# Factor de corrección: real / predicho (solo filas con ventas > 500K para excluir outliers)
# >1 si el modelo subestima, <1 si sobreestima
_sum_real = float(y_real_pesos[mask].sum())
_sum_pred = float(y_pred_pesos[mask].sum())
factor_corr = round(_sum_real / _sum_pred, 4) if _sum_pred > 0 else 1.0

print(f'\nMétricas validación ({df_val["Fecha"].min().date()} → {df_val["Fecha"].max().date()}):')
print(f'  R²   = {r2_v:.4f}')
print(f'  MAPE = {mape_v:.2f}%')
print(f'  RMSE = ${rmse_v:,.0f}')
print(f'  MAE  = ${mae_v:,.0f}')
print(f'  Sesgo medio = ${sesgo:,.0f}')
print(f'  Factor corrección = {factor_corr:.4f} ({"subestima→multiplica" if factor_corr > 1 else "sobreestima→reduce"})')

# pred_vs_real para scatter plot
for i, row in enumerate(df_val.itertuples()):
    pvr.append({
        'real': round(float(y_real_pesos[i]), 0),
        'pred': round(float(y_pred_pesos[i]), 0),
        'tipo': row._2,   # Tipo Consulta
    })

metrics = {
    'r2':              r2_v,
    'mape':            mape_v,
    'rmse':            rmse_v,
    'mae':             mae_v,
    'sesgo_medio':     sesgo,
    'factor_correccion': factor_corr,
    'training_info': {
        'fecha_min_train': str(df_train['Fecha'].min().date()) if df_train is not None else 'Abr 2023',
        'fecha_max_train': str(df_train['Fecha'].max().date()) if df_train is not None else 'Abr 2026',
        'n_train':         len(df_train) if df_train is not None else 2761,
        'fecha_min_test':  str(df_val['Fecha'].min().date()),
        'fecha_max_test':  str(df_val['Fecha'].max().date()),
        'n_test':          len(df_val),
        'n_opt':           model.get_booster().num_boosted_rounds(),
    },
    'pred_vs_real': pvr,
}

with open('model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print('\nOK  model_metrics.json actualizado.')

# ── Historial de versiones (append-only) ─────────────────────────────────────
try:
    _git_commit = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    _git_commit = 'N/A'

try:
    _git_branch = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    _git_branch = 'N/A'

try:
    _xgb_params = model.get_xgb_params()
except Exception:
    _xgb_params = {}

# Leer historial existente
_hist_path = 'model_versions.json'
try:
    with open(_hist_path, encoding='utf-8') as _f:
        _historial = json.load(_f)
except (FileNotFoundError, json.JSONDecodeError):
    _historial = []

_version_num = (_historial[-1]['version'] + 1) if _historial else 1

_entrada = {
    'version':    _version_num,
    'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M'),
    'git_commit': _git_commit,
    'git_branch': _git_branch,
    'modelo': {
        'archivo':           'xgb_ventas_clinica_v5.json',
        'n_estimadores_opt': int(model.get_booster().num_boosted_rounds()),
        'n_features':        len(FEATURES_V5),
        'features':          list(FEATURES_V5),
        'params_xgb': {
            k: v for k, v in _xgb_params.items()
            if k in ('n_estimators','learning_rate','max_depth',
                     'min_child_weight','subsample','colsample_bytree',
                     'reg_alpha','reg_lambda')
        },
    },
    'datos': {
        'entrenamiento': {
            'archivo':    'entrenamiento.csv',
            'fecha_min':  str(df_train['Fecha'].min().date()) if df_train is not None else None,
            'fecha_max':  str(df_train['Fecha'].max().date()) if df_train is not None else None,
            'n_filas':    int(len(df_train)) if df_train is not None else None,
        },
        'validacion': {
            'archivo':    'validacion.csv',
            'fecha_min':  str(df_val['Fecha'].min().date()),
            'fecha_max':  str(df_val['Fecha'].max().date()),
            'n_filas':    int(len(df_val)),
        },
    },
    'metricas': {
        'r2':               round(r2_v,    4),
        'mape_pct':         round(mape_v,  2),
        'rmse':             round(rmse_v,  0),
        'mae':              round(mae_v,   0),
        'sesgo_medio':      round(sesgo,   0),
        'factor_correccion':round(factor_corr, 4),
    },
}

_historial.append(_entrada)
with open(_hist_path, 'w', encoding='utf-8') as _f:
    json.dump(_historial, _f, ensure_ascii=False, indent=2)
print(f'OK  model_versions.json — versión {_version_num} registrada.')
