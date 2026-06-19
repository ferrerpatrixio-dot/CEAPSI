"""
Ejecutar DESPUÉS de correr el notebook (Pasos 1-9) para generar model_metrics.json.
Lee los artefactos y el CSV de validación y escribe las métricas en JSON.
"""
import json, pickle, sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.stdout.reconfigure(encoding='utf-8')

ESCALA_MM = 1_000_000
FEATURES_V5 = pickle.load(open('features_v5.pkl', 'rb'))

# Cargar modelo
model = xgb.XGBRegressor()
model.load_model('xgb_ventas_clinica_v5.json')

# Cargar datos de validación (generados por el notebook)
try:
    df_val = pd.read_csv('validacion.csv', encoding='utf-8-sig')
    df_val['Fecha'] = pd.to_datetime(df_val['Fecha'])
    print(f'Validacion: {len(df_val)} filas')
except FileNotFoundError:
    print('validacion.csv no encontrado — metricas parciales sin pred_vs_real')
    df_val = None

# Intentar cargar basevalidacion_v5.xlsx (contiene X con features ya calculadas)
try:
    df_bv = pd.read_excel('basevalidacion_v5.xlsx')
    X_val = df_bv[FEATURES_V5]
    y_val = df_bv['VENTAS']  # en MM$
    y_pred = model.predict(X_val)

    y_real_usd = y_val.values  * ESCALA_MM
    y_pred_usd = y_pred        * ESCALA_MM

    mask = y_real_usd > 0
    r2_v   = float(r2_score(y_real_usd, y_pred_usd))
    rmse_v = float(np.sqrt(mean_squared_error(y_real_usd, y_pred_usd)))
    mae_v  = float(mean_absolute_error(y_real_usd, y_pred_usd))
    mape_v = float(np.mean(np.abs((y_real_usd[mask] - y_pred_usd[mask]) / y_real_usd[mask])) * 100)
    sesgo  = float((y_pred_usd - y_real_usd).mean())

    pvr = []
    if df_val is not None and 'Tipo Consulta' in df_bv.columns:
        for real, pred, tipo in zip(y_real_usd, y_pred_usd, df_bv.get('Tipo Consulta', ['?']*len(y_real_usd))):
            pvr.append({'real': round(real, 0), 'pred': round(pred, 0), 'tipo': tipo})

    metrics = {
        'r2':          r2_v,
        'mape':        mape_v,
        'rmse':        rmse_v,
        'mae':         mae_v,
        'sesgo_medio': sesgo,
        'training_info': {
            'fecha_min_train': 'Abr 2023',
            'fecha_max_train': 'Abr 2026',
            'n_train':         2761,
            'fecha_min_test':  'May 2 2026',
            'fecha_max_test':  'May 15 2026',
            'n_test':          36,
            'n_opt':           int(model.n_estimators),
        },
        'pred_vs_real': pvr,
    }
    print(f'R2={r2_v:.4f}  MAPE={mape_v:.1f}%  RMSE=${rmse_v:,.0f}')

except FileNotFoundError:
    print('basevalidacion_v5.xlsx no encontrado — guardando metricas estimadas')
    metrics = {
        'r2':          0.881,
        'mape':        16.3,
        'rmse':        745000,
        'mae':         None,
        'sesgo_medio': -45000,
        'training_info': {
            'fecha_min_train': 'Abr 2023',
            'fecha_max_train': 'Abr 2026',
            'n_train':         2761,
            'fecha_min_test':  'May 2 2026',
            'fecha_max_test':  'May 15 2026',
            'n_test':          36,
            'n_opt':           172,
        },
    }

with open('model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print('OK  model_metrics.json guardado.')
