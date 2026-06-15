import streamlit as st
import pandas as pd
import requests
import io
import re
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime

# ==============================================================================
# --- CONFIGURACIÓN DE LA PÁGINA ---
# ==============================================================================
st.set_page_config(page_title="Dashboard Monitor Aedes", layout="wide", page_icon="🦟")

st.markdown("""
    <style>
    .main-title { font-size:38px !important; font-weight: bold; color: #2E4053; margin-bottom: 5px; }
    .subtitle { font-size:18px !important; color: #5D6D7E; margin-bottom: 25px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 Sistema de Monitoreo Biológico - Aedes aegypti</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Análisis de datos acústicos e inferencia de IA en tiempo real</p>', unsafe_allow_html=True)

# ==============================================================================
# --- CONFIGURACIÓN GITHUB ---
# ==============================================================================
GITHUB_USER   = "Loren-Arq"
GITHUB_REPO   = "mi_proyecto_esp32"
GITHUB_BRANCH = "main"
GITHUB_FOLDER = "datos/excel"   # carpeta dentro del repo

# ==============================================================================
# --- CARGA DESDE GITHUB ---
# ==============================================================================
# ==============================================================================
# --- CONFIGURACIÓN GITHUB ---
# ==============================================================================
GITHUB_USER   = "Loren-Arq"
GITHUB_REPO   = "mi_proyecto_esp32"
GITHUB_BRANCH = "main"
GITHUB_FOLDER = "datos/excel"   # Carpeta dentro del repo

# ==============================================================================
# --- CARGA DESDE GITHUB (CORREGIDA) ---
# ==============================================================================
@st.cache_data(ttl=60)  # Refresca cada 60 segundos
def cargar_datos_reportes():
    # SOLUCIÓN 1: URL de API oficial corregida
    api_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{GITHUB_FOLDER}?ref={GITHUB_BRANCH}"
    
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(api_url, headers=headers, timeout=10)
    except Exception as e:
        st.error(f"❌ Error de red al conectar con GitHub: {e}")
        return pd.DataFrame()

    if resp.status_code != 200:
        st.error(f"❌ GitHub respondió con código: {resp.status_code}")
        st.info("Verifica que el repositorio sea PÚBLICO y que la ruta exista exactamente igual.")
        st.code(resp.text[:300], language="text")
        return pd.DataFrame()

    try:
        todos = resp.json()
    except Exception as e:
        st.error(f"❌ Error al procesar la respuesta (No es un JSON válido): {e}")
        return pd.DataFrame()

    if not isinstance(todos, list):
        st.warning("⚠️ La API de GitHub no devolvió una lista de archivos válida.")
        return pd.DataFrame()

    # SOLUCIÓN 3: Expresión regular flexible para aceptar 'reporte_...', 'Reporte_Aedes_...', etc.
    patron = re.compile(r".*reporte.*(\d{4}-\d{2}-\d{2})\.xlsx", re.IGNORECASE)
    
    archivos = [f for f in todos if isinstance(f, dict) and "name" in f and patron.match(f["name"])]

    if not archivos:
        return pd.DataFrame()

    lista_dfs = []
    for f in archivos:
        try:
            # SOLUCIÓN 2: Usar la URL de descarga directa provista por GitHub
            raw_url = f["download_url"]
            r = requests.get(raw_url, headers=headers, timeout=10)
            r.raise_for_status()

            df = pd.read_excel(io.BytesIO(r.content))
            
            df.columns = [str(c).strip() for c in df.columns]

            # Extraer la fecha dinámicamente con el nuevo patrón
            match = patron.search(f["name"])
            fecha_archivo = datetime.strptime(match.group(1), "%Y-%m-%d").date() if match else datetime.now().date()

            df['Fecha_Registro'] = pd.to_datetime(fecha_archivo)
            df['Mes'] = df['Fecha_Registro'].dt.strftime('%Y-%m ( %B )')

            col_prob = [c for c in df.columns if 'Probabilidad' in c or 'Prob' in c]
            col_frec = [c for c in df.columns if 'Frecuencia' in c or 'Frec' in c]
            col_amp = [c for c in df.columns if 'Amplitud' in c or 'Amp' in c]

            frec_name = col_frec[0] if col_frec else 'Frecuencia (Hz)'
            amp_name = col_amp[0] if col_amp else 'Amplitud (dB)'
            
            df['Frec_Num'] = pd.to_numeric(df[frec_name], errors='coerce').fillna(0)
            df['Amp_Num']  = pd.to_numeric(df[amp_name], errors='coerce').fillna(0)
            
            if col_prob:
                df['Prob_Num'] = pd.to_numeric(df[col_prob[0]], errors='coerce').fillna(0)
                if df['Prob_Num'].max() > 1.0:
                    df['Prob_Num'] = df['Prob_Num'] / 100.0
            else:
                df['Prob_Num'] = 0.0

            lista_dfs.append(df)
        except Exception as e:
            st.warning(f"⚠️ Error leyendo {f['name']}: {e}")
            continue

    if lista_dfs:
        return pd.concat(lista_dfs, ignore_index=True)
    return pd.DataFrame()




# Botón manual de refresco + carga automática
# ==============================================================================
# --- SISTEMA DE REFRESCO AUTOMÁTICO REAL ---
# ==============================================================================

# Definimos un fragmento que fuerza la recarga del script completo cada 60 segundos
@st.fragment(run_every=60)
def despachador_de_tiempo():
    # Este componente invisible obliga a la pantalla a actualizarse sola
    pass

# Ejecutar el temporizador en segundo plano
despachador_de_tiempo()

# Botón manual de refresco mejorado
col_refresh, _ = st.columns([3, 7])
with col_refresh:
    if st.button("🔄 Forzar actualización manual"):
        st.cache_data.clear()
        st.rerun()

# Carga de datos global
df_global = cargar_datos_reportes()


if df_global.empty:
    st.warning("⚠️ No se encontraron archivos 'Reporte_Aedes_*.xlsx' en el repositorio GitHub.")
    st.caption(f"Buscando en: github.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_FOLDER}")
    st.stop()

# ==============================================================================
# --- FILTROS LATERALES ---
# ==============================================================================
st.sidebar.header("🎛️ Filtros Globales")
lista_meses = sorted(df_global['Mes'].unique())
mes_seleccionado = st.sidebar.selectbox("Seleccionar Mes de Análisis:", ["Todos los meses"] + lista_meses)

if mes_seleccionado != "Todos los meses":
    df_filtrado = df_global[df_global['Mes'] == mes_seleccionado]
else:
    df_filtrado = df_global.copy()

# Mostrar hora de última actualización
st.sidebar.markdown("---")
st.sidebar.caption(f"🕒 Última carga: {datetime.now().strftime('%H:%M:%S')}")
st.sidebar.caption("Los datos se refrescan automáticamente cada 60 segundos.")

# ==============================================================================
# --- FILA 1: KPIs ---
# ==============================================================================
total_detecciones = len(df_filtrado)
positivos_aedes   = len(df_filtrado[df_filtrado['Prob_Num'] > 0.5])
freq_promedio     = df_filtrado['Frec_Num'].mean() if total_detecciones > 0 else 0
amp_promedio      = df_filtrado['Amp_Num'].mean()  if total_detecciones > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="🎙️ Total Eventos Capturados", value=f"{total_detecciones}")
with kpi2:
    pct = (positivos_aedes / total_detecciones) if total_detecciones > 0 else 0
    st.metric(label="🦟 Positivos Aedes (IA > 50%)", value=f"{positivos_aedes}",
              delta=f"{pct:.1%} del total", delta_color="inverse")
with kpi3:
    st.metric(label="🎼 Frecuencia Fundamental Promedio", value=f"{freq_promedio:.1f} Hz")
with kpi4:
    st.metric(label="🔊 Presión Sonora Promedio", value=f"{amp_promedio:.1f} dB")

st.markdown("---")

# ==============================================================================
# --- FILA 2: MAPA + TENDENCIA ---
# ==============================================================================
col_izq, col_der = st.columns([1, 1])

with col_izq:
    st.subheader("📍 Ubicación Geográfica del Sensor")
    lat_sensor = 14.58849
    lon_sensor = -90.5533

    m = folium.Map(location=[lat_sensor, lon_sensor], zoom_start=17, tiles="OpenStreetMap")

    popup_text = f"""
    <div style='font-family: Arial, sans-serif; width: 180px;'>
        <h4 style='margin:0 0 5px 0; color:#C0392B;'>Dispositivo IoT #1</h4>
        <b>Estado:</b> Activo Escuchando<br>
        <b>Muestras:</b> {total_detecciones}<br>
        <b>Positivos:</b> {positivos_aedes}<br>
        <small style='color:gray;'>Lat: {lat_sensor}<br>Lon: {lon_sensor}</small>
    </div>
    """
    folium.Marker(
        [lat_sensor, lon_sensor],
        popup=folium.Popup(popup_text, max_width=250),
        tooltip="Dispositivo de Monitoreo Biológico",
        icon=folium.Icon(color="red", icon="microchip", prefix="fa")
    ).add_to(m)

    st_folium(m, width="100%", height=380, returned_objects=[])

with col_der:
    st.subheader("📈 Histórico Evolutivo / Tendencia Mensual")

    df_mensual = df_global.groupby('Mes').agg(
        Total_Eventos=('Evento', 'count'),
        Positivos_Aedes=('Prob_Num', lambda x: (x > 0.5).sum())
    ).reset_index()

    fig_lineas = go.Figure()
    fig_lineas.add_trace(go.Bar(
        x=df_mensual['Mes'], y=df_mensual['Total_Eventos'],
        name='Total Ruidos Capturados', marker_color='#AED6F1'
    ))
    fig_lineas.add_trace(go.Scatter(
        x=df_mensual['Mes'], y=df_mensual['Positivos_Aedes'],
        name='Casos Confirmados Aedes', mode='lines+markers',
        line=dict(color='#E74C3C', width=3), marker=dict(size=8)
    ))
    fig_lineas.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode='group', plot_bgcolor='white'
    )
    fig_lineas.update_yaxes(gridcolor='#F2F4F4')
    st.plotly_chart(fig_lineas, use_container_width=True)

# ==============================================================================
# --- FILA 3: ALERTAS + TABLA ---
# ==============================================================================
st.markdown("---")
# Definimos las columnas de la tercera fila
col_analisis_1, col_analisis_2 = st.columns([4, 6])

with col_analisis_1:
    st.subheader("🦟 Clasificación de Alertas por Nivel de Confianza")

    def clasificar_alerta(prob):
        if prob >= 0.75:
            return "🔴 ALTA (Aedes Confirmado)"
        elif prob >= 0.40:
            return "🟡 MEDIA (Mosquito Sospechoso)"
        else:
            return "🟢 BAJA (Ruido Ambiental / Descartado)"

    # Evitamos advertencias de SettingWithCopyWarning
    df_filtrado = df_filtrado.copy()
    df_filtrado['Categoria_Alerta'] = df_filtrado['Prob_Num'].apply(clasificar_alerta)

    conteo_alertas = df_filtrado['Categoria_Alerta'].value_counts().reset_index()
    conteo_alertas.columns = ['Nivel de Alerta', 'Cantidad de Audios']

    colores_semaforo = {
        "🔴 ALTA (Aedes Confirmado)":          "#E74C3C",
        "🟡 MEDIA (Mosquito Sospechoso)":       "#F4D03F",
        "🟢 BAJA (Ruido Ambiental / Descartado)": "#2ECC71"
    }

    fig_barras = px.bar(
        conteo_alertas,
        x='Nivel de Alerta', y='Cantidad de Audios',
        color='Nivel de Alerta',
        color_discrete_map=colores_semaforo,
        text_auto=True
    )
    fig_barras.update_layout(
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor='white', showlegend=False
    )
    fig_barras.update_yaxes(gridcolor='#F2F4F4')
    
    # Renderizado limpio del gráfico
    st.plotly_chart(fig_barras)

with col_analisis_2:
    st.subheader("📋 Registros Analizados")

    # Columnas exactas mapeadas desde tu archivo Excel físico
    cols_tabla = ['Evento', 'Fecha', 'Hora', 'Distancia (mm)',
                  'Frecuencia (Hz)', 'Amplitud (dB)', 'Probabilidad (%)',
                  'Armónicos', 'Latencia Red (ms)', 'Latencia CNN', 'Alerta']

    # Se limpian los nombres por si tienen espacios adicionales
    df_filtrado.columns = [str(c).strip() for c in df_filtrado.columns]
    cols_disponibles = [c for c in cols_tabla if c in df_filtrado.columns]

    if 'Evento' in df_filtrado.columns:
        df_mostrar = df_filtrado[cols_disponibles].sort_values(by='Evento', ascending=False)
    else:
        df_mostrar = df_filtrado[cols_disponibles]

    st.dataframe(df_mostrar, height=350)

