import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from pathlib import Path

# ======================================================================
# CONFIGURAÇÃO BÁSICA DA PÁGINA
# ======================================================================
st.set_page_config(
    page_title="Trigona spinipes | Monitoramento Climático",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# ESTILOS CUSTOMIZADOS (CSS)
# ======================================================================
st.markdown(
    """
<style>
/* Fonte global */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header principal */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 18px;
    margin-bottom: 2rem;
    box-shadow: 0 15px 35px rgba(0,0,0,0.25);
}

.main-header h1 {
    color: #ffffff;
    font-size: 2.6rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: 0.04em;
}

.main-header p {
    color: rgba(255,255,255,0.95);
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* Cards de KPI */
.metric-card {
    background: radial-gradient(circle at top left, #ffffff 0%, #f5f7fb 40%, #d9e2ff 100%);
    padding: 1.2rem 1.3rem;
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    border: 1px solid rgba(255,255,255,0.6);
    backdrop-filter: blur(6px);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(0,0,0,0.18);
}

.metric-label {
    font-size: 0.85rem;
    color: #555;
    margin-bottom: 0.15rem;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #2c3e50;
    margin-bottom: 0.1rem;
}

.metric-sub {
    font-size: 0.8rem;
    color: #777;
}

/* Tabs customizadas */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background-color: #f6f7fb;
    padding: 0.4rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
}

.stTabs [data-baseweb="tab"] {
    background-color: #ffffff;
    border-radius: 10px;
    padding: 0.6rem 1.3rem;
    font-weight: 600;
    font-size: 0.9rem;
    color: #555 !important;
    border: 1px solid transparent;
    transition: all 0.25s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    border-color: #d0d4ff;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #ffffff !important;
    box-shadow: 0 8px 22px rgba(102, 126, 234, 0.35);
}

/* Caixas de informação */
.info-box {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    padding: 1.2rem 1.4rem;
    border-radius: 14px;
    border-left: 5px solid #ff6b6b;
    margin: 1rem 0;
    color: #4a2c2c;
}

.warning-box {
    background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
    padding: 1.2rem 1.4rem;
    border-radius: 14px;
    border-left: 5px solid #5f27cd;
    margin: 1rem 0;
    color: #2d114f;
}

/* Botões */
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    transition: all 0.25s ease;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(102, 126, 234, 0.45);
}

/* Divisor */
hr {
    margin: 2rem 0;
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #667eea, transparent);
}
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================================
# FUNÇÕES DE CARREGAMENTO E CÁLCULO
# ======================================================================

BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "data"

@st.cache_data
def carregar_dados_ocorrencias():
    """
    Carrega o CSV de ocorrências com separador por tabulação ('\\t')
    e faz uma limpeza básica.
    """
    try:
        # tenta no caminho data/ocorrencias.csv
        caminho1 = DATA_DIR / "ocorrencias.csv"
        caminho2 = Path("ocorrencias.csv")

        if caminho1.exists():
            df = pd.read_csv(caminho1, sep="\t")
        elif caminho2.exists():
            df = pd.read_csv(caminho2, sep="\t")
        else:
            return None

        df = df.dropna(
            subset=["year", "decimalLatitude", "decimalLongitude"]
        )
        df = df[(df["year"] >= 1900) & (df["year"] <= 2100)]
        df["year"] = df["year"].astype(int)

        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados de ocorrências: {e}")
        return None


@st.cache_data
def calcular_kpis(df: pd.DataFrame):
    """Calcula alguns KPIs principais a partir do dataframe de ocorrências."""
    if df is None or df.empty:
        return None

    anos_counts = df["year"].value_counts()
    ano_pico = int(anos_counts.idxmax())
    qtd_pico = int(anos_counts.max())

    # Conta apenas estados onde o país é Brasil (BR)
    if "countryCode" in df.columns and "stateProvince" in df.columns:
        estados_br_count = df[df["countryCode"] == "BR"]["stateProvince"].nunique()
    elif "stateProvince" in df.columns:
        estados_br_count = df["stateProvince"].nunique()
    else:
        estados_br_count = 0

    return {
        "total_registros": int(len(df)),
        "anos_cobertura": int(df["year"].max() - df["year"].min()),
        "lat_media": float(df["decimalLatitude"].mean()),
        "lon_media": float(df["decimalLongitude"].mean()),
        "estados_unicos": int(estados_br_count),
        "primeiro_registro": int(df["year"].min()),
        "ultimo_registro": int(df["year"].max()),
        "pico_registros_ano": ano_pico,
        "pico_registros_qtd": qtd_pico,
    }


@st.cache_data
def analisar_perfil_latitudinal(arquivo_tif: Path):
    """
    Lê o GeoTIFF de adequabilidade e calcula KPIs e perfil latitudinal.
    """
    try:
        if not arquivo_tif.exists():
            return None

        with rasterio.open(arquivo_tif) as src:
            data = src.read(1)
            nodata = src.nodata

            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            else:
                data = np.where(data == -9999, np.nan, data)

            valid_pixels = data[~np.isnan(data)]
            if valid_pixels.size == 0:
                return None

            # KPIs de adequabilidade
            pct_alta = (np.sum(valid_pixels > 0.6) / valid_pixels.size) * 100
            pct_media = (
                np.sum((valid_pixels > 0.4) & (valid_pixels <= 0.6))
                / valid_pixels.size
            ) * 100
            pct_baixa = (np.sum(valid_pixels <= 0.4) / valid_pixels.size) * 100
            adequab_media = float(np.nanmean(valid_pixels))

            # Perfil latitudinal
            with np.errstate(divide="ignore", invalid="ignore"):
                lat_means = np.nanmean(data, axis=1)

            height = data.shape[0]
            bounds = src.bounds
            lats = np.linspace(bounds.top, bounds.bottom, height)

            mask_valid = ~np.isnan(lat_means)
            lat_means_valid = lat_means[mask_valid]
            lats_valid = lats[mask_valid]

            if lats_valid.size == 0:
                return None

            idx_pico = int(np.argmax(lat_means_valid))
            lat_pico = float(lats_valid[idx_pico])
            valor_pico = float(lat_means_valid[idx_pico])

            return {
                "lats": lats_valid,
                "means": lat_means_valid,
                "lat_pico": lat_pico,
                "valor_pico": valor_pico,
                "pct_alta": pct_alta,
                "pct_media": pct_media,
                "pct_baixa": pct_baixa,
                "adequabilidade_media": adequab_media,
            }
    except Exception as e:
        st.error(f"Erro ao analisar cenário futuro ({arquivo_tif.name}): {e}")
        return None

def colorize_raster_interactive(data):
    """
    Aplica um colormap personalizado (Laranja Forte -> Amarelo -> Verde)
    aos dados raster normalizados para visualização no Folium.
    """
    valid_data = data[~np.isnan(data)]
    if valid_data.size == 0:
        return np.zeros((data.shape[0], data.shape[1], 4))
    
    min_val, max_val = valid_data.min(), valid_data.max()
    
    if max_val == min_val:
        norm_data = np.zeros_like(data)
    else:
        norm_data = (data - min_val) / (max_val - min_val)
    
    # DEFINIÇÃO DAS CORES (Hex Codes)
    # Laranja Forte (#d73027) -> Laranja Suave (#fc8d59) -> Amarelo (#fee08b) 
    # -> Verde Claro (#d9ef8b) -> Verde Escuro (#1a9850)
    colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
    
    # Cria o colormap personalizado
    cmap = LinearSegmentedColormap.from_list("custom_orange_green", colors)
    
    colored_data = cmap(norm_data)
    # Define alpha=0 onde for NaN (transparente)
    colored_data[np.isnan(data), 3] = 0
    return colored_data

def get_bounds(src):
    """Obtém os limites do raster para projeção no Folium."""
    return [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]

# ======================================================================
# CARREGAMENTO INICIAL
# ======================================================================

df_ocorrencias = carregar_dados_ocorrencias()
kpis = calcular_kpis(df_ocorrencias)

# Arquivos de previsão gerados pelo script abelhas.py
PREVISOES_DIR = DATA_DIR / "previsoes_futuras"
arquivos_predicao = {
    "2021-2040": PREVISOES_DIR
    / "previsao_trigona_wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2021-2040.tif",
    "2041-2060": PREVISOES_DIR
    / "previsao_trigona_wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif",
    "2061-2080": PREVISOES_DIR
    / "previsao_trigona_wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2061-2080.tif",
    "2081-2100": PREVISOES_DIR
    / "previsao_trigona_wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2081-2100.tif",
}

# ======================================================================
# LAYOUT – HEADER E SIDEBAR
# ======================================================================

# HEADER
st.markdown(
    """
<div class="main-header">
  <h1>🐝 Trigona spinipes | Monitoramento Climático</h1>
  <p>Dashboard interativo para acompanhar a distribuição histórica e os cenários futuros da abelha Irapuá sob mudanças climáticas.</p>
</div>
""",
    unsafe_allow_html=True,
)

# SIDEBAR
with st.sidebar:
    st.markdown("### 🔎 Filtros rápidos")

    if df_ocorrencias is not None:
        ano_min = int(df_ocorrencias["year"].min())
        ano_max = int(df_ocorrencias["year"].max())
        faixa_anos = st.slider(
            "Intervalo de anos (para gráficos e mapa histórico)",
            min_value=ano_min,
            max_value=ano_max,
            value=(ano_min, ano_max),
            step=1,
        )
    else:
        faixa_anos = None

    st.markdown("---")
    st.markdown("### ℹ️ Sobre")
    st.write(
        "Este painel utiliza dados do GBIF e variáveis "
        "bioclimáticas do WorldClim para treinar um modelo Random Forest e "
        "prever a adequabilidade de habitat da *Trigona spinipes* em diferentes períodos."
    )

    st.markdown("---")
    st.caption("Desenvolvido em Python + Streamlit · Modelo: Random Forest")


# ======================================================================
# KPIs GLOBAIS – SEMPRE VISÍVEIS
# ======================================================================
if kpis:
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    with col_k1:
        st.markdown(
            f"""
        <div class="metric-card">
          <div class="metric-label">Registros totais</div>
          <div class="metric-value">{kpis['total_registros']:,}</div>
          <div class="metric-sub">Ocorrências válidas (GBIF)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_k2:
        st.markdown(
            f"""
        <div class="metric-card">
          <div class="metric-label">Cobertura temporal</div>
          <div class="metric-value">{kpis['anos_cobertura']} anos</div>
          <div class="metric-sub">{kpis['primeiro_registro']} – {kpis['ultimo_registro']}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_k3:
        st.markdown(
            f"""
        <div class="metric-card">
          <div class="metric-label">Estados com registros (BR)</div>
          <div class="metric-value">{kpis['estados_unicos']}</div>
          <div class="metric-sub">Distribuição espacial no Brasil</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_k4:
        st.markdown(
            f"""
        <div class="metric-card">
          <div class="metric-label">Ano de pico de registros</div>
          <div class="metric-value">{kpis['pico_registros_ano']}</div>
          <div class="metric-sub">{kpis['pico_registros_qtd']} registros no ano de maior atividade</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

# ======================================================================
# ABAS PRINCIPAIS
# ======================================================================

(
    aba_geral,
    aba_hist_especie,
    aba_ocorrencias,
    aba_cenarios,
    aba_descricao,
) = st.tabs(
    [
        "📊 Painel geral",
        "🐝 Histórico da espécie",
        "📈 Ocorrências & mapa",
        "🌡️ Cenários futuros por período",
        "📄 Descrição do projeto",
    ]
)

# ----------------------------------------------------------------------
# ABA 1 – PAINEL GERAL
# ----------------------------------------------------------------------
with aba_geral:
    st.subheader("Visão geral dos dados históricos")

    if df_ocorrencias is None or df_ocorrencias.empty:
        st.warning("Não foi possível carregar os dados de ocorrência.")
    else:
        # Aplicar filtro de anos da sidebar
        if faixa_anos is not None:
            df_filtrado = df_ocorrencias[
                (df_ocorrencias["year"] >= faixa_anos[0])
                & (df_ocorrencias["year"] <= faixa_anos[1])
            ].copy()
        else:
            df_filtrado = df_ocorrencias.copy()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Evolução histórica dos registros")
            contagem = df_filtrado["year"].value_counts().sort_index()

            fig, ax = plt.subplots(figsize=(8, 4), facecolor="none")
            ax.fill_between(contagem.index, contagem.values, alpha=0.25, color="#667eea")
            ax.plot(contagem.index, contagem.values, color="#667eea", linewidth=2.5)
            ax.set_xlabel("Ano")
            ax.set_ylabel("Número de registros")
            ax.grid(True, linestyle="--", alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig, use_container_width=True)

        with col2:
            st.markdown("#### 🗺️ Top 8 estados com mais registros")
            if "stateProvince" in df_filtrado.columns:
                top = df_filtrado["stateProvince"].value_counts().head(8)
                fig2, ax2 = plt.subplots(figsize=(8, 4), facecolor="none")
                cores = plt.cm.viridis(np.linspace(0.25, 0.9, len(top)))
                top.plot(kind="barh", ax=ax2, color=cores)
                ax2.invert_yaxis()
                ax2.set_xlabel("Número de registros")
                ax2.set_ylabel("")
                ax2.grid(True, axis="x", linestyle="--", alpha=0.2)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_visible(False)
                st.pyplot(fig2, use_container_width=True)
            else:
                st.info("Coluna `stateProvince` não encontrada no CSV de ocorrências.")

        st.markdown("#### 💡 Insights rápidos")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"""
            <div class="info-box">
              <h4 style="margin-top:0;">Período de maior atividade</h4>
              <p>O ano de <strong>{kpis['pico_registros_ano']}</strong> concentra 
              <strong>{kpis['pico_registros_qtd']} registros</strong>, sugerindo um período de intensificação
              do monitoramento ou de maior interesse na espécie.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col_b:
            st.markdown(
                f"""
            <div class="warning-box">
              <h4 style="margin-top:0;">Amplitude histórica</h4>
              <p>Os dados cobrem aproximadamente <strong>{kpis['anos_cobertura']} anos</strong>, 
              permitindo análises de tendência temporal e comparação com cenários projetados
              de mudanças climáticas.</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------
# ABA 2 – HISTÓRICO DA ESPÉCIE
# ----------------------------------------------------------------------
with aba_hist_especie:
    st.subheader("Contexto biológico e ecológico da Trigona spinipes")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
### Visão geral
A *Trigona spinipes*, conhecida como **Irapuá**, **Arapuá** ou **abelha-cachorro**, 
é uma abelha social sem ferrão (tribo Meliponini) amplamente distribuída no Brasil.

#### 🔬 Características morfológicas
- Tamanho médio: **6–7 mm** de comprimento  
- Coloração: corpo predominantemente **escuro**, com áreas amareladas  
- Não possui ferrão funcional (defesa baseada em **mordidas**)  

#### 🛡️ Comportamento defensivo
Apesar de não ferroar, a espécie é bastante **territorial**:
- Ataques em enxame quando o ninho é perturbado  
- Tendência a se enrolar em cabelos, pelos e roupas  
- Mordidas com mandíbulas fortes  

#### 🌸 Papel ecológico
Como polinizadora generalista, a Irapuá visita uma grande diversidade de plantas:
- Polinização de culturas agrícolas (cenoura, girassol, citros, maracujá etc.)  
- Manutenção de ecossistemas nativos  
"""
        )

        st.markdown(
            """
#### ⚠️ Relação com a agricultura
Em algumas situações, pode haver **conflito** com determinadas culturas (corte de botões florais),
mas os **benefícios ecológicos** tendem a superar os danos ocasionais.
"""
        )

    with col2:
        # 1. Caixa de Classificação Científica
        st.markdown(
            """
            <div class="info-box">
              <h4 style="margin-top:0;">📋 Classificação científica</h4>
              <table style="width:100%; border:none;">
                <tr><td><strong>Reino:</strong></td><td>Animalia</td></tr>
                <tr><td><strong>Filo:</strong></td><td>Arthropoda</td></tr>
                <tr><td><strong>Classe:</strong></td><td>Insecta</td></tr>
                <tr><td><strong>Ordem:</strong></td><td>Hymenoptera</td></tr>
                <tr><td><strong>Família:</strong></td><td>Apidae</td></tr>
                <tr><td><strong>Tribo:</strong></td><td>Meliponini</td></tr>
                <tr><td><strong>Gênero:</strong></td><td><em>Trigona</em></td></tr>
                <tr><td><strong>Espécie:</strong></td><td><em>T. spinipes</em></td></tr>
              </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Caixa de Etimologia
        st.markdown(
            """
            <div class="warning-box">
              <h4 style="margin-top:0;">🏛️ Etimologia</h4>
              <p>"Irapuá" vem de raízes tupi-guarani:</p>
              <ul>
                <li><strong>eíra</strong> — mel</li>
                <li><strong>apu'a</strong> — redondo</li>
              </ul>
              <p>O nome remete ao <strong>"mel redondo"</strong>, referência ao formato globoso do ninho.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3. Imagem Alinhada
        caminho_imagem = BASE_DIR / "assets" / "ninho_irapua.jpg"
        if not caminho_imagem.exists():
             caminho_imagem = BASE_DIR / "assets" / "ninho_irapua.png"

        if caminho_imagem.exists():
            st.image(
                str(caminho_imagem),
                caption="Ninho típico de Trigona spinipes (Acervo do Projeto)",
                use_container_width=True,
            )
        else:
            st.warning("Imagem 'ninho_irapua.jpg' não encontrada na pasta 'assets'.")

# ----------------------------------------------------------------------
# ABA 3 – OCORRÊNCIAS & MAPA HISTÓRICO
# ----------------------------------------------------------------------
with aba_ocorrencias:
    st.subheader("Ocorrências históricas e distribuição geográfica")

    if df_ocorrencias is None or df_ocorrencias.empty:
        st.warning("Não há dados de ocorrência carregados.")
    else:
        # aplicar filtro temporal
        if faixa_anos is not None:
            df_f = df_ocorrencias[
                (df_ocorrencias["year"] >= faixa_anos[0])
                & (df_ocorrencias["year"] <= faixa_anos[1])
            ].copy()
        else:
            df_f = df_ocorrencias.copy()

        col_top1, col_top2, col_top3 = st.columns(3)
        with col_top1:
            st.metric("Primeiro registro", int(df_f["year"].min()))
        with col_top2:
            st.metric("Último registro", int(df_f["year"].max()))
        with col_top3:
            st.metric("Total de registros", len(df_f))

        st.markdown("### 📅 Linha do tempo detalhada")
        cont_anos = df_f["year"].value_counts().sort_index()
        fig_t, ax_t = plt.subplots(figsize=(10, 4), facecolor="none")
        ax_t.bar(cont_anos.index, cont_anos.values, color="#cbd5ff", alpha=0.8)
        ax_t.set_xlabel("Ano")
        ax_t.set_ylabel("Registros")
        ax_t.spines["top"].set_visible(False)
        ax_t.spines["right"].set_visible(False)
        st.pyplot(fig_t, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🗺️ Mapa de ocorrências (amostra)")

        col_map, col_stats = st.columns([3, 1])

        with col_map:
            lat_c = df_f["decimalLatitude"].mean()
            lon_c = df_f["decimalLongitude"].mean()
            m = folium.Map(location=[lat_c, lon_c], zoom_start=4)

            amostra = df_f.sample(n=min(1500, len(df_f)), random_state=42)
            for _, row in amostra.iterrows():
                folium.CircleMarker(
                    location=[row["decimalLatitude"], row["decimalLongitude"]],
                    radius=2, color="#667eea", fill=True, fill_color="#667eea", fill_opacity=0.7,
                    popup=f"Ano: {int(row['year'])}",
                ).add_to(m)
            st_folium(m, width=900, height=520)

        with col_stats:
            st.markdown("#### Estatísticas espaciais")
            st.write(f"**Latitude média:** {df_f['decimalLatitude'].mean():.2f}°")
            st.write(f"**Longitude média:** {df_f['decimalLongitude'].mean():.2f}°")
            if "stateProvince" in df_f.columns:
                st.write(f"**Estados:** {df_f['stateProvince'].nunique()}")

# ----------------------------------------------------------------------
# ABA 4 – CENÁRIOS FUTUROS POR PERÍODO
# ----------------------------------------------------------------------
with aba_cenarios:
    st.subheader("Cenários futuros de adequabilidade climática por período")

    st.markdown(
        """
Nesta aba, você pode explorar os resultados do modelo Random Forest aplicado aos cenários 
de clima futuro (CMIP6 / SSP2-4.5). Cada período representa uma média climatológica de 20 anos.
"""
    )

    # sub-abas por período
    sub_tabs = st.tabs(list(arquivos_predicao.keys()))

    for (periodo, caminho), sub_tab in zip(arquivos_predicao.items(), sub_tabs):
        with sub_tab:
            st.markdown(f"### Período {periodo}")

            if not caminho.exists():
                st.warning(
                    f"Arquivo de previsão para {periodo} não encontrado em:\n`{caminho}`"
                )
                continue

            # --- MAPA INTERATIVO ---
            st.markdown("#### 🗺️ Mapa Interativo de Adequabilidade")
            st.caption("Visualize as áreas de maior adequabilidade (verde) e menor (laranja forte) com zoom dinâmico.")
            
            try:
                with rasterio.open(caminho) as src:
                    data = src.read(1)
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)
                    else:
                        data = np.where(data == -9999, np.nan, data)

                    # Chama a função de cores
                    img = colorize_raster_interactive(data)
                    
                    m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
                    
                    folium.raster_layers.ImageOverlay(
                        image=img, 
                        bounds=get_bounds(src), 
                        opacity=0.7,
                        name=f"Adequabilidade {periodo}"
                    ).add_to(m)
                    
                    folium.LayerControl().add_to(m)

                    # --- DIVISÃO: MAPA (Esq) x LEGENDA (Dir) ---
                    # Cria duas colunas: a primeira (maior) pro mapa, a segunda (menor) pra legenda
                    col_mapa, col_legenda = st.columns([6, 1])

                    with col_mapa:
                        # Renderiza o mapa com altura fixa de 500
                        st_folium(m, width=900, height=500, use_container_width=True)

                    with col_legenda:
                        # --- LEGENDA VERTICAL PERSONALIZADA (MODIFICADA) ---
                        # Invertida conforme pedido: Laranja no Topo (Alta), Verde na Base (Baixa)
                        # 'to bottom' começa com a primeira cor da lista (#d73027 - Laranja/Vermelho) no topo
                        html_vertical = """
                        <div style="display: flex; flex-direction: row; height: 500px; align-items: center;">
                            <div style="
                                background: linear-gradient(to bottom, #d73027, #fc8d59, #fee08b, #91cf60, #1a9850);
                                width: 25px;
                                height: 100%;
                                border-radius: 5px;
                                border: 1px solid #ccc;
                                margin-right: 10px;
                            "></div>
                            <div style="display: flex; flex-direction: column; justify-content: space-between; height: 100%; font-size: 0.85rem; color: #444; font-weight: 500;">
                                <span>Alta incidência</span>
                                <span>Média incidência</span>
                                <span>Baixa incidência</span>
                            </div>
                        </div>
                        """
                        st.markdown(html_vertical, unsafe_allow_html=True)
                    # ----------------------------------------------

            except Exception as e:
                st.error(f"Erro ao carregar visualização do mapa para {periodo}: {e}")

# ----------------------------------------------------------------------
# ABA 5 – DESCRIÇÃO DO PROJETO
# ----------------------------------------------------------------------
with aba_descricao:
    st.subheader("Descrição do projeto e do pipeline de modelagem")
    st.markdown(
        """
Esta aba resume **apenas a descrição do projeto**, da fonte de dados e do pipeline de modelagem.

### 🎯 Objetivo do projeto
Prever a **distribuição potencial futura** da abelha sem ferrão *Trigona spinipes* (Irapuá) no Brasil.

### 🧩 Fontes de dados
1. **Ocorrências da espécie** (GBIF)
2. **Variáveis climáticas atuais** (WorldClim v2.1)
3. **Cenários climáticos futuros** (CMIP6, SSP2-4.5, Modelo BCC-CSM2-MR)

### 🧠 Pipeline de modelagem (Random Forest)
1. **Carregamento e Limpeza**
2. **Geração de pontos de pseudo-ausência**
3. **Extração de variáveis climáticas**
4. **Treinamento do modelo**
5. **Projeção para cenários futuros**

### 📌 Como o dashboard se conecta ao modelo
Este `app.py` atua como a **camada de visualização** dos arquivos GeoTIFF gerados pelo modelo.
"""
    )