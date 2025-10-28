import unicodedata
import io
import hashlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="Painel de Auditoria Assistencial",
    layout="wide"
)

st.title("Painel de Auditoria Assistencial por Especialidade (Z-score)")

st.markdown(
    """
    Este painel identifica carga prescritiva,
    comparando médicos dentro da mesma especialidade e mês.
    
    • Indicador principal: **Exames / Consulta Finalizada**  
      (= Total de Exames ÷ Atendimentos Finalizados)  
      
    • Outlier = z-score alto nesse indicador.
    """
)

# =========================================================
# Helpers de normalização e leitura
# =========================================================
def _normalize_str(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower().strip()

@st.cache_data(show_spinner=False)
def _read_uploaded_file(name: str, digest: str, _content) -> pd.DataFrame:
    try:
        # garante bytes a partir de memoryview/bytes/bytearray
        if isinstance(_content, memoryview):
            content_bytes = _content.tobytes()
        elif isinstance(_content, (bytes, bytearray)):
            content_bytes = bytes(_content)
        else:
            content_bytes = bytes(_content)

        if name.lower().endswith((".xls", ".xlsx")):
            return pd.read_excel(io.BytesIO(content_bytes), dtype=str)
        # CSV: tenta detectar separador automaticamente
        try:
            return pd.read_csv(io.BytesIO(content_bytes), sep=None, engine="python", dtype=str)
        except Exception:
            for sep in [";", ",", "\t", "|"]:
                try:
                    return pd.read_csv(io.BytesIO(content_bytes), sep=sep, dtype=str)
                except Exception:
                    continue
            # último recurso
            return pd.read_csv(io.BytesIO(content_bytes), dtype=str)
    except Exception as e:
        raise RuntimeError(f"Falha ao ler arquivo: {e}")

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # mapa por versão normalizada (sem acento, minúscula)
    synonyms = {
        "mes": "Mes",
        "mes ": "Mes",
        "unidade": "Unidade",
        "especialidade": "Especialidade",
        "especialidade descricao": "Especialidade",
        "nome": "Médico",
        "medico": "Médico",
        "médico": "Médico",
        "atendimentos finalizados": "Finalizados",
        "finalizados": "Finalizados",
        "consultas com exames": "ConsultasComExames",
        "consultas c/ exames": "ConsultasComExames",
        "consultas c/exames": "ConsultasComExames",
        "total exames": "TotalExames",
        "total de exames": "TotalExames",
        "relacao": "RelacaoPlanilha",
        "relacao (ex/cons)": "RelacaoPlanilha",
        "relacao ex/cons": "RelacaoPlanilha",
    }
    rename_map = {}
    for c in df.columns:
        key = _normalize_str(c)
        new = synonyms.get(key)
        if new:
            rename_map[c] = new
    df2 = df.rename(columns=rename_map).copy()
    return df2

def _parse_br_number(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": np.nan, "(Vazio)": np.nan, "NA": np.nan, "None": np.nan})
    s1 = s.str.replace(r"\.", "", regex=True).str.replace(",", ".", regex=False)
    return pd.to_numeric(s1, errors="coerce")

def _normalize_mes_values(s: pd.Series) -> pd.Series:
    meses = [
        "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
        "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
    ]
    idx_by_norm = { _normalize_str(m): m for m in meses }
    def _conv(x):
        if pd.isna(x):
            return np.nan
        xs = str(x).strip()
        # numérico 1-12
        if xs.isdigit():
            n = int(xs)
            if 1 <= n <= 12:
                return meses[n-1]
        # tenta normalizada
        return idx_by_norm.get(_normalize_str(xs), xs)
    return s.apply(_conv)

# =========================================================
# 1) Upload e leitura/caching
# =========================================================
uploaded = st.file_uploader(
    "Envie o arquivo de produção (CSV ou Excel). Deve conter colunas como: Mês, Unidade, Especialidade, Médico, Finalizados, Consultas com Exames, Total Exames.",
    type=["csv", "xls", "xlsx"]
)

if uploaded is None:
    st.info("⬆️ Faça upload do arquivo para iniciar a análise.")
    st.stop()

_buf = uploaded.getbuffer()
_digest = hashlib.md5(_buf).hexdigest()
df_raw = _read_uploaded_file(uploaded.name, _digest, _buf)
df_raw = _standardize_columns(df_raw)

# =========================================================
# 2) Validação mínima + parsing
# =========================================================
required_cols = [
    "Mes", "Unidade", "Especialidade", "Médico",
    "Finalizados", "ConsultasComExames", "TotalExames",
]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Seu arquivo não possui as colunas obrigatórias: {missing}")
    st.stop()

df = df_raw.copy()
df["Finalizados_num"]         = _parse_br_number(df["Finalizados"])
df["ConsultasComExames_num"]  = _parse_br_number(df["ConsultasComExames"])
df["TotalExames_num"]         = _parse_br_number(df["TotalExames"])
if "RelacaoPlanilha" in df.columns:
    df["RelacaoPlanilha_num"] = _parse_br_number(df["RelacaoPlanilha"])
else:
    df["RelacaoPlanilha_num"] = np.nan

# métricas
df["rel_final"] = np.divide(
    df["TotalExames_num"],
    df["Finalizados_num"],
    out=np.full(len(df), np.nan),
    where=(df["Finalizados_num"] > 0)
)
df["rel_cex_calc"] = np.divide(
    df["TotalExames_num"],
    df["ConsultasComExames_num"],
    out=np.full(len(df), np.nan),
    where=(df["ConsultasComExames_num"] > 0)
)
df["rel_cex"] = df["rel_cex_calc"]
mask_rel = df["RelacaoPlanilha_num"].notna()
df.loc[mask_rel, "rel_cex"] = df.loc[mask_rel, "RelacaoPlanilha_num"]

# dataset base
data = df[[
    "Mes", "Unidade", "Especialidade", "Médico",
    "Finalizados_num", "ConsultasComExames_num", "TotalExames_num",
    "rel_final", "rel_cex"
]].copy()
data.rename(columns={
    "Finalizados_num": "Finalizados",
    "ConsultasComExames_num": "ConsultasComExames",
    "TotalExames_num": "TotalExames",
}, inplace=True)

# normaliza mês e cria ordem
data["Mes"] = _normalize_mes_values(data["Mes"]).astype(object)
ordem_meses = [
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
]
mes_rank = {m: i for i, m in enumerate(ordem_meses)}
data["Mes_ordem"] = data["Mes"].map(mes_rank)

# otimiza memória
for c in ["Mes", "Unidade", "Especialidade", "Médico"]:
    data[c] = data[c].astype("category")

# =========================================================
# 3) Funções de estatística (z-score por grupo)
# =========================================================
def _calc_stats_e_z(sub: pd.DataFrame):
    sub = sub.copy()
    vals_final = sub["rel_final"].dropna()
    if len(vals_final) >= 2:
        med_rel_final = float(vals_final.median())
        std_rel_final = float(vals_final.std(ddof=1)) or 1e-6
        sub["z_rel_final"] = (sub["rel_final"] - med_rel_final) / std_rel_final
    else:
        med_rel_final = np.nan
        std_rel_final = np.nan
        sub["z_rel_final"] = np.nan

    vals_cex_ratio = sub["rel_cex"].dropna()
    if len(vals_cex_ratio) >= 2:
        med_rel_cex = float(vals_cex_ratio.median())
        std_rel_cex = float(vals_cex_ratio.std(ddof=1)) or 1e-6
        sub["z_rel_cex"] = (sub["rel_cex"] - med_rel_cex) / std_rel_cex
    else:
        med_rel_cex = np.nan
        std_rel_cex = np.nan
        sub["z_rel_cex"] = np.nan

    def classify_z(z):
        if pd.isna(z):
            return "dentro"
        if abs(z) >= 2:
            return "extremo"
        if z >= 1:
            return "mais"
        if z <= -1:
            return "menos"
        return "dentro"

    sub["Faixa"] = sub["z_rel_final"].map(classify_z)
    sub["rank_z_rel_final"] = sub["z_rel_final"].rank(method="dense", ascending=False)
    sub["rank_z_rel_final_str"] = sub["rank_z_rel_final"].fillna("—").astype(str)
    sub["rank_rel_cex"] = sub["rel_cex"].rank(method="dense", ascending=False)

    stats = dict(
        med_rel_final=med_rel_final,
        std_rel_final=std_rel_final,
        med_rel_cex=med_rel_cex,
        std_rel_cex=std_rel_cex,
    )
    return sub, stats

def compute_group_stats(df_all: pd.DataFrame, mes: str, esp: str, unidade: str | None = None):
    sub = df_all[(df_all["Mes"] == mes) & (df_all["Especialidade"] == esp)].copy()
    if unidade and unidade != "Todas":
        sub = sub[sub["Unidade"].astype(str) == str(unidade)].copy()
    if sub.empty:
        return sub, dict(med_rel_final=np.nan, std_rel_final=np.nan, med_rel_cex=np.nan, std_rel_cex=np.nan)
    return _calc_stats_e_z(sub)

def compute_group_stats_generic(subframe: pd.DataFrame):
    # compara todos os médicos do mês, independente da especialidade
    return _calc_stats_e_z(subframe)

# =========================================================
# 4) Sidebar e filtros
# =========================================================
st.sidebar.header("Filtros")

meses_unicos_df = (
    data[["Mes", "Mes_ordem"]]
    .dropna()
    .drop_duplicates()
    .sort_values("Mes_ordem")
)
meses_disponiveis = [m for m in meses_unicos_df["Mes"].astype(str).tolist() if m in ordem_meses]

especialidades = sorted(data["Especialidade"].dropna().astype(str).unique().tolist())
especialidades = ["Todas"] + especialidades

unidades = sorted(data["Unidade"].dropna().astype(str).unique().tolist())
unidades = ["Todas"] + unidades

if "mes_escolhido" not in st.session_state:
    st.session_state.mes_escolhido = meses_disponiveis[-1] if meses_disponiveis else None
if "esp_escolhida" not in st.session_state:
    st.session_state.esp_escolhida = especialidades[0] if especialidades else None
if "unidade_escolhida" not in st.session_state:
    st.session_state.unidade_escolhida = unidades[0] if unidades else None

idx_mes = meses_disponiveis.index(st.session_state.mes_escolhido) if st.session_state.mes_escolhido in meses_disponiveis else max(len(meses_disponiveis)-1, 0)
st.sidebar.selectbox("Mês", meses_disponiveis, index=idx_mes, key="mes_escolhido")

idx_esp = especialidades.index(st.session_state.esp_escolhida) if st.session_state.esp_escolhida in especialidades else 0
st.sidebar.selectbox("Especialidade", especialidades, index=idx_esp, key="esp_escolhida")

idx_uni = unidades.index(st.session_state.unidade_escolhida) if st.session_state.unidade_escolhida in unidades else 0
st.sidebar.selectbox("Unidade", unidades, index=idx_uni, key="unidade_escolhida")

# filtro mínimo de volume para reduzir ruído
min_finalizados = st.sidebar.slider("Mínimo de atendimentos finalizados", min_value=0, max_value=50, value=5, step=1)

mes_escolhido = st.session_state.mes_escolhido
esp_escolhida = st.session_state.esp_escolhida
unidade_escolhida = st.session_state.unidade_escolhida

data_filtrada = data[data["Finalizados"] >= min_finalizados].copy()

# =========================================================
# 5) Subconjunto principal e estatísticas
# =========================================================
if esp_escolhida == "Todas":
    filtro_mes = data_filtrada[data_filtrada["Mes"].astype(str) == mes_escolhido].copy()
    if unidade_escolhida != "Todas":
        filtro_mes = filtro_mes[filtro_mes["Unidade"].astype(str) == unidade_escolhida]
    sub_atual, stats_atual = compute_group_stats_generic(filtro_mes)
else:
    sub_atual, stats_atual = compute_group_stats(data_filtrada, mes_escolhido, esp_escolhida, unidade_escolhida)

if sub_atual.empty:
    st.warning(f"Não há dados suficientes para Mês '{mes_escolhido}', Especialidade '{esp_escolhida}' e Unidade '{unidade_escolhida}'.")
    st.stop()

# =========================================================
# 6) Gráfico principal (3 modos)
# =========================================================
st.subheader("Distribuição dentro da especialidade")

def _fmt_stat(val):
    return f"{val:.2f}" if (isinstance(val, (int, float)) and np.isfinite(val)) else "—"

st.markdown(
    f"**{mes_escolhido} — {esp_escolhida} — {unidade_escolhida}**  |  "
    f"Mediana(Exames/Finalizados) = {_fmt_stat(stats_atual['med_rel_final'])}  |  "
    f"σ = {_fmt_stat(stats_atual['std_rel_final'])}"
)

FAIXA_COLORS = {"dentro":  "#6e7582", "menos": "#355c9a", "mais": "#b96b1f", "extremo": "#9e2f2f"}
y_base = 0.02
fig = go.Figure()

xgrid = np.linspace(-3, 3, 400)
curve = (1.0 / np.sqrt(2*np.pi)) * np.exp(-0.5 * xgrid * xgrid)
for x0, x1, col in [(-3,-2,"rgba(214,39,40,0.06)"),(-2,-1,"rgba(31,119,180,0.06)"),(-1,1,"rgba(108,117,125,0.05)"),(1,2,"rgba(255,127,14,0.06)"),(2,3,"rgba(214,39,40,0.06)")]:
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=curve.max()*1.05, fillcolor=col, line=dict(width=0), layer="below")
fig.add_trace(go.Scatter(x=np.r_[xgrid, xgrid[::-1]], y=np.r_[curve, np.zeros_like(curve)], fill="toself", mode="lines", line=dict(color="rgba(60,60,60,0.6)", width=1, dash="dot"), fillcolor="rgba(60,60,60,0.05)", name="Curva Normal(0,1)", visible=True))
for zval in [-2,-1,0,1,2]:
    fig.add_trace(go.Scatter(x=[zval, zval], y=[0, curve.max()*1.05], mode="lines", line=dict(color=("rgba(30,30,30,0.8)" if zval==0 else "rgba(150,150,150,0.4)"), width=(2 if zval==0 else 1), dash=("solid" if zval==0 else "dash")), hoverinfo="skip", showlegend=False, visible=True))
n_traces_fixos = len(fig.data)

modo_linha_traces, modo_volume_traces, modo_cex_traces = [], [], []
for faixa, chunk_original in sub_atual.groupby("Faixa"):
    color = FAIXA_COLORS.get(faixa, "#6e7582")
    chunk = chunk_original.reset_index(drop=True)
    def num_as_str(s):
        return pd.to_numeric(s, errors="coerce").round(2).astype(str).replace("nan", "—")
    hover_series = (
        "<b>" + chunk["Médico"].astype(str) + "</b><br>"
        + "Unidade: " + chunk["Unidade"].astype(str) + "<br>"
        + "Especialidade: " + chunk["Especialidade"].astype(str) + "<br>"
        + "Mês: " + chunk["Mes"].astype(str) + "<br><br>"
        + "Finalizados: " + pd.to_numeric(chunk["Finalizados"], errors="coerce").fillna(0).round(0).astype("Int64").astype(str) + "<br>"
        + "Consultas c/ Exames: " + pd.to_numeric(chunk["ConsultasComExames"], errors="coerce").fillna(0).round(0).astype("Int64").astype(str) + "<br>"
        + "Total de Exames: " + pd.to_numeric(chunk["TotalExames"], errors="coerce").fillna(0).round(0).astype("Int64").astype(str) + "<br><br>"
        + "Exames / Consulta Finalizada: " + num_as_str(chunk["rel_final"]) + "<br>"
        + "Exames / Consulta c/ Exame: " + num_as_str(chunk["rel_cex"]) + "<br><br>"
        + "z(Exames/Finalizados): " + num_as_str(chunk["z_rel_final"]) + "<br>"
        + "z(Exames/Consulta c/ Exame): " + num_as_str(chunk["z_rel_cex"]) + "<br>"
        + "Rank Finalizados: " + chunk["rank_z_rel_final_str"].astype(str) + "<br>"
        + "Rank Consulta c/ Exame: " + pd.to_numeric(chunk["rank_rel_cex"], errors="coerce").fillna(0).round(0).astype("Int64").astype(str).replace("<NA>", "—")
    )
    hover_text = hover_series.tolist()

    tr1 = go.Scatter(x=chunk["z_rel_final"], y=[y_base]*len(chunk), mode="markers", marker=dict(color=color, size=7, opacity=0.8, line=dict(width=0.4, color="black")), name=faixa, hovertemplate="%{text}<extra></extra>", text=hover_text, showlegend=(faixa != "dentro"), visible=True)
    fig.add_trace(tr1)
    modo_linha_traces.append(len(fig.data)-1)

    vols = pd.to_numeric(chunk["Finalizados"], errors="coerce").fillna(0).to_numpy()
    y_vol = y_base + ((vols - vols.min()) / (vols.max() - vols.min() + 1e-9)) * 0.25 if len(vols) else np.array([y_base]*len(chunk))
    tr2 = go.Scatter(x=chunk["z_rel_final"], y=y_vol, mode="markers", marker=dict(color=color, size=7, opacity=0.8, line=dict(width=0.4, color="black")), name=f"{faixa} (volume)", hovertemplate="%{text}<extra></extra>", text=hover_text, showlegend=False, visible=False)
    fig.add_trace(tr2)
    modo_volume_traces.append(len(fig.data)-1)

    chunk_cex = chunk[chunk["z_rel_cex"].notna()].copy()
    if not chunk_cex.empty:
        hover_text_cex = [hover_text[i] for i in chunk_cex.index]
        tr3 = go.Scatter(x=chunk_cex["z_rel_cex"], y=[y_base]*len(chunk_cex), mode="markers", marker=dict(color=color, size=7, opacity=0.8, line=dict(width=0.4, color="black")), name=f"{faixa} (c/ exame)", hovertemplate="%{text}<extra></extra>", text=hover_text_cex, showlegend=False, visible=False)
        fig.add_trace(tr3)
        modo_cex_traces.append(len(fig.data)-1)

fig.update_layout(
    xaxis=dict(title="Desvio-padrão (z)", range=[-3.2, 3.2], zeroline=False, gridcolor="rgba(0,0,0,0.05)", linecolor="rgba(0,0,0,0.3)"),
    yaxis=dict(showticklabels=False, title="Escala interna", gridcolor="rgba(0,0,0,0.05)", linecolor="rgba(0,0,0,0.3)"),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
    margin=dict(l=40, r=40, t=40, b=40),
)

modo = st.radio("Modo do gráfico", ("Linha (Exames / Finalizados)", "Volume de Consultas", "Consultas com Exame"), horizontal=True, index=0)
for i in range(len(fig.data)):
    fig.data[i].visible = (i < n_traces_fixos)
if modo == "Linha (Exames / Finalizados)":
    for idx in modo_linha_traces: fig.data[idx].visible = True
elif modo == "Volume de Consultas":
    for idx in modo_volume_traces: fig.data[idx].visible = True
else:
    for idx in modo_cex_traces: fig.data[idx].visible = True

st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 7) Tabela detalhada
# =========================================================
st.subheader("Tabela (mês, especialidade e unidade selecionados)")
tabela = sub_atual[[
    "Médico","Unidade","Mes","Especialidade",
    "Finalizados","ConsultasComExames","TotalExames",
    "rel_final","rel_cex","z_rel_final","z_rel_cex","Faixa",
    "rank_z_rel_final_str","rank_rel_cex"
]].copy()
tabela.rename(columns={
    "Mes": "Mês",
    "Finalizados": "Atendimentos Finalizados",
    "ConsultasComExames": "Consultas c/ Exames",
    "TotalExames": "Total de Exames",
    "rel_final": "Exames / Consulta Finalizada",
    "rel_cex": "Exames / Consulta com Exame",
    "z_rel_final": "z(Exames/Finalizados)",
    "z_rel_cex": "z(Exames/Consulta c/ Exame)",
    "Faixa": "Faixa (Z Final)",
    "rank_z_rel_final_str": "Posição no Grupo (Finalizados)",
    "rank_rel_cex": "Rank Exames/Consulta c/ Exame",
}, inplace=True)
tabela = tabela.sort_values(by="z(Exames/Finalizados)", ascending=False)
st.dataframe(tabela, use_container_width=True)
csv_download = tabela.to_csv(index=False).encode("utf-8-sig")
st.download_button("Baixar tabela atual em CSV", data=csv_download, file_name=f"tabela_{mes_escolhido}_{esp_escolhida}_{unidade_escolhida}.csv", mime="text/csv")

# =========================================================
# 8) Evolução histórica
# =========================================================
st.subheader("Evolução histórica do médico")
if esp_escolhida == "Todas":
    medicos_da_esp = data["Médico"].dropna().astype(str).unique().tolist()
else:
    medicos_da_esp = data[data["Especialidade"] == esp_escolhida]["Médico"].dropna().astype(str).unique().tolist()
medico_sel = st.selectbox("Escolha um médico", sorted(medicos_da_esp))
historico = data[data["Médico"] == medico_sel].copy()
if esp_escolhida != "Todas":
    historico = historico[historico["Especialidade"] == esp_escolhida].copy()
hist = historico.copy()
# evita bugs de groupby com categóricos em algumas versões do pandas
hist["Mes"] = hist["Mes"].astype(str)
hist["Especialidade"] = hist["Especialidade"].astype(str)
historico_esp = (
    hist.groupby(["Mes", "Mes_ordem", "Especialidade"], observed=True)
        .agg({
            "rel_final": "mean",
            "Finalizados": "sum",
            "ConsultasComExames": "sum",
            "TotalExames": "sum"
        })
        .reset_index()
        .sort_values("Mes_ordem")
)
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(x=historico_esp["Mes"], y=historico_esp["rel_final"], mode="lines+markers", line=dict(width=2), marker=dict(size=8), name="Exames / Consulta Finalizada"))
fig_line.update_layout(template="plotly_white", xaxis_title="Mês", yaxis_title="Exames / Consulta Finalizada", title=f"Evolução de {medico_sel} ({'todas as especialidades' if esp_escolhida=='Todas' else esp_escolhida})")
st.plotly_chart(fig_line, use_container_width=True)
st.caption("Obs: 'Exames / Consulta Finalizada' = Total de Exames ÷ Atendimentos Finalizados. Esse é o indicador que define o z-score e os outliers.")

# =========================================================
# 9) Painel de ofensores por unidade
# =========================================================
st.subheader("Ofensores na Unidade Selecionada")
if esp_escolhida == "Todas":
    base_para_unidades = data[data["Mes"].astype(str) == mes_escolhido].copy()
else:
    base_para_unidades = data[(data["Mes"].astype(str) == mes_escolhido) & (data["Especialidade"] == esp_escolhida)].copy()
unidades_disp = sorted(base_para_unidades["Unidade"].dropna().astype(str).unique().tolist())
if not unidades_disp:
    st.info("Nenhuma unidade disponível nesse recorte.")
else:
    unidade_sel = st.selectbox("Unidade para inspecionar possíveis ofensores", unidades_disp, index=0)
    st.markdown(f"**Unidade {unidade_sel} | Mês {mes_escolhido} | Especialidade {esp_escolhida}**")
    sub_u = compute_group_stats_generic(base_para_unidades[base_para_unidades["Unidade"].astype(str) == unidade_sel].copy())[0]
    if sub_u.empty:
        st.info("Não há dados suficientes nessa unidade para calcular desvios.")
    else:
        bloco_finalizados = sub_u[["Médico","Unidade","Especialidade","Finalizados","ConsultasComExames","TotalExames","rel_final","z_rel_final","rank_z_rel_final"]].copy()
        bloco_finalizados["Indicador"] = "Exames / Consulta Finalizada"
        bloco_finalizados["Valor do Indicador"] = bloco_finalizados["rel_final"]
        bloco_finalizados["z-score (unidade)"]  = bloco_finalizados["z_rel_final"]
        bloco_finalizados["Posição na Unidade"] = bloco_finalizados["rank_z_rel_final"]

        bloco_cex = sub_u[["Médico","Unidade","Especialidade","Finalizados","ConsultasComExames","TotalExames","rel_cex","z_rel_cex","rank_rel_cex"]].copy()
        bloco_cex["Indicador"] = "Exames / Consulta com Exame"
        bloco_cex["Valor do Indicador"] = bloco_cex["rel_cex"]
        bloco_cex["z-score (unidade)"]  = bloco_cex["z_rel_cex"]
        bloco_cex["Posição na Unidade"] = bloco_cex["rank_rel_cex"]

        ofensores_long = pd.concat([bloco_finalizados, bloco_cex], ignore_index=True)
        def classifica_tipo(z):
            if pd.isna(z): return ""
            if z >= 2: return "extremo"
            if z >= 1: return "suspeito"
            return "normal"
        ofensores_long["Classificação"] = ofensores_long["z-score (unidade)"].map(classifica_tipo)
        corte_txt = st.selectbox("Mostrar quem?", ("Somente extremos (z ≥ 2)", "Extremos e suspeitos (z ≥ 1)", "Todos"), index=1)
        if corte_txt == "Somente extremos (z ≥ 2)":
            ofensores_filtrados = ofensores_long[ofensores_long["Classificação"] == "extremo"].copy()
        elif corte_txt == "Extremos e suspeitos (z ≥ 1)":
            ofensores_filtrados = ofensores_long[ofensores_long["Classificação"].isin(["extremo","suspeito"])].copy()
        else:
            ofensores_filtrados = ofensores_long.copy()
        ofensores_filtrados = ofensores_filtrados.sort_values(by="z-score (unidade)", ascending=False)
        tabela_ofensores = ofensores_filtrados[["Médico","Unidade","Especialidade","Finalizados","ConsultasComExames","TotalExames","Indicador","Valor do Indicador","z-score (unidade)","Classificação","Posição na Unidade"]].copy()
        tabela_ofensores.rename(columns={"Finalizados":"Atendimentos Finalizados","ConsultasComExames":"Consultas c/ Exames","TotalExames":"Total de Exames"}, inplace=True)
        st.dataframe(tabela_ofensores, use_container_width=True)
        csv_of = tabela_ofensores.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar ofensores (unidade selecionada)", data=csv_of, file_name=f"ofensores_{mes_escolhido}_{esp_escolhida}_{unidade_sel}.csv", mime="text/csv")
