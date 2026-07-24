"""
    Funções com o objetivo de criar gráficos usando a biblioteca PLOTLY.
"""
import pandas as pd
import random
import numpy as np

import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import gaussian_kde, probplot
from scipy import stats


PALETA = px.colors.qualitative.Prism


def _obter_cor_aleatoria() -> str:
    return random.choice(PALETA)


def _gerar_texto_estatistico(df: pd.DataFrame, coluna: str) -> str:
    dados = df[coluna]
        
    media = dados.mean()
    desvio = dados.std()
    
    # Cálculo dos limites de Outlier (Regra do IQR - Intervalo Interquartil)
    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - (1.5 * iqr)
    lim_sup = q3 + (1.5 * iqr)
    
    nulos = dados.isna().sum()
    mediana = dados.median()
    minimo = dados.min()
    maximo = dados.max()
    unicos = dados.nunique()

    return (
    f"<b>{coluna.upper()}</b><br><br>"
    f"<b>Estatística Descritiva</b><br>"
    f"Média: {media:.2f}<br>"
    f"Mediana: {mediana:.2f}<br>"
    f"Desv. Padrão: {desvio:.2f}<br><br>"
    f"<b>Análise de Limites (IQR)</b><br>"
    f"Lim. Superior: {lim_sup:.2f}<br>"
    f"Lim. Inferior: {lim_inf:.2f}<br>"
    f"Mín / Máx: {minimo:.2f} | {maximo:.2f}<br><br>"
    f"Valores Nulos: {nulos} ({(nulos/len(dados))*100:.1f}%)<br>"
    f"Valores Únicos: {unicos}"
    )


def grafico_frequencia_absoluta(df: pd.DataFrame, target: str) -> go.Figure:
    tabela = df.groupby(target, as_index=False).size()
    fig = px.bar(
        tabela,
        x=target,
        y='size',
        title=f"GRÁFICO FREQUÊNCIA DA VARIÁVEL: {target.upper()}"
    )
    fig.update_layout(title_x=0.5)

    return fig


def grafico_frequencia_percentual(df: pd.DataFrame, target: str, cor=None) -> go.Figure:
    if cor is None:
        cor = _obter_cor_aleatoria()
        
    tabela = (df[target].value_counts(normalize=True) * 100).reset_index() 
    
    coluna_valores = [col for col in tabela.columns if col != target][0]
    
    fig = px.bar(
        tabela,
        x=target,
        y=coluna_valores,
        color_discrete_sequence=[cor],
        title=f"GRÁFICO FREQUÊNCIA (%) DA VARIÁVEL: {target.upper()}"
    )
    
    fig.update_traces(
        texttemplate='%{y:.1f}%', 
        textposition='outside'
    )
    
    fig.update_layout(
        title_x=0.5,
        yaxis_title="Porcentagem (%)"
    )
        
    return fig


def grafico_frequencia_percentual_alvo(df: pd.DataFrame, target: str, hue: str = None, mapa_cores: dict = None) -> go.Figure:
    
    colunas = [target] if hue is None else [target, hue]
    df_plot = df[colunas].dropna().copy()
    
    # Força conversão para string para evitar eixo contínuo
    df_plot[target] = df_plot[target].astype(str)
    
    if mapa_cores:
        mapa_cores = {str(k): v for k, v in mapa_cores.items()}
        
    if hue:
        df_plot[hue] = df_plot[hue].astype(str)
        
        # MATEMÁTICA CORRIGIDA (OPÇÃO 1): 
        # Agrupa primeiro pelo Eixo X (target) e vê a proporção das cores (hue)
        tabela = (df_plot.groupby(target)[hue]
                         .value_counts(normalize=True)
                         .mul(100)
                         .rename('percentual')
                         .reset_index())
        
        coluna_cor = hue               # A cor contina sendo o Sexo (Logo, o seu mapa funciona!)
        barmode_cfg = 'group'          
        titulo = f"PERFIL DE {hue.upper()} DENTRO DA VARIÁVEL {target.upper()}"
        
    else:
        tabela = (df_plot[target].value_counts(normalize=True) * 100).reset_index()
        tabela.columns = [target, 'percentual'] 
        coluna_cor = target            
        barmode_cfg = 'relative'       
        titulo = f"FREQUÊNCIA (%) DA VARIÁVEL: {target.upper()}"

    # Montagem do gráfico
    seq_cores = None if mapa_cores else [px.colors.qualitative.Set1[0]]

    fig = px.bar(
        tabela,
        x=target,                      # Eixo X = Survived (0, 1)
        y='percentual',
        color=coluna_cor,              # Cor = Sex (male, female)
        barmode=barmode_cfg,
        color_discrete_map=mapa_cores, # Seu dicionário azul/rosa vai brilhar aqui
        color_discrete_sequence=seq_cores,
        title=titulo
    )
    
    fig.update_traces(
        texttemplate='%{y:.1f}%', 
        textposition='outside'
    )
    
    fig.update_layout(
        title_x=0.5,
        yaxis_title="Porcentagem (%)",
        xaxis_title=target.upper(),
        legend_title=coluna_cor.upper() if hue else None
    )
        
    return fig


def grafico_frequencia_percentual_opcao2(df: pd.DataFrame, target: str, hue: str = None, mapa_cores: dict = None) -> go.Figure:
    
    colunas = [target] if hue is None else [target, hue]
    df_plot = df[colunas].dropna().copy()
    
    df_plot[target] = df_plot[target].astype(str)
    
    if mapa_cores:
        mapa_cores = {str(k): v for k, v in mapa_cores.items()}
        
    if hue:
        df_plot[hue] = df_plot[hue].astype(str)
        
        # MATEMÁTICA OPÇÃO 2: 
        # Agrupa pelo 'hue' (Eixo X) e calcula a proporção do 'target' (Cores)
        tabela = (df_plot.groupby(hue)[target]
                         .value_counts(normalize=True)
                         .mul(100)
                         .rename('percentual')
                         .reset_index())
        
        # INVERSÃO VISUAL: O Eixo X agora é o 'hue' e a cor é o 'target'
        eixo_x = hue
        coluna_cor = target               
        barmode_cfg = 'group'          
        titulo = f"TAXA DE {target.upper()} POR {hue.upper()}"
        
    else:
        tabela = (df_plot[target].value_counts(normalize=True) * 100).reset_index()
        tabela.columns = [target, 'percentual'] 
        eixo_x = target
        coluna_cor = target            
        barmode_cfg = 'relative'       
        titulo = f"FREQUÊNCIA (%) DA VARIÁVEL: {target.upper()}"

    seq_cores = None if mapa_cores else [px.colors.qualitative.Set1[0]]

    fig = px.bar(
        tabela,
        x=eixo_x,                      # Eixo X = Sex (male, female)
        y='percentual',
        color=coluna_cor,              # Cor = Survived (0, 1)
        barmode=barmode_cfg,
        color_discrete_map=mapa_cores, # O dicionário agora precisa mapear 0 e 1!
        color_discrete_sequence=seq_cores,
        title=titulo
    )
    
    fig.update_traces(
        texttemplate='%{y:.1f}%', 
        textposition='outside'
    )
    
    fig.update_layout(
        title_x=0.5,
        yaxis_title="Porcentagem (%)",
        xaxis_title=eixo_x.upper(),
        legend_title=coluna_cor.upper() if hue else None
    )
        
    return fig


def grafico_distribuicao_zscore(df: pd.DataFrame, target: str, cor = _obter_cor_aleatoria) -> go.Figure:
    tabela = df.copy()
    tabela[f'{target}_zscore'] = stats.zscore(tabela[target], nan_policy='omit')

    # Figure factory precisa de dados sem NaNs
    dados_limpos = tabela[f'{target}_zscore'].dropna()

    # ff.create_distplot é o equivalente ao sns.histplot(kde=True)
    fig = ff.create_distplot(
        [dados_limpos], 
        group_labels=[target], 
        colors=[cor],
        show_hist=True, 
        show_rug=False
    )
        
    fig.update_layout(
        title=f'DISTRIBUIÇÃO Z-SCORE DA VARIÁVEL: {target.upper()}',
        title_x=0.5,
        xaxis_title=f'{target} (Z-Score)',
        yaxis_title='Densidade'
    )
    
    return fig


def grafico_distribuicao(df: pd.DataFrame, target: str, cor = _obter_cor_aleatoria) -> go.Figure:
    dados_limpos = df[target].dropna()

    fig = ff.create_distplot(
        [dados_limpos], 
        group_labels=[target], 
        colors=[cor],
        show_hist=True, 
        show_rug=False
    )
        
    fig.update_layout(
        title=f'DISTRIBUIÇÃO DA VARIÁVEL: {target.upper()}',
        title_x=0.5,
        xaxis_title=f'{target}',
        yaxis_title='Densidade'
    )
    
    return fig


def grafico_boxplot(df: pd.DataFrame, target: str, orientation='h', hue=None) -> go.Figure:
    # No Plotly Express, a orientação é definida mapeando o 'target' para o eixo X ou Y
    kwargs = {}
    if orientation == 'h':
        kwargs['x'] = target
    else:
        kwargs['y'] = target
        
    if hue:
        kwargs['color'] = hue

    fig = px.box(
        df,
        **kwargs,
        title=f'BOXPLOT: {target.upper()}',
        color_discrete_sequence=PALETA # Permite usar toda a paleta se houver "hue"
    )

    fig.update_layout(title_x=0.5)

    return fig


def grafico_qqplot(df: pd.DataFrame, target: str) -> go.Figure:
    coluna = pd.to_numeric(df[target], errors='coerce')
    
    coluna = coluna.dropna()
    
    if coluna.empty:
        fig = go.Figure()
        fig.update_layout(title=f"QQ-Plot: {target.upper()} (Sem Dados Numéricos)")
        return fig

    (osm, osr), (slope, inter, _) = probplot(coluna, dist="norm")

    # 5. Montagem do gráfico
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Amostra",
                             marker=dict(color='red', size=5)))
    
    fig.add_trace(go.Scatter(x=osm, y=slope * osm + inter, mode="lines", name="Normal",
                             line=dict(color="#030101", width=2)))

    fig.update_layout(title=f"QQ-Plot: {target.upper()}")
    fig.update_layout(title_x=0.5)

    return fig


# Mantemos a sua função auxiliar intacta, pois ela é brilhante para tratar os dados
def _frame_numerico(df: pd.DataFrame, incluir=()):
    num = df.select_dtypes("number").copy()
    for c in df.columns:
        s = df[c]
        if isinstance(s.dtype, pd.CategoricalDtype):
            if s.dtype.ordered or s.nunique(dropna=True) == 2:   # ordinal OU binário
                num[c] = s.cat.codes.replace(-1, np.nan)
    for c in incluir:                        # força qualquer coluna extra pedida
        if c not in num.columns:
            num[c] = pd.factorize(df[c])[0].astype(float)
            num.loc[df[c].isna(), c] = np.nan
    return num


def grafico_correlacao(df: pd.DataFrame, incluir=()) -> go.Figure:
    """
    Gera um painel com Heatmaps de correlação Pearson (linear) e Spearman (monotônica).
    """
    # 1. Prepara os dados numéricos e codificados usando a função auxiliar
    num = _frame_numerico(df, incluir=incluir)
    
    # 2. Trava de segurança: precisa de pelo menos 2 colunas para calcular correlação
    if num.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(title="Correlação: Colunas numéricas insuficientes na base de dados.")
        return fig

    # 3. Cálculo matemático das matrizes de correlação
    cols = num.columns.tolist()
    cp = num.corr(method="pearson").values
    cs = num.corr(method="spearman").values

    # 4. Criação da estrutura de subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Pearson (Correlação Linear)", "Spearman (Correlação Monotônica)"),
        horizontal_spacing=0.15 # Espaço extra para os nomes das colunas não encavalarem
    )

    # 5. Adiciona os Heatmaps
    for j, matriz_valores in enumerate([cp, cs], start=1):
        fig.add_trace(
            go.Heatmap(
                z=matriz_valores, 
                x=cols, 
                y=cols, 
                zmin=-1, zmax=1,
                colorscale="RdBu_r", 
                coloraxis="coloraxis",
                texttemplate="%{z:.2f}", # Imprime os valores com 2 casas decimais nos quadrados
                textfont_size=10,
                hoverinfo="x+y+z"
            ), 
            row=1, col=j
        )

    # 6. Atualização do Layout Geral
    fig.update_layout(
        title_text="<b>Análise de Correlação das Variáveis</b>",
        title_x=0.5,
        height=600,
        width=1200,
        coloraxis=dict(colorscale="RdBu_r", cmin=-1, cmax=1),
    )
    
    # Inverte o eixo Y para a diagonal principal (1.00) ficar de cima-esquerda para baixo-direita
    fig.update_yaxes(autorange="reversed")

    return fig


def analise_variaveis_quantitativas(df: pd.DataFrame, lista_variaveis: list) -> go.Figure:
    for coluna in lista_variaveis:
        cor = _obter_cor_aleatoria()
        texto_estatistico = _gerar_texto_estatistico(df, coluna)

        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Histograma / Densidade", "Distribuição Detalhada", "Boxplot", "QQ-Plot"))

        fig_zscore = grafico_distribuicao_zscore(df, target=coluna, cor=cor)
        for trace in fig_zscore.data:
            fig.add_trace(trace, row=1, col=1)
            fig.update_xaxes(title_text=coluna, row=1, col=1)
            fig.update_yaxes(title_text="Densidade", row=1, col=1)

        fig_dist = grafico_distribuicao(df, target=coluna, cor=cor)
        for trace in fig_dist.data:
            fig.add_trace(trace, row=1, col=2)
            fig.update_xaxes(title_text=coluna, row=1, col=2)
            fig.update_yaxes(title_text="Frequência", row=1, col=2)

        fig_box = grafico_boxplot(df, target=coluna)
        for trace in fig_box.data:
            fig.add_trace(trace, row=2, col=1)
            fig.update_xaxes(title_text=coluna, row=2, col=1)

        fig_qq = grafico_qqplot(df, target=coluna)
        for trace in fig_qq.data:
            fig.add_trace(trace, row=2, col=2)
            fig.update_xaxes(title_text="Quantis Teóricos", row=2, col=2)
            fig.update_yaxes(title_text="Valores da Amostra", row=2, col=2)

        fig.update_layout(
            title_text=f"Análise da Variável: {coluna.upper()}",
            title_font_size=50,
            title_x=0.5,
            showlegend=False, 
            height=700, 
            width=1100,            # Aumentei um pouco a largura total
            margin=dict(r=200)     # Adiciona margem direita para a caixa não sumir da tela
        )

        # Adicionando a Anotação Lateral (A Caixinha)
        fig.add_annotation(
            text=texto_estatistico,
            xref="paper", yref="paper",
            x=1.05, y=0.5,         # x=1.05 joga a caixa para fora do gráfico, à direita
            xanchor="left", 
            yanchor="middle",
            align="left",
            showarrow=False,       # Tira a setinha padrão de anotações
            bordercolor="black",   # Borda preta igual da sua imagem
            borderwidth=1,
            bgcolor="white",       # Fundo branco
            borderpad=10           # Espaço interno entre o texto e a borda
        )
        
        fig.show()


def analise_variaveis_categoricas(df: pd.DataFrame, lista_variaveis: list, alvo = None) -> go.Figure:
    for coluna in lista_variaveis:
        cor = _obter_cor_aleatoria()
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Histograma / Densidade", "Distribuição Detalhada"))

        fig_barras = grafico_frequencia_percentual(df, coluna)
        for trace in fig_barras.data:
            fig.add_trace(trace, row=1, col=1)
            fig.update_xaxes(title_text=coluna, row=1, col=1)
            fig.update_yaxes(title_text="Frequência (%)", row=1, col=1)

        fig_barras_alvo = grafico_frequencia_percentual_alvo(df, coluna)
        for trace in fig_barras_alvo.data:
            fig.add_trace(trace, row=1, col=1)
            fig.update_xaxes(title_text=coluna, row=1, col=1)
            fig.update_yaxes(title_text="Frequência (%)", row=1, col=1)
