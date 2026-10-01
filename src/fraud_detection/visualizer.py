"""
Módulo de visualização com SEABORN e MATPLOTLIB.
Otimizado para bases grandes e para uma análise exploratória visual e objetiva.
"""
from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from matplotlib.ticker import FuncFormatter


sns.set_theme(style="whitegrid", palette="muted")

# Limite de pontos usados no QQ-Plot (evita estourar a memória em bases gigantes)
_MAX_PONTOS_QQ = 50_000

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _obter_cor_aleatoria() -> str:
    """Retorna uma cor hexadecimal aleatória de uma paleta profissional padrão."""
    paleta = sns.color_palette("tab10").as_hex()
    return random.choice(paleta)


def _valores_finitos(df: pd.DataFrame, variavel: str) -> np.ndarray:
    """Retorna os valores numéricos finitos da coluna (sem NaN e sem ±inf)."""
    serie = pd.to_numeric(df[variavel], errors="coerce")
    valores = serie.to_numpy(dtype=float, na_value=np.nan)
    return valores[np.isfinite(valores)]


def _eixo_sem_dados(ax, titulo: str, mensagem: str):
    """Deixa o eixo com um aviso centralizado em vez de quebrar o painel inteiro."""
    ax.set_title(titulo, fontweight="bold")
    ax.text(0.5, 0.5, mensagem, ha="center", va="center", fontsize=11,
            color="gray", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def _rotular_barras(ax, casas_decimais: int):
    """Escreve o valor percentual em cima de cada barra."""
    for barra in ax.patches:
        altura = barra.get_height()
        if np.isfinite(altura) and altura > 0:
            ax.annotate(f"{altura:.{casas_decimais}f}%",
                        (barra.get_x() + barra.get_width() / 2.0, altura),
                        ha="center", va="bottom", fontsize=10,
                        xytext=(0, 5), textcoords="offset points")


def _ordem_por_frequencia(df: pd.DataFrame, variavel: str) -> list:
    """Categorias ordenadas da mais para a menos frequente (ignora categorias sem registros)."""
    contagem = df[variavel].value_counts()
    return contagem[contagem > 0].index.tolist()


def _texto_resumo_estatistico(df: pd.DataFrame, variavel: str) -> str:
    """Gera o texto do resumo estatístico exibido na caixa lateral do painel quantitativo."""
    titulo = str(variavel).upper()
    total = len(df)
    qtd_nulos = int(df[variavel].isna().sum())
    pct_nulos = qtd_nulos / total * 100 if total else 0.0

    serie = pd.to_numeric(df[variavel], errors="coerce")
    qtd_infinitos = int(np.isinf(serie.to_numpy(dtype=float, na_value=np.nan)).sum())
    dados = pd.Series(_valores_finitos(df, variavel))

    if dados.empty:
        return (
            f"{titulo}\n\n"
            f"Sem valores numéricos válidos.\n\n"
            f"Valores nulos: {qtd_nulos} ({pct_nulos:.1f}%)"
        )

    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    qtd_outliers = int(((dados < limite_inferior) | (dados > limite_superior)).sum())
    pct_outliers = qtd_outliers / len(dados) * 100

    desvio = dados.std()
    texto_desvio = f"{desvio:.2f}" if np.isfinite(desvio) else "N/D"

    linha_infinitos = f"Valores infinitos: {qtd_infinitos}\n" if qtd_infinitos else ""

    return (
        f"{titulo}\n\n"
        f"--- Estatística descritiva ---\n"
        f"Média: {dados.mean():.2f}\n"
        f"Mediana: {dados.median():.2f}\n"
        f"Desvio padrão: {texto_desvio}\n\n"
        f"--- Limites do IQR ---\n"
        f"Limite superior: {limite_superior:.2f}\n"
        f"Limite inferior: {limite_inferior:.2f}\n"
        f"Mín. / Máx.: {dados.min():.2f} | {dados.max():.2f}\n\n"
        f"--- Qualidade dos dados ---\n"
        f"Valores nulos: {qtd_nulos} ({pct_nulos:.1f}%)\n"
        f"{linha_infinitos}"
        f"Valores únicos: {dados.nunique()}\n"
        f"Outliers: {qtd_outliers} ({pct_outliers:.2f}%)"
    )


def _texto_resumo_categorico(df: pd.DataFrame, variavel: str) -> str:
    """Gera o texto do resumo estatístico para variáveis categóricas."""
    dados = df[variavel]
    total = len(dados)
    qtd_ausentes = int(dados.isna().sum())
    qtd_validos = total - qtd_ausentes
    pct_ausentes = qtd_ausentes / total * 100 if total else 0.0

    contagem = dados.value_counts()
    contagem = contagem[contagem > 0]

    if contagem.empty:
        categoria_top, freq_top, participacao = "N/D", 0, 0.0
    else:
        categoria_top = contagem.index[0]
        freq_top = int(contagem.iloc[0])
        # Mesma base do gráfico de frequência relativa (apenas valores não nulos)
        participacao = freq_top / qtd_validos * 100

    texto_categoria = str(categoria_top)
    if len(texto_categoria) > 25:
        texto_categoria = texto_categoria[:22] + "..."

    return (
        f"{str(variavel).upper()}\n\n"
        f"--- Visão geral ---\n"
        f"Total de registros: {total}\n"
        f"Valores ausentes: {qtd_ausentes} ({pct_ausentes:.1f}%)\n"
        f"Categorias únicas: {dados.nunique()}\n\n"
        f"--- Categoria mais frequente ---\n"
        f"Valor: {texto_categoria}\n"
        f"Contagem: {freq_top}\n"
        f"Participação: {participacao:.1f}%\n"
        f"(sobre os não nulos)"
    )


def _converter_para_numerico(df: pd.DataFrame, incluir=()) -> pd.DataFrame:
    """
    Monta a base numérica usada nas correlações.

    - Colunas numéricas entram como estão; colunas booleanas entram como 0/1.
    - Categóricas ordenadas ou binárias entram pelos códigos da categoria.
    - Colunas em `incluir` que ainda não entraram são codificadas em ordem
      alfabética/crescente (ordem estável, não depende da ordem das linhas).
    """
    numerico = df.select_dtypes("number").copy()

    for coluna in df.columns:
        serie = df[coluna]
        if coluna in numerico.columns:
            continue
        if pd.api.types.is_bool_dtype(serie):
            numerico[coluna] = serie.astype("float64")
        elif isinstance(serie.dtype, pd.CategoricalDtype):
            if serie.dtype.ordered or serie.nunique(dropna=True) == 2:
                codigos = serie.cat.codes.astype("float64")
                numerico[coluna] = codigos.where(codigos != -1)

    for coluna in incluir:
        if coluna not in numerico.columns:
            codigos = pd.factorize(df[coluna], sort=True)[0].astype("float64")
            numerico[coluna] = np.where(codigos == -1, np.nan, codigos)

    return numerico


def _numero_normal(valor, _):
    """Rótulo do eixo como número comum (1.000 em vez de 10³)."""
    return f"{valor:,.0f}".replace(",", ".")

# ---------------------------------------------------------------------------
# Gráficos quantitativos
# ---------------------------------------------------------------------------

def plotar_boxplot(df: pd.DataFrame, variavel: str, ax=None, orientacao: str = "h",
                   hue=None, mapa_cores=None, cor=None, escala: str = "linear"):
    """
    Boxplot da variável.

    escala: 'linear' (padrão) ou 'symlog' (para variáveis de cauda longa, como valores monetários).
    """
    if orientacao not in ("h", "v"):
        raise ValueError("orientacao deve ser 'h' (horizontal) ou 'v' (vertical).")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    titulo = f"BOXPLOT: {str(variavel).upper()}"

    colunas = [variavel] if hue is None or hue == variavel else [variavel, hue]
    dados = df[colunas].copy()
    dados[variavel] = pd.to_numeric(dados[variavel], errors="coerce").astype("float64")
    dados[variavel] = dados[variavel].replace([np.inf, -np.inf], np.nan)

    if dados[variavel].notna().sum() == 0:
        return _eixo_sem_dados(ax, titulo, "Sem dados numéricos")

    x = variavel if orientacao == "h" else None
    y = variavel if orientacao == "v" else None

    if hue is None:
        sns.boxplot(data=dados, x=x, y=y, ax=ax, color=cor or _obter_cor_aleatoria())
    else:
        sns.boxplot(data=dados, x=x, y=y, hue=hue, ax=ax, palette=mapa_cores or "Set2")

    eixo = ax.xaxis if orientacao == "h" else ax.yaxis
    (ax.set_xscale if orientacao == "h" else ax.set_yscale)(escala)

    if escala != "linear":
        eixo.set_major_formatter(FuncFormatter(_numero_normal))
        if dados[variavel].min() >= 0:  # sem negativos: o eixo começa no 0
            (ax.set_xlim if orientacao == "h" else ax.set_ylim)(0, None)

    ax.set_title(titulo, fontweight="bold")
    return ax


def plotar_histograma(df: pd.DataFrame, variavel: str, ax=None, cor=None, bins = 100):
    """Histograma com frequência em escala logarítmica (100 bins)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    titulo = f"DISTRIBUIÇÃO (LOG): {str(variavel).upper()}"

    dados = _valores_finitos(df, variavel)
    if dados.size == 0:
        return _eixo_sem_dados(ax, titulo, "Sem dados numéricos")

    ax.hist(dados, bins=bins, color=cor or _obter_cor_aleatoria(),
            edgecolor="black", linewidth=0.5, log=True)

    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel(variavel)
    ax.set_ylabel("Frequência (log)")
    return ax


def plotar_histograma_zscore(df: pd.DataFrame, variavel: str, ax=None, cor=None):
    """Histograma do z-score da variável, com frequência em escala logarítmica."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    titulo = f"Z-SCORE (LOG): {str(variavel).upper()}"

    # OTIMIZAÇÃO: z-score vetorizado via numpy
    dados = _valores_finitos(df, variavel)
    if dados.size < 2:
        return _eixo_sem_dados(ax, titulo, "Dados insuficientes")
    if np.ptp(dados) == 0:
        return _eixo_sem_dados(ax, titulo, "Variância zero:\nz-score indefinido")

    z_scores = stats.zscore(dados)

    ax.hist(z_scores, bins=100, color=cor or _obter_cor_aleatoria(),
            edgecolor="black", linewidth=0.5, log=True)

    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel(f"{variavel} (z-score)")
    ax.set_ylabel("Frequência (log)")
    return ax


def plotar_qq_normalidade(df: pd.DataFrame, variavel: str, ax=None):
    """QQ-Plot contra a distribuição normal teórica."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    titulo = f"QQ-PLOT: {str(variavel).upper()}"

    dados = _valores_finitos(df, variavel)
    if dados.size < 2:
        return _eixo_sem_dados(ax, titulo, "Sem dados numéricos suficientes")
    if np.ptp(dados) == 0:
        return _eixo_sem_dados(ax, titulo, "Variância zero:\nQQ-Plot indefinido")

    # TRAVA DE SEGURANÇA: em bases gigantes o QQ-Plot usa uma amostra
    # de no máximo 50 mil pontos para não estourar a memória.
    amostrado = dados.size > _MAX_PONTOS_QQ
    if amostrado:
        rng = np.random.default_rng(42)
        dados = rng.choice(dados, size=_MAX_PONTOS_QQ, replace=False)
        titulo += f" (amostra de {_MAX_PONTOS_QQ:,})".replace(",", ".")

    (quantis_teoricos, valores_ordenados), (inclinacao, intercepto, _) = \
        stats.probplot(dados, dist="norm")

    # Ponto pequeno (s=2) para ficar mais limpo em bases grandes
    ax.scatter(quantis_teoricos, valores_ordenados, color="red", s=2, alpha=0.5, label="Amostra")
    ax.plot(quantis_teoricos, inclinacao * quantis_teoricos + intercepto,
            color="black", linewidth=2, label="Normal teórica")

    ax.set_title(titulo, fontweight="bold")
    ax.set_xlabel("Quantis teóricos")
    ax.set_ylabel("Valores da amostra")
    ax.legend(loc="upper left", markerscale=4)
    return ax


def plotar_heatmaps_correlacao(df, incluir=(), top_n=10, excluir=()):
    """
    Mapas de calor de Pearson e Spearman (um embaixo do outro) e os pares mais correlacionados.
    Mostra só o triângulo de baixo: o de cima é espelho e a diagonal é sempre 1.
    excluir: colunas que não fazem sentido correlacionar (ex.: ["ID"]).
    """
    numerico = _converter_para_numerico(df.drop(columns=list(excluir)), incluir=incluir)
    if numerico.shape[1] < 2:
        print("Colunas numéricas insuficientes para calcular correlação.")
        return None

    corr_pearson = numerico.corr(method="pearson")
    corr_spearman = numerico.corr(method="spearman")

    n = numerico.shape[1]
    lado = max(6, 0.6 * n)                     # tamanho cresce com o número de variáveis
    fonte = 10 if n <= 10 else 8 if n <= 20 else 6
    mascara_cima = np.triu(np.ones((n, n), dtype=bool))  # esconde diagonal + triângulo de cima

    fig, axes = plt.subplots(2, 1, figsize=(lado + 2, 2 * lado))
    for ax, corr, titulo in [(axes[0], corr_pearson, "Pearson (correlação linear)"),
                             (axes[1], corr_spearman, "Spearman (correlação monotônica)")]:
        sns.heatmap(corr, mask=mascara_cima, annot=True, fmt=".2f", annot_kws={"size": fonte},
                    cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
                    cbar_kws={"shrink": .7}, ax=ax)
        ax.set_title(titulo, fontweight="bold", fontsize=14)
        # sem a 1ª linha e a última coluna, que ficariam vazias com a máscara
        ax.set_ylim(n, 1)
        ax.set_xlim(0, n - 1)
        ax.grid(False)  # sem as linhas do tema aparecendo no triângulo vazio

    fig.suptitle("Análise de correlação entre variáveis", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.98))  # espaço para o título geral
    plt.show()

    mascara = np.tril(np.ones(corr_pearson.shape), k=-1).astype(bool)
    pares_pearson = corr_pearson.where(mascara).stack().dropna().reset_index()
    pares_pearson.columns = ["Variavel_1", "Variavel_2", "Pearson"]
    pares_spearman = corr_spearman.where(mascara).stack().dropna().reset_index()
    pares_spearman.columns = ["Variavel_1", "Variavel_2", "Spearman"]
    top_corr = pd.merge(pares_pearson, pares_spearman, on=["Variavel_1", "Variavel_2"], how="outer")
    top_corr = (top_corr.assign(_abs=top_corr["Pearson"].abs())
                .sort_values("_abs", ascending=False, na_position="last")
                .drop(columns="_abs").reset_index(drop=True))
    return top_corr.head(top_n) if top_n is not None else top_corr


def plotar_heatmaps_cruzada(df, linha, coluna, normalizar="index", ax=None, cmap="Blues"):
    """
    Mapa de calor de duas variáveis categóricas (tabela de contingência), no estilo
    de uma matriz de confusão: categorias de `linha` no eixo y e de `coluna` no eixo x.

    normalizar: "index" -> % de cada coluna dentro de cada linha (cada linha soma 100%)
                "columns" -> % de cada linha dentro de cada coluna (cada coluna soma 100%)
                None -> contagem absoluta
    O título traz o V de Cramér (0 = nenhuma associação, 1 = associação total),
    que é a "correlação" adequada para variáveis categóricas.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    contagem = pd.crosstab(df[linha], df[coluna])

    # V de Cramér a partir do qui-quadrado
    qui2 = stats.chi2_contingency(contagem, correction=False)[0]
    n = contagem.to_numpy().sum()
    v_cramer = np.sqrt(qui2 / (n * (min(contagem.shape) - 1)))

    if normalizar is None:
        tabela, fmt, sufixo = contagem, "d", "Contagem"
    else:
        tabela, fmt = pd.crosstab(df[linha], df[coluna], normalize=normalizar).mul(100), ".1f"
        sufixo = (f"% de cada {coluna} dentro de cada {linha}" if normalizar == "index"
                  else f"% de cada {linha} dentro de cada {coluna}")

    sns.heatmap(tabela, annot=True, fmt=fmt, cmap=cmap, vmin=0, linewidths=.5,
                cbar_kws={"label": sufixo if normalizar is None else "%"}, ax=ax)

    # n de cada linha no rótulo (grupos pequenos = % instáveis)
    ax.set_yticklabels([f"{c} (n={t:,})".replace(",", ".") for c, t in contagem.sum(axis=1).items()], rotation=0)
    ax.set_title(f"{linha.upper()} × {coluna.upper()}\n{sufixo} | V de Cramér = {v_cramer:.2f}",
                 fontweight="bold")
    ax.set_xlabel(coluna)
    ax.set_ylabel(linha)
    return ax


def plotar_scatter(df, x, y, hue=None, ax=None, max_pontos=50_000, escala="linear", semente=42):
    """Dispersão x × y. Amostra as classes grandes e desenha a rara por cima."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    colunas = [x, y] + ([hue] if hue is not None else [])
    dados = df[colunas].replace([np.inf, -np.inf], np.nan).dropna()

    if hue is None:
        dados = dados.sample(min(len(dados), max_pontos), random_state=semente)
        ax.scatter(dados[x], dados[y], s=5, alpha=0.5)
    else:
        # value_counts vem da maior para a menor classe: a rara é desenhada por último
        for classe, n in dados[hue].value_counts().items():
            grupo = dados[dados[hue] == classe].sample(min(n, max_pontos), random_state=semente)
            ax.scatter(grupo[x], grupo[y], s=5, alpha=0.5, label=f"{classe} (n={n:,})".replace(",", "."))
        ax.legend(title=hue, markerscale=3)

    ax.set_xscale(escala)
    ax.set_yscale(escala)
    ax.set_title(f"DISPERSÃO: {str(x).upper()} × {str(y).upper()}", fontweight="bold")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def gerar_painel_quantitativo(df: pd.DataFrame, variaveis, hue=None, mapa_cores=None):
    """Gera o painel completo (Z-Score, Distribuição, Boxplot, QQ-Plot) para variáveis numéricas."""
    if isinstance(variaveis, str):
        variaveis = [variaveis]

    for variavel in variaveis:
        cor = _obter_cor_aleatoria()
        texto_resumo = _texto_resumo_estatistico(df, variavel)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Distribuições univariadas usam a mesma cor
        plotar_histograma_zscore(df, variavel=variavel, ax=axes[0, 0], cor=cor)
        plotar_histograma(df, variavel=variavel, ax=axes[0, 1], cor=cor)

        # O boxplot usa hue/mapa_cores, se informados, para separar grupos (ex.: fraude x legítima)
        plotar_boxplot(df, variavel=variavel, ax=axes[1, 0], hue=hue,
                       mapa_cores=mapa_cores, cor=cor)

        plotar_qq_normalidade(df, variavel=variavel, ax=axes[1, 1])

        fig.suptitle(f"Análise quantitativa: {str(variavel).upper()}", fontsize=20, fontweight="bold")

        plt.subplots_adjust(right=0.78, hspace=0.3, wspace=0.2)

        fig.text(0.80, 0.5, texto_resumo, fontsize=11, family="monospace", va="center",
                 bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="black", linewidth=1.5))

        plt.show()


def plotar_composicao_horizontal(df, variavel, alvo, ax=None, mapa_cores=None,
                                 ordenar_por=None, ordem_classes=None, n_minimo=30):
    """
    Barras 100% empilhadas na horizontal: uma linha por categoria de `variavel`,
    com o % de cada valor de `alvo` dentro dela. Boa para variáveis com muitas categorias.

    ordenar_por: classe do alvo usada para ordenar as linhas (ex.: "Poor"). Ela vira o
                 primeiro pedaço da barra e ganha uma linha tracejada com a média geral.
                 Sem ela, as linhas seguem a frequência de cada categoria.
    ordem_classes: ordem dos pedaços da barra (ex.: ["Poor", "Fair", "Good", "Excellent"]).
    n_minimo: categorias com menos registros que isso ficam de fora (evita que grupos
              minúsculos subam para o topo só por ruído). Use 1 para mostrar tudo.
    `variavel` também aceita lista (ex.: ["Gender", "Country"]): cada linha vira "Female | Brazil".
    """
    variaveis = [variavel] if isinstance(variavel, str) else list(variavel)
    base = df.dropna(subset=variaveis + [alvo])

    categoria = base[variaveis[0]].astype(str)
    for v in variaveis[1:]:
        categoria = categoria + " | " + base[v].astype(str)

    tabela = pd.crosstab(categoria, base[alvo], normalize="index").mul(100)
    contagem = categoria.value_counts()
    ocultas = int((contagem < n_minimo).sum())
    if ocultas:
        print(f"{ocultas} categoria(s) com menos de {n_minimo} registros ficaram de fora.")
    contagem = contagem[contagem >= n_minimo]
    tabela = tabela.loc[contagem.index]

    colunas = list(ordem_classes) if ordem_classes else list(tabela.columns)
    if ordenar_por is not None:
        colunas = [ordenar_por] + [c for c in colunas if c != ordenar_por]
        linhas = tabela[ordenar_por].sort_values(ascending=False).index
    else:
        linhas = contagem.index
    tabela = tabela.reindex(index=linhas, columns=colunas, fill_value=0)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(tabela))))

    if isinstance(mapa_cores, dict):
        cores = [mapa_cores[c] for c in colunas]
    else:
        cores = sns.color_palette(mapa_cores or "Set2", len(colunas))

    tabela.plot(kind="barh", stacked=True, ax=ax, color=cores, edgecolor="black", width=0.8)

    # % no meio de cada pedaço (esconde os menores que 4% para não poluir)
    for barras in ax.containers:
        ax.bar_label(barras, labels=[f"{v:.1f}%" if v >= 4 else "" for v in barras.datavalues],
                     label_type="center", fontsize=8)

    ax.set_yticklabels([f"{c} (n={contagem[c]:,})".replace(",", ".") for c in tabela.index])
    ax.invert_yaxis()  # primeira categoria no topo

    if ordenar_por is not None:
        media = (base[alvo] == ordenar_por).mean() * 100
        ax.axvline(media, color="black", linestyle="--", linewidth=1.5,
                   label=f"Média geral de {ordenar_por} ({media:.1f}%)")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentual (%)")
    ax.set_ylabel("")
    # legenda em uma linha acima do gráfico (não briga com a caixa de resumo do painel)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(colunas) + 1,
              frameon=False, fontsize=9)
    ax.set_title(f"% DE {str(alvo).upper()} DENTRO DE CADA {' × '.join(variaveis).upper()}",
                 fontweight="bold", pad=28)
    return ax


# ---------------------------------------------------------------------------
# Gráficos categóricos
# ---------------------------------------------------------------------------

def plotar_barras_absolute(df: pd.DataFrame, variavel: str, alvo: str, ax=None, mapa_cores=None):
    """Contagem absoluta por categoria, separada pela variável alvo, em escala log."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    titulo = f"CONTAGEM POR ALVO (LOG): {str(variavel).upper()}"
    ordem = _ordem_por_frequencia(df, variavel)
    if not ordem:
        return _eixo_sem_dados(ax, titulo, "Sem dados")

    # hue separa as classes do alvo; a ordem segue a frequência (igual ao gráfico de frequência)
    sns.countplot(data=df, x=variavel, hue=alvo, order=ordem,
                  palette=mapa_cores or "Set2", ax=ax, edgecolor="black")

    # Escala log obrigatória, senão classes raras (ex.: 0,1% de fraudes) ficam invisíveis
    ax.set_yscale("symlog")

    ax.set_title(titulo, fontweight="bold")
    ax.set_ylabel("Contagem (escala log)")
    ax.set_xlabel(variavel)
    ax.tick_params(axis="x", rotation=90)
    return ax


def plotar_barras_taxa_por_categoria(df, variavel, alvo, ax=None, mapa_cores=None, empilhado=True):
    variaveis = [variavel] if isinstance(variavel, str) else list(variavel)

    contagem = df.groupby(variaveis, observed=True).size()
    if len(variaveis) == 1:
        contagem = contagem.sort_values(ascending=False)  # mesma ordem da frequência relativa
    tabela = (pd.crosstab([df[v] for v in variaveis], df[alvo], normalize="index")
              .mul(100).reindex(contagem.index))

    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, 1.1 * len(tabela)), 5))

    if isinstance(mapa_cores, dict):
        cores = [mapa_cores[c] for c in tabela.columns]
    else:
        cores = sns.color_palette(mapa_cores or "Set2", tabela.shape[1])

    tabela.plot(kind="bar", stacked=empilhado, ax=ax, color=cores, edgecolor="black", width=0.8)

    # % em cada barra: no meio da pilha (esconde < 3%) ou em cima da coluna
    for barras in ax.containers:
        if empilhado:
            ax.bar_label(barras, labels=[f"{v:.1f}%" if v >= 3 else "" for v in barras.datavalues],
                         label_type="center", fontsize=9)
        else:
            ax.bar_label(barras, fmt="%.1f%%", fontsize=8, padding=2)

    # Rótulo de cada barra: valor da(s) variável(is) interna(s) + n
    chaves = [k if isinstance(k, tuple) else (k,) for k in contagem.index]
    internos = ["\n".join(map(str, k[1:])) if len(k) > 1 else str(k[0]) for k in chaves]
    ax.set_xticklabels([f"{r}\n(n={n:,})".replace(",", ".") for r, n in zip(internos, contagem)], rotation=0)

    # Segmentação: linha entre os grupos da 1ª variável e o nome do grupo embaixo
    if len(variaveis) > 1:
        externos = [k[0] for k in chaves]
        inicio = 0
        for i in range(1, len(externos) + 1):
            if i == len(externos) or externos[i] != externos[inicio]:
                if i < len(externos):
                    ax.axvline(i - 0.5, color="black", linewidth=1.5)
                ax.annotate(str(externos[inicio]), xy=((inicio + i - 1) / 2, 0),
                            xycoords=("data", "axes fraction"), xytext=(0, -(13 * len(variaveis) + 12)),
                            textcoords="offset points", ha="center", va="top", fontweight="bold")
                inicio = i

    ax.set_title(f"% DE {str(alvo).upper()} DENTRO DE CADA {' × '.join(variaveis).upper()}", fontweight="bold")
    ax.set_ylabel("Percentual (%)")
    ax.set_xlabel(" × ".join(variaveis), labelpad=22 if len(variaveis) > 1 else 4)
    ax.set_ylim(0, 100 if empilhado else tabela.to_numpy().max() * 1.12)  # folga para os rótulos

    # na pilha, a legenda é invertida para seguir a ordem visual (de cima para baixo)
    alcas, rotulos = ax.get_legend_handles_labels()
    if empilhado:
        alcas, rotulos = alcas[::-1], rotulos[::-1]
    ax.legend(alcas, rotulos, title=alvo, bbox_to_anchor=(1.01, 1), loc="upper left")
    return ax


def plotar_barras_frequencia_variavel(df: pd.DataFrame, variavel: str, ax=None, cor=None, mapa_cores=None):
    """Frequência relativa (%) de cada categoria, da mais para a menos frequente."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    titulo = f"FREQUÊNCIA RELATIVA: {str(variavel).upper()}"

    frequencias = df[variavel].value_counts(normalize=True).mul(100)
    frequencias = frequencias[frequencias > 0]
    if frequencias.empty:
        return _eixo_sem_dados(ax, titulo, "Sem dados")

    tabela_freq = pd.DataFrame({variavel: frequencias.index, "percentual": frequencias.to_numpy()})
    ordem = tabela_freq[variavel].tolist()

    if mapa_cores:
        # O seaborn moderno exige hue ao usar uma paleta mapeada por categoria
        sns.barplot(data=tabela_freq, x=variavel, y="percentual", order=ordem,
                    hue=variavel, hue_order=ordem, palette=mapa_cores, ax=ax, legend=False)
    else:
        sns.barplot(data=tabela_freq, x=variavel, y="percentual", order=ordem,
                    color=cor or _obter_cor_aleatoria(), ax=ax)

    _rotular_barras(ax, casas_decimais=1)

    ax.set_title(titulo, fontweight="bold")
    ax.set_ylabel("Percentual (%)")
    ax.set_xlabel(variavel)
    ax.tick_params(axis="x", rotation=90)
    return ax


def gerar_painel_categorico(df: pd.DataFrame, variaveis, alvo=None, mapa_cores=None):
    """Gera o painel completo para variáveis categóricas, opcionalmente cruzando com um alvo."""
    if isinstance(variaveis, str):
        variaveis = [variaveis]

    for variavel in variaveis:
        texto_resumo = _texto_resumo_categorico(df, variavel)

        # Com variável alvo (ex.: isFraud): painel triplo
        if alvo is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            plotar_barras_frequencia_variavel(df, variavel=variavel, ax=axes[0])
            plotar_barras_absolute(df, variavel=variavel, alvo=alvo, ax=axes[1], mapa_cores=mapa_cores)
            plotar_barras_taxa_por_categoria(df, variavel=variavel, alvo=alvo, ax=axes[2], empilhado=False)

            # Espaço extra à direita para a caixa de texto
            plt.subplots_adjust(right=0.82, wspace=0.3)
            posicao_texto = 0.84

        # Sem alvo: painel duplo mais simples
        else:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            plotar_barras_frequencia_variavel(df, variavel=variavel, ax=axes[0])

            titulo_contagem = f"CONTAGEM ABSOLUTA: {str(variavel).upper()}"
            ordem = _ordem_por_frequencia(df, variavel)
            if ordem:
                sns.countplot(data=df, x=variavel, order=ordem, ax=axes[1],
                              color=_obter_cor_aleatoria(), edgecolor="black")
                axes[1].set_title(titulo_contagem, fontweight="bold")
                axes[1].set_ylabel("Contagem")
                axes[1].tick_params(axis="x", rotation=90)
            else:
                _eixo_sem_dados(axes[1], titulo_contagem, "Sem dados")

            plt.subplots_adjust(right=0.78, wspace=0.3)
            posicao_texto = 0.80

        fig.suptitle(f"Análise categórica: {str(variavel).upper()}", fontsize=20, fontweight="bold", y=1.02)

        # Caixa de texto padronizada
        fig.text(posicao_texto, 0.5, texto_resumo, fontsize=11, family="monospace", va="center",
                 bbox=dict(boxstyle="round,pad=1", facecolor="white", edgecolor="black", linewidth=1.5))

        plt.show()