"""Comparação de variáveis entre grupos

Cada função imprime uma conclusão curta (o que foi encontrado, o quanto é forte
e se é confiável) e devolve os números em um dicionário. 

    comparar_dois_grupos(df, variavel, grupo)   -> 2 grupos, variável numérica
    comparar_n_grupos(df, variavel, grupo)      -> 3+ grupos, variável numérica
    comparar_categoricas(df, variavel, grupo)   -> duas variáveis categóricas
"""

import contextlib
import functools
import io
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import (
    chi2_contingency, f_oneway, kruskal, levene, mannwhitneyu,
    normaltest, shapiro, ttest_ind,
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

LIMIAR_RESIDUO = 1.96
LIMITE_SHAPIRO = 5000       # acima disso o Shapiro fica sensível demais
MIN_OBS_NORMALIDADE = 20    # abaixo disso não dá para avaliar normalidade
LIMIAR_REDUNDANCIA = 0.90   # associação forte demais: uma variável deve derivar da outra
VERBOSE = False

# Faixas de referência por métrica: (pequeno, médio, grande).
FAIXAS = {
    "Cohen's d": (0.2, 0.5, 0.8),
    "Rank-biserial": (0.1, 0.3, 0.5),
    "Eta²": (0.01, 0.06, 0.14),
    "Epsilon²": (0.01, 0.06, 0.14),
    "Cramér's V": (0.1, 0.3, 0.5),
}


# --------------------------------------------------------------------------- #
# Auxiliares
# --------------------------------------------------------------------------- #

def _silenciavel(funcao):
    """Acrescenta o parâmetro verbose (padrão True); com False, nada é impresso."""
    @functools.wraps(funcao)
    def envolvida(*args, verbose=VERBOSE, **kwargs):
        if verbose:
            return funcao(*args, **kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            return funcao(*args, **kwargs)
    return envolvida


def _magnitude(efeito, metrica):
    """Traduz o tamanho de efeito em uma palavra."""
    if efeito is None or np.isnan(efeito):
        return "indefinida"
    pequeno, medio, grande = FAIXAS[metrica]
    valor = abs(efeito)
    if valor < pequeno:
        return "DESPREZÍVEL"
    if valor < medio:
        return "PEQUENA"
    if valor < grande:
        return "MÉDIA"
    return "GRANDE"


def _formatar_p(p_valor):
    return "p < 0.001" if p_valor < 0.001 else f"p = {p_valor:.3f}"


def _cabecalho(titulo):
    print(f"\n{'─' * 64}\n{titulo}\n{'─' * 64}")


def _separar(df, variavel, grupo, ordem=None):
    """Devolve (nomes dos grupos, lista de Series com os valores).

    Sem `ordem`, categóricas seguem a ordem das categorias (Poor, Fair, ...)
    e as demais colunas, a ordem alfabética/crescente.
    """
    if not pd.api.types.is_numeric_dtype(df[variavel]):
        raise TypeError(f"'{variavel}' não é numérica. Use comparar_categoricas().")
    serie = df[grupo]
    if ordem is None:
        if isinstance(serie.dtype, pd.CategoricalDtype):
            ordem = serie.cat.remove_unused_categories().cat.categories
        else:
            ordem = sorted(serie.dropna().unique())
    nomes = list(ordem)
    return nomes, [df.loc[serie == nome, variavel].dropna() for nome in nomes]


def _diagnosticar(amostras, alpha):
    """Normalidade (dentro de cada grupo) e homocedasticidade (no conjunto).
    """
    if min(len(a) for a in amostras) < MIN_OBS_NORMALIDADE:
        normal = False
    else:
        normal = all((shapiro if len(a) <= LIMITE_SHAPIRO else normaltest)(a)[1] >= alpha
                     for a in amostras)
    variancias_iguais = levene(*amostras, center="median")[1] > alpha
    return normal, variancias_iguais


def _alerta_redundancia(efeito, variavel, grupo):
    """Avisa quando a associação é forte demais para ser um achado real."""
    if abs(efeito) >= LIMIAR_REDUNDANCIA:
        print(f"\n  ATENÇÃO: associação quase perfeita. Verifique se '{variavel}' não foi\n"
              f"  derivada de '{grupo}' (ou vice-versa) — nesse caso o teste apenas\n"
              f"  redescobre a regra que criou a coluna.")


# --------------------------------------------------------------------------- #
# 2 grupos
# --------------------------------------------------------------------------- #

@_silenciavel
def comparar_dois_grupos(df, variavel, grupo, alpha=0.05, ordem=None):
    """Compara uma variável numérica entre exatamente 2 grupos.

    Normal + variâncias iguais      -> T-Test de Student
    Normal + variâncias diferentes  -> T-Test de Welch
    Não normal                      -> Mann-Whitney U
    """
    nomes, amostras = _separar(df, variavel, grupo, ordem)
    if len(nomes) != 2:
        raise ValueError(f"Esperados 2 grupos, encontrados {len(nomes)}: {nomes}")
    a, b = amostras
    _cabecalho(f"{variavel} — {nomes[0]} (n={len(a)}) vs {nomes[1]} (n={len(b)})")

    normal, variancias_iguais = _diagnosticar(amostras, alpha)
    if normal:
        teste = "T-Test de Student" if variancias_iguais else "T-Test de Welch"
        resultado = ttest_ind(b, a, equal_var=variancias_iguais)
        p_valor, ic = resultado.pvalue, resultado.confidence_interval(1 - alpha)
        base, centro_a, centro_b = "média", a.mean(), b.mean()
        var_pool = (((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                    / (len(a) + len(b) - 2))
        efeito, metrica = (centro_b - centro_a) / np.sqrt(var_pool), "Cohen's d"
    else:
        teste, base = "Mann-Whitney U", "mediana"
        estatistica, p_valor = mannwhitneyu(b, a, alternative="two-sided")
        ic, centro_a, centro_b = None, a.median(), b.median()
        efeito, metrica = 2 * estatistica / (len(a) * len(b)) - 1, "Rank-biserial"

    diferenca = centro_b - centro_a
    significativo = p_valor < alpha
    magnitude = _magnitude(efeito, metrica)

    maior, menor = (nomes[1], nomes[0]) if diferenca > 0 else (nomes[0], nomes[1])
    print(f"\n  {nomes[0]}: {base} {centro_a:.2f}   |   {nomes[1]}: {base} {centro_b:.2f}")
    if significativo:
        print(f"\n  {maior} tem {variavel} MAIOR que {menor}, por {abs(diferenca):.2f}.")
        print(f"  Essa diferença é {magnitude} e é confiável ({_formatar_p(p_valor)}).")
        if magnitude == "DESPREZÍVEL":
            print("  Mesmo confiável, a diferença é pequena demais para ter uso prático.")
    else:
        print(f"\n  Não há diferença entre {nomes[0]} e {nomes[1]} ({_formatar_p(p_valor)}).")

    return {
        "variavel": variavel, "grupos": nomes, "teste": teste,
        "normal": normal, "variancias_iguais": variancias_iguais,
        "p_valor": p_valor, "diferenca": diferenca,
        "ic": (ic.low, ic.high) if ic is not None else None,
        "metrica_efeito": metrica, "tamanho_efeito": efeito,
        "magnitude": magnitude, "significativo": significativo,
    }


# --------------------------------------------------------------------------- #
# N grupos
# --------------------------------------------------------------------------- #

def _posthoc(nomes, amostras, alpha, parametrico):
    """Comparação par a par: Tukey HSD (paramétrico) ou Mann-Whitney com correção de Holm.
    O Mann-Whitney não tem IC: no caso não paramétrico ele exigiria bootstrap.
    """
    if parametrico:
        valores = np.concatenate([a.to_numpy() for a in amostras])
        rotulos = np.concatenate([[str(n)] * len(a) for n, a in zip(nomes, amostras)])
        res = pairwise_tukeyhsd(valores, rotulos, alpha=alpha)
        pares = list(combinations(res.groupsunique, 2))  # mesma ordem dos resultados
        return pd.DataFrame({
            "grupo_1": [p[0] for p in pares], "grupo_2": [p[1] for p in pares],
            "diferenca": res.meandiffs,
            "ic_inferior": res.confint[:, 0], "ic_superior": res.confint[:, 1],
            "p_ajustado": res.pvalues, "significativo": res.reject,
        })

    pares = list(combinations(range(len(nomes)), 2))
    p_valores = [mannwhitneyu(amostras[j], amostras[i], alternative="two-sided")[1]
                 for i, j in pares]
    rejeita, p_ajustado, _, _ = multipletests(p_valores, alpha=alpha, method="holm")
    return pd.DataFrame({
        "grupo_1": [nomes[i] for i, _ in pares], "grupo_2": [nomes[j] for _, j in pares],
        "diferenca": [amostras[j].median() - amostras[i].median() for i, j in pares],
        "ic_inferior": np.nan, "ic_superior": np.nan,
        "p_ajustado": p_ajustado, "significativo": rejeita,
    })


@_silenciavel
def comparar_n_grupos(df, variavel, grupo, alpha=0.05, ordem=None):
    """Compara uma variável numérica entre 3 ou mais grupos.

    Normal + variâncias iguais  -> ANOVA de uma via (post-hoc: Tukey HSD)
    Caso contrário              -> Kruskal-Wallis   (post-hoc: Mann-Whitney + Holm)

    Retorna (resumo, tabela_posthoc). A tabela vem vazia se os grupos não diferem.
    """
    nomes, amostras = _separar(df, variavel, grupo, ordem)
    if len(nomes) < 3:
        raise ValueError(f"Use comparar_dois_grupos para {len(nomes)} grupos.")
    k, n_total = len(nomes), sum(len(a) for a in amostras)
    _cabecalho(f"{variavel} por {grupo} — {k} grupos, n={n_total}")

    normal, variancias_iguais = _diagnosticar(amostras, alpha)
    parametrico = normal and variancias_iguais
    if parametrico:
        teste, centro, metrica = "ANOVA de uma via", "média", "Eta²"
        estatistica, p_valor = f_oneway(*amostras)
        # Eta²: proporção da variância explicada pelo grupo.
        efeito = (estatistica * (k - 1)) / (estatistica * (k - 1) + n_total - k)
    else:
        # Com variâncias heterogêneas a ANOVA clássica infla o erro tipo I.
        teste, centro, metrica = "Kruskal-Wallis", "mediana", "Epsilon²"
        estatistica, p_valor = kruskal(*amostras)
        efeito = max(0.0, (estatistica - k + 1) / (n_total - k))

    significativo = p_valor < alpha
    magnitude = _magnitude(efeito, metrica)
    resumo = {"variavel": variavel, "grupos": nomes, "teste": teste,
              "normal": normal, "variancias_iguais": variancias_iguais,
              "estatistica": estatistica, "p_valor": p_valor,
              "metrica_efeito": metrica, "tamanho_efeito": efeito,
              "magnitude": magnitude, "significativo": significativo}

    # --- ranking dos grupos: a informação mais útil, em uma linha ---
    centros = [a.mean() if parametrico else a.median() for a in amostras]
    ranking = sorted(zip(nomes, centros), key=lambda item: item[1])
    print(f"\n  {variavel} por grupo ({centro}, do menor para o maior):")
    print("    " + "  <  ".join(f"{nome} {valor:.2f}" for nome, valor in ranking))

    if not significativo:
        print(f"\n  Os grupos NÃO diferem em {variavel} ({_formatar_p(p_valor)}).")
        print("  A variação entre eles é compatível com o acaso.")
        return resumo, pd.DataFrame()

    tabela = (_posthoc(nomes, amostras, alpha, parametrico)
              .sort_values("diferenca", key=abs, ascending=False).reset_index(drop=True))
    iguais = tabela[~tabela["significativo"]]

    # --- conclusão em português ---
    (menor_nome, menor_valor), (maior_nome, maior_valor) = ranking[0], ranking[-1]
    print(f"\n  Os grupos DIFEREM em {variavel}. "
          f"A diferença é {magnitude} ({metrica} = {efeito:.3f}, {_formatar_p(p_valor)}).")
    print(f"  Do menor para o maior: {menor_nome} ({menor_valor:.2f}) até {maior_nome} "
          f"({maior_valor:.2f}), uma distância de {maior_valor - menor_valor:.2f}.")

    if iguais.empty:
        print(f"  Todos os {len(tabela)} pares de grupos diferem entre si.")
    elif len(iguais) == len(tabela):
        # O teste global passou raspando, mas nenhum par sobrevive à correção.
        print("  Porém NENHUM par específico sobrevive à correção: o resultado global é frágil\n"
              "  e não dá para apontar quais grupos diferem.")
    else:
        print(f"\n  {len(tabela) - len(iguais)} dos {len(tabela)} pares diferem. "
              f"Os pares SEM diferença são:")
        for _, linha in iguais.iterrows():
            print(f"    {linha['grupo_1']} e {linha['grupo_2']} "
                  f"(podem ser tratados como um grupo só)")

    if magnitude == "DESPREZÍVEL":
        print("\n  Ressalva: apesar de confiável, o efeito é pequeno demais para ter uso prático.\n"
              "  Com amostra grande, quase tudo dá significativo.")
    _alerta_redundancia(efeito, variavel, grupo)
    return resumo, tabela


# --------------------------------------------------------------------------- #
# Categórica x categórica
# --------------------------------------------------------------------------- #

@_silenciavel
def comparar_categoricas(df, variavel, grupo, alpha=0.05, ordem_variavel=None, ordem_grupo=None):
    """Testa a associação entre duas variáveis categóricas (Qui-Quadrado).

    H0: as duas variáveis são independentes.

    Retorna (resumo, tabela_residuos), com uma linha por combinação de
    categorias, da mais fora do esperado para a menos.
    """
    tabela = pd.crosstab(df[variavel], df[grupo])
    if ordem_variavel is not None:
        tabela = tabela.reindex(index=ordem_variavel)
    if ordem_grupo is not None:
        tabela = tabela.reindex(columns=ordem_grupo)
    if min(tabela.shape) < 2:
        raise ValueError(f"Cada variável precisa de ao menos 2 categorias. "
                         f"Tabela {tabela.shape[0]}x{tabela.shape[1]}.")

    n = int(tabela.to_numpy().sum())
    _cabecalho(f"{variavel} x {grupo} — tabela {tabela.shape[0]}x{tabela.shape[1]}, n={n}")

    chi2, p_valor, gl, esperado = chi2_contingency(tabela)
    significativo = p_valor < alpha

    # O Qui-Quadrado é uma aproximação: com frequências esperadas baixas o
    # p-valor fica pouco confiável.
    minimo = esperado.min()
    pressuposto_ok = minimo >= 5 and (esperado >= 5).mean() >= 0.8

    # Cramér's V: o Qui-Quadrado cresce com o N, então sozinho não diz se a
    # associação é forte.
    cramers_v = np.sqrt(chi2 / (n * (min(tabela.shape) - 1)))
    magnitude = _magnitude(cramers_v, "Cramér's V")

    percentuais = tabela.div(tabela.sum(axis=1), axis=0) * 100
    print(f"\n  % de cada {grupo} dentro de cada {variavel}:")
    print("    " + percentuais.round(1).to_string().replace("\n", "\n    "))

    esperado_df = pd.DataFrame(esperado, index=tabela.index, columns=tabela.columns)
    tabela_residuos = (
        pd.DataFrame({"observado": tabela.stack(), "esperado": esperado_df.stack(),
                      "residuo": sm.stats.Table(tabela).standardized_resids.stack()})
        .rename_axis(["categoria", "grupo"]).reset_index()
        .sort_values("residuo", key=abs, ascending=False).reset_index(drop=True)
    )
    tabela_residuos["destaque"] = tabela_residuos["residuo"].abs() > LIMIAR_RESIDUO
    destaques = tabela_residuos[tabela_residuos["destaque"]]

    # --- conclusão em português ---
    if significativo:
        print(f"\n  '{variavel}' e '{grupo}' estão ASSOCIADAS. A associação é {magnitude} "
              f"(Cramér's V = {cramers_v:.3f}, {_formatar_p(p_valor)}).")
        if len(destaques):
            # Só os 3 desvios mais fortes: o resto raramente muda a leitura.
            print("\n  Combinações mais fora do esperado:")
            for _, linha in destaques.head(3).iterrows():
                direcao = "mais" if linha["residuo"] > 0 else "menos"
                print(f"    {linha['categoria']} tem {direcao} '{linha['grupo']}' do que "
                      f"o esperado ({linha['observado']} vs {linha['esperado']:.0f})")
            if len(destaques) > 3:
                print(f"    (+{len(destaques) - 3} outras — veja a tabela de resíduos devolvida)")
        if magnitude == "DESPREZÍVEL":
            print("\n  Ressalva: a associação é confiável mas fraca demais para ter uso prático.")
    else:
        print(f"\n  '{variavel}' e '{grupo}' são INDEPENDENTES ({_formatar_p(p_valor)}).")
        print("  Saber uma não ajuda a prever a outra.")

    if not pressuposto_ok:
        alternativa = ("o teste exato de Fisher" if tabela.shape == (2, 2)
                       else "agrupar as categorias raras")
        print(f"\n  ATENÇÃO: há células com poucos casos esperados (mínimo {minimo:.1f}).\n"
              f"  O p-valor é pouco confiável — considere {alternativa}.")
    _alerta_redundancia(cramers_v, variavel, grupo)

    return ({"variavel": variavel, "grupo": grupo, "teste": "Qui-Quadrado",
             "chi2": chi2, "gl": gl, "p_valor": p_valor,
             "cramers_v": cramers_v, "magnitude": magnitude,
             "esperado_minimo": minimo, "pressuposto_ok": pressuposto_ok,
             "significativo": significativo},
            tabela_residuos)