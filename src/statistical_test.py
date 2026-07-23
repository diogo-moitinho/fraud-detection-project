"""Bateria de testes estatísticos (a sua parte de 'teste').
Compara cada alvo contra o alvo para ver se ela tem relação com a inadimplência.
"""

import logging

import pandas as pd
import statsmodels.api as sm
from scipy.stats import (normaltest, levene, ttest_ind, mannwhitneyu, chi2_contingency)
from src.config import TARGET_COLUMN

logger = logging.getLogger(__name__)


def _teste_numerico(grupo_a, grupo_b, variavel, relatorio, baixa_relevancia, alpha=0.05):
    """Compara a distribuição de uma alvo numérica entre os dois grupos (0 e 1)."""
    _, p_a = normaltest(grupo_a)
    _, p_b = normaltest(grupo_b)

    if p_a < alpha or p_b < alpha:
        # Pelo menos um grupo não é normal -> teste não-paramétrico.
        distribuicao = 'NÃO NORMAL'
        teste = 'Mann-Whitney U'
        _, p_final = mannwhitneyu(grupo_a, grupo_b, alternative='two-sided')
    else:
        distribuicao = 'NORMAL'
        _, p_levene = levene(grupo_a, grupo_b)
        variancias_iguais = p_levene > alpha
        teste = 'T-Test Padrão' if variancias_iguais else 'T-Test de Welch'
        _, p_final = ttest_ind(grupo_a, grupo_b, equal_var=variancias_iguais)

    if p_final < alpha:
        veredito = 'REJEITAMOS H0. Diferença SIGNIFICATIVA entre os grupos.'
    else:
        veredito = 'NÃO REJEITAMOS H0. Sem diferença significativa.'
        baixa_relevancia.append(variavel)  # só entra aqui quem NÃO diferencia os grupos

    logger.info(f'[{variavel}] {distribuicao} | {teste} | p={p_final:.4f} -> {veredito}')
    relatorio[variavel] = {'Distribuicao': distribuicao, 'Teste': teste,
                           'P-valor': p_final, 'Veredito': veredito}


def _teste_categorico(df, col_grupo, relatorio, alpha=0.05):
    """Qui-Quadrado: verifica se a inadimplência depende de uma alvo categórica."""
    tabela = pd.crosstab(df[col_grupo], df[TARGET_COLUMN])
    _, p_valor, _, _ = chi2_contingency(tabela)

    if p_valor < alpha:
        veredito = f'A alvo DEPENDE de {col_grupo}.'
    else:
        veredito = f'A alvo é INDEPENDENTE de {col_grupo}.'

    logger.info(f'[{col_grupo}] Qui-Quadrado | p={p_valor:.4f} -> {veredito}')
    relatorio[col_grupo] = {'Teste': 'Qui-Quadrado', 'P-valor': p_valor, 'Veredito': veredito}

    # Post-hoc: quais categorias puxam a alvo para cima/baixo.
    residuos = sm.stats.Table(tabela).standardized_resids
    for idx, row in residuos.iterrows():
        if 1 not in row.index:
            break
        r = row[1]
        if r > 1.96:
            logger.info(f'    {idx}: puxa a alvo para CIMA  (resíduo {r:.2f})')
        elif r < -1.96:
            logger.info(f'    {idx}: puxa a alvo para BAIXO (resíduo {r:.2f})')


def aplicar_testes(df: pd.DataFrame):
    """Roda todos os testes e devolve (relatorio, variaveis_baixa_relevancia)."""
    logger.info('Iniciando bateria de testes estatísticos.')
    relatorio = {}
    baixa_relevancia = []

    numericas = [c for c in df.columns
                 if pd.api.types.is_numeric_dtype(df[c])
                 and c.lower() != 'id' and c != TARGET_COLUMN]
    categoricas = [c for c in df.columns
                   if c not in numericas and c.lower() != 'id' and c != TARGET_COLUMN]

    for coluna in numericas:
        grupo_a = df.loc[df[TARGET_COLUMN] == 0, coluna].dropna()
        grupo_b = df.loc[df[TARGET_COLUMN] == 1, coluna].dropna()
        _teste_numerico(grupo_a, grupo_b, coluna, relatorio, baixa_relevancia)

    for coluna in categoricas:
        _teste_categorico(df, coluna, relatorio)

    return relatorio, baixa_relevancia