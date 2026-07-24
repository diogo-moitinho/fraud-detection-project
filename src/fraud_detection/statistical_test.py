"""
Bateria de testes estatísticos.
Compara as variáveis independentes contra a variável alvo (target) 
para avaliar dependência e relevância preditiva.
"""

import logging
import pandas as pd
import statsmodels.api as sm
from scipy.stats import (normaltest, levene, ttest_ind, mannwhitneyu, chi2_contingency)
from fraud_detection.config import TARGET_COLUMN

logger = logging.getLogger(__name__)


def registrar_relatorio(relatorio: dict) -> None:
    """Registra o relatório final formatado através do logger."""
    print("=== RESUMO DOS TESTES ESTATÍSTICOS ===")
    for variavel, dicionario in relatorio.items():
        print(f"Análise Variável: {variavel}")
        for chave, valor in dicionario.items():
            print(f"    |-- {chave}: {valor}")
    print("======================================")


def _teste_numerico(grupo_a: pd.Series, grupo_b: pd.Series, variavel: str, relatorio: dict, baixa_relevancia: list, alpha: float = 0.05) -> None:
    """Compara a distribuição de uma variável numérica entre os dois grupos (0 e 1 do target)."""
    _, p_a = normaltest(grupo_a)
    _, p_b = normaltest(grupo_b)

    if p_a < alpha or p_b < alpha:
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
        baixa_relevancia.append(variavel)

    logger.info(f"[{variavel}] {distribuicao} | {teste} | p-valor={p_final:.4e} -> {veredito}")
    relatorio[variavel] = {
        'Distribuicao': distribuicao, 
        'Teste': teste,
        'P-valor': p_final, 
        'Veredito': veredito
    }


def _teste_categorico(df: pd.DataFrame, col_grupo: str, relatorio: dict, alpha: float = 0.05) -> None:
    """Qui-Quadrado: verifica se a variável alvo depende de uma variável categórica."""
    tabela = pd.crosstab(df[col_grupo], df[TARGET_COLUMN])
    _, p_valor, _, _ = chi2_contingency(tabela)

    if p_valor < alpha:
        veredito = f'A variável alvo DEPENDE de {col_grupo}.'
    else:
        veredito = f'A variável alvo é INDEPENDENTE de {col_grupo}.'

    logger.info(f"[{col_grupo}] Qui-Quadrado | p-valor={p_valor:.4e} -> {veredito}")
    relatorio[col_grupo] = {
        'Teste': 'Qui-Quadrado', 
        'P-valor': p_valor, 
        'Veredito': veredito
    }

    # Post-hoc: Detalhamento de quais categorias influenciam o target (Log nível DEBUG)
    residuos = sm.stats.Table(tabela).standardized_resids
    for idx, row in residuos.iterrows():
        if 1 not in row.index:
            break
        r = row[1]
        if r > 1.96:
            logger.debug(f"[{col_grupo}] Categoria '{idx}' puxa o target para CIMA (resíduo {r:.2f})")
        elif r < -1.96:
            logger.debug(f"[{col_grupo}] Categoria '{idx}' puxa o target para BAIXO (resíduo {r:.2f})")


def aplicar_testes(df: pd.DataFrame) -> tuple[dict, list]:
    """
    Roda todos os testes estatísticos e devolve o relatório e a lista de variáveis irrelevantes.
    """
    logger.info("Iniciando bateria de testes estatísticos...")
    relatorio = {}
    baixa_relevancia = []

    # Separação dinâmica das colunas, ignorando chaves primárias e o target
    colunas_ignoradas = {'id', TARGET_COLUMN.lower()}
    
    numericas = [c for c in df.columns
                 if pd.api.types.is_numeric_dtype(df[c])
                 and c.lower() not in colunas_ignoradas]
                 
    categoricas = [c for c in df.columns
                   if c not in numericas 
                   and c.lower() not in colunas_ignoradas]

    for coluna in numericas:
        grupo_a = df.loc[df[TARGET_COLUMN] == 0, coluna].dropna()
        grupo_b = df.loc[df[TARGET_COLUMN] == 1, coluna].dropna()
        
        # Trava de segurança caso a coluna fique vazia após o dropna
        if not grupo_a.empty and not grupo_b.empty:
            _teste_numerico(grupo_a, grupo_b, coluna, relatorio, baixa_relevancia)
        else:
            logger.warning(f"[{coluna}] Ignorada: Grupos vazios após remoção de nulos.")

    for coluna in categoricas:
        _teste_categorico(df, coluna, relatorio)

    registrar_relatorio(relatorio)
    logger.info("Bateria de testes estatísticos concluída.")
    
    return relatorio, baixa_relevancia