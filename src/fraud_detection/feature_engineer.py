"""Feature Engineering: criação de novas colunas e preparação dos dados
(codificação das categóricas + alvo em inteiro) para o LightGBM."""

import pandas as pd
import numpy as np

from fraud_detection.config import TARGET_COLUMN




def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria novas colunas a partir das existentes.
    NUNCA usa health_condition (alvo).
    """

    return df

def preparar_dados(df_treino: pd.DataFrame, df_teste: pd.DataFrame):
    """
    Aplica as features, codifica as categóricas de forma consistente entre
    treino e teste, e transforma o alvo em inteiro.

    Retorna: X, X_test, y, cat_cols, cls2i, test_ids
    """
    train = criar_features(df_treino)
    test  = criar_features(df_teste)

    # Guarda o id do teste (para a submissão), mas ele NÃO entra como feature.
    id_cols = [c for c in test.columns if c.lower() == 'id']
    test_ids = test[id_cols[0]] if id_cols else None

    features = [c for c in train.columns if c != TARGET_COLUMN and c.lower() != 'id']
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]

    X, X_test = train[features].copy(), test[features].copy()
    for c in cat_cols:
        # União dos níveis => treino e teste usam a mesma codificação (evita nível não visto).
        niveis = pd.Index(pd.concat([X[c], X_test[c]]).astype('object').dropna().unique())
        dtype = pd.CategoricalDtype(categories=niveis)
        X[c]      = X[c].astype('object').astype(dtype)
        X_test[c] = X_test[c].astype('object').astype(dtype)

    # Alvo em texto -> inteiro (guardamos a ordem para mapear de volta na submissão).
    classes = np.sort(train[TARGET_COLUMN].unique())
    cls2i = {c: i for i, c in enumerate(classes)}
    y = train[TARGET_COLUMN].map(cls2i).values

    return X, X_test, y, cat_cols, cls2i, test_ids
