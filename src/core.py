"""
Core: ponto de entrada. Configura o log, carrega os dados e amarra
estatística -> features -> otimização -> treino -> submissão.
"""

import logging
import sys
import pandas as pd
from config import TRAIN_PATH, TEST_PATH
from statistical_test import aplicar_testes
from feature_engineer import preparar_dados
from model import otimizar, treinar

logger = logging.getLogger(__name__)


def configurar_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('modelagem.log', mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )


def carregar_dados():
    logger.info(f'Carregando treino: {TRAIN_PATH}')
    logger.info(f'Carregando teste:  {TEST_PATH}')
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def rodar_pipeline(df_treino, df_teste, tune=True, n_trials=30, salvar='submissao.csv'):
    """Roda o pipeline inteiro. Devolve (submissao, oof_balanced_accuracy)."""
    logger.info(f'Shapes -> treino: {df_treino.shape} | teste: {df_teste.shape}')

    logger.info('TESTES')
    aplicar_testes(df_treino)

    logger.info('PREPARANDO DADOS')
    X, X_test, y, cat_cols, cls2i, test_ids = preparar_dados(df_treino, df_teste)


    logger.info('OTIMIZACAO')
    # 3. Otimização (opcional).
    params = None
    if tune:
        study = otimizar(X, y, cat_cols, n_trials=n_trials)
        params = study.best_params
        print(f'\nMelhores hiperparâmetros (Balanced Acc = {study.best_value:.5f}):')
        for k, v in params.items():
            print(f"    '{k}': {v},")

    logger.info('TREINO FINAL')
    submissao, oof_ba = treinar(X, y, X_test, cat_cols, cls2i, test_ids, params=params)

    if salvar:
        submissao.to_csv(salvar, index=False)
        logger.info(f'Submissão salva em: {salvar}')

    return submissao, oof_ba


if __name__ == '__main__':
    configurar_logging()
    logger.info('Pipeline iniciada.')
    try:
        treino, teste = carregar_dados()
        rodar_pipeline(treino, teste, tune=True, n_trials=30)
        logger.info('Pipeline finalizada com sucesso.')
    except Exception:
        logger.exception('Erro durante a execução da pipeline.')