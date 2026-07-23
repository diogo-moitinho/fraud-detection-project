"""
Modelo Classificação: treino com validação cruzada (StratifiedKFold) e busca de
hiperparâmetros com Optuna.
"""

import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import joblib
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, classification_report

from config import RANDOM_SEED, TARGET_COLUMN, N_ESTAMATORS

logger = logging.getLogger(__name__)


def _params_base(num_classes: int) -> dict:
    """Parâmetros fixos que todo modelo desse projeto usa."""
    return dict(
        objective='multiclass',
        num_class=num_classes,
        metric='multi_logloss',
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )


def treinar(X, y, X_test, cat_cols, cls2i, test_ids, params=None, n_splits=5,
            salvar_modelo=True, pasta_modelos='modelos_salvos'):
    """Treina em KFold, imprime o desempenho e devolve (submissao, oof_balanced_accuracy).

    Passe em 'params' o dicionário de hiperparâmetros que o Optuna encontrou.
    Se 'params' for None, usa uns valores razoáveis de baseline.

    Com salvar_modelo=True, cada rodada é gravada em 'pasta_modelos' num arquivo
    com data/hora e o score no nome (uma rodada nunca sobrescreve a outra).
    """
    num_classes = len(np.unique(y))

    p = _params_base(num_classes)
    p['n_estimators'] = 300
    if params:
        p.update(params)
    # Blindagem: garante que o essencial não seja sobrescrito por engano.
    p['objective'] = 'multiclass'
    p['num_class'] = num_classes

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    oof       = np.zeros((len(X), num_classes))
    test_pred = np.zeros((len(X_test), num_classes))
    modelos   = []  # guarda o modelo de cada fold (o conjunto reproduz a previsão)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**p)
        model.fit(
            X.iloc[tr_idx], y[tr_idx],
            eval_set=[(X.iloc[va_idx], y[va_idx])],
            eval_metric='multi_logloss',
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        modelos.append(model)
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])
        test_pred += model.predict_proba(X_test) / n_splits

        fold_ba = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        print(f'fold {fold}: balanced_acc = {fold_ba:.5f}  (best_iter={model.best_iteration_})')

    oof_ba = balanced_accuracy_score(y, oof.argmax(1))
    print(f'\n==> OOF balanced accuracy = {oof_ba:.5f}  <== confie NESTE número, não no LB')
    print('\nRelatório por classe:')
    print(classification_report(y, oof.argmax(1),
                                target_names=[str(c) for c in np.unique(y)], digits=4))

    # Converte os índices previstos de volta para os rótulos originais.
    i2cls = {i: c for c, i in cls2i.items()}
    pred_labels = np.array([i2cls[i] for i in test_pred.argmax(1)])

    submissao = pd.DataFrame({TARGET_COLUMN: pred_labels})
    if test_ids is not None:
        submissao.insert(0, 'id', test_ids.values)

    if salvar_modelo:
        os.makedirs(pasta_modelos, exist_ok=True)
        carimbo = datetime.now().strftime('%Y%m%d_%H%M%S')
        caminho = os.path.join(pasta_modelos, f'lgbm_{carimbo}_ba{oof_ba:.4f}.joblib')
        bundle = {
            'modelos': modelos,            # lista com os modelos de cada fold
            'features': list(X.columns),   # ordem das colunas usada no treino
            'cat_cols': cat_cols,
            'cls2i': cls2i,                # mapa rótulo -> índice (para reverter a previsão)
            'params': p,
            'oof_ba': oof_ba,
            'n_splits': n_splits,
            'criado_em': carimbo,
        }
        joblib.dump(bundle, caminho)
        logger.info(f'Modelo salvo em: {caminho}')

    return submissao, oof_ba



def otimizar(X, y, cat_cols, n_trials=30):
    """Roda o Optuna com amostragem e validação Holdout."""
    num_classes = len(np.unique(y))
    
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=0.25, stratify=y, random_state=RANDOM_SEED
    )

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_sub, y_sub, test_size=0.25, stratify=y_sub, random_state=RANDOM_SEED
    )

    def objective(trial):
        params = _params_base(num_classes)
        params.update({
            'n_estimators': 1500, 
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 0.9),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        })

        model = lgb.LGBMClassifier(**params)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='multi_logloss',
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        
        preds = model.predict(X_va)
        return balanced_accuracy_score(y_va, preds)

    study = optuna.create_study(direction='maximize', study_name='LGBM_Tuning')
    study.optimize(objective, n_trials=n_trials)

    logger.info(f'Melhor Balanced Accuracy no Optuna: {study.best_value:.5f}')
    return study


def carregar_modelo(caminho: str) -> dict:
    """Carrega um modelo salvo pelo treinar() (bundle com os folds + metadados)."""
    bundle = joblib.load(caminho)
    logger.info(f"Modelo carregado ({bundle['criado_em']}, OOF={bundle['oof_ba']:.4f}).")
    return bundle


def prever(bundle: dict, X) -> np.ndarray:
    """Prevê com um modelo carregado, fazendo a média dos folds e devolvendo os rótulos.

    'X' precisa estar preparado igual ao X_test (rode features.preparar_dados antes).
    """
    modelos = bundle['modelos']
    proba = sum(m.predict_proba(X[bundle['features']]) for m in modelos) / len(modelos)
    i2cls = {i: c for c, i in bundle['cls2i'].items()}
    return np.array([i2cls[i] for i in proba.argmax(1)])

def importancias(bundle: dict, tipo: str = 'gain') -> pd.DataFrame:
    """Importância média das features entre os folds, com o desvio-padrão.
 
    tipo='gain'  -> quanto cada feature melhorou o modelo (o que interessa na prática)
    tipo='split' -> quantas vezes a feature foi usada para cortar (o padrão do LightGBM)
 
    Devolve um DataFrame ordenado: feature | importancia | desvio.
    """
    features = bundle['features']
    # Uma linha por fold, uma coluna por feature.
    matriz = np.array([
        m.booster_.feature_importance(importance_type=tipo) for m in bundle['modelos']
    ])
    df = pd.DataFrame({
        'feature': features,
        'importancia': matriz.mean(axis=0),
        'desvio': matriz.std(axis=0),  # variação entre os folds: alto = instável
    })
    return df.sort_values('importancia', ascending=False).reset_index(drop=True)



def plotar_importancias(bundle: dict, top: int = 20, tipo: str = 'gain', figsize=(8, 6)):
    """Gráfico de barras das features mais importantes (seaborn). Devolve o DataFrame."""
    import matplotlib.pyplot as plt
    import seaborn as sns
 
    df = importancias(bundle, tipo=tipo).head(top)
 
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=df, x='importancia', y='feature',
                     hue='feature', palette='viridis', legend=False)
    # A barra de erro mostra a variação entre os folds.
    ax.errorbar(x=df['importancia'], y=range(len(df)), xerr=df['desvio'],
                fmt='none', ecolor='gray', capsize=3, alpha=0.6)
    ax.set_title(f'Top {len(df)} features por {tipo} (média de {len(bundle["modelos"])} folds)')
    ax.set_xlabel(f'Importância ({tipo})')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.show()
 
    return df
