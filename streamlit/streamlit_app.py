"""Demonstração interativa do modelo de detecção de fraude em tempo real.

O app carrega o mesmo artefato produzido pelo notebook 02_model — pipeline,
threshold de operação e contrato de features no mesmo arquivo — e pontua
transações com ele. Nada é retreinado aqui.

Rodar localmente:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #

ARTIFACT_PATH = os.path.join("notebooks", "models", "fraud_realtime_v1.joblib")

TIPOS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# Campos que só existem depois que a transação liquidou. O modelo real-time é
# treinado sem eles de propósito: no momento da autorização, eles não existem.
POS_LIQUIDACAO = [
    "newbalanceOrig",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
]

# Medidos na EDA (01_eda) sobre o motor de regras que acompanha o PaySim.
INCUMBENTE_FLAG_RATE = 0.000003
DATASET_FRAUD_RATE = 0.001291
INCUMBENTE_TETO_RECALL = INCUMBENTE_FLAG_RATE / DATASET_FRAUD_RATE

CENARIOS = {
    "Transferência esvaziando a conta, de madrugada": {
        "tipo": "TRANSFER",
        "amount": 181_000.00,
        "oldbalanceOrg": 181_000.00,
        "oldbalanceDest": 0.00,
        "is_merchant_dest": 0,
        "hour_of_day": 3,
        "day_of_week": 2,
    },
    "Pagamento cotidiano a comerciante": {
        "tipo": "PAYMENT",
        "amount": 95.50,
        "oldbalanceOrg": 4_300.00,
        "oldbalanceDest": 0.00,
        "is_merchant_dest": 1,
        "hour_of_day": 14,
        "day_of_week": 3,
    },
    "Saque de rotina": {
        "tipo": "CASH_OUT",
        "amount": 1_200.00,
        "oldbalanceOrg": 8_500.00,
        "oldbalanceDest": 25_000.00,
        "is_merchant_dest": 0,
        "hour_of_day": 11,
        "day_of_week": 5,
    },
}


# --------------------------------------------------------------------------- #
# Carregamento do modelo
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Carregando o modelo…")
def carregar_artefato(caminho: str = ARTIFACT_PATH) -> dict:
    """Lê o artefato salvo pelo notebook de modelagem.

    O threshold vem junto do pipeline de propósito: um modelo salvo sem o seu
    ponto de operação não é utilizável — quem carregar vai chamar predict em
    0,5 e obter um recall que não tem relação com o reportado.
    """
    return joblib.load(caminho)


def pontuar(frame: pd.DataFrame, artefato: dict) -> pd.DataFrame:
    """Pontua transações e aplica o threshold guardado no artefato.

    Colunas pós-liquidação são recusadas, não ignoradas: quem as fornece está
    descrevendo uma transação que já se completou, quando prevenir não é mais
    possível.
    """
    vazadas = [c for c in POS_LIQUIDACAO if c in artefato["features"]]
    if vazadas:
        raise ValueError(
            f"o artefato declara features pós-liquidação {vazadas}; "
            "ele é forense, não serve para pontuação em tempo real"
        )

    faltando = [c for c in artefato["features"] if c not in frame.columns]
    if faltando:
        raise ValueError(f"faltam as colunas obrigatórias: {faltando}")

    X = frame[artefato["features"]]
    probabilidades = artefato["pipeline"].predict_proba(X)[:, 1]

    saida = frame.copy()
    saida["fraud_probability"] = probabilidades
    saida["flagged"] = (probabilidades >= artefato["threshold"]).astype(int)
    return saida


# --------------------------------------------------------------------------- #
# Estilo
# --------------------------------------------------------------------------- #

CSS = """
<style>
  .bloco-veredito {
    border: 1px solid rgba(255,255,255,.14);
    border-left-width: 3px;
    padding: 18px 20px;
    margin-bottom: 8px;
  }
  .bloco-veredito.alerta { border-left-color: #e74c3c; }
  .bloco-veredito.liberado { border-left-color: #2ecc71; }
  .veredito-titulo {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 4px;
  }
  .veredito-texto { opacity: .75; font-size: .92rem; margin: 0; }

  .trilho {
    position: relative;
    height: 10px;
    background: rgba(255,255,255,.08);
    margin: 26px 0 8px;
  }
  .trilho .preenchido { position: absolute; inset: 0 auto 0 0; }
  .trilho .preenchido.alerta { background: #e74c3c; }
  .trilho .preenchido.liberado { background: #2ecc71; }
  .trilho .corte {
    position: absolute;
    top: -7px;
    bottom: -7px;
    width: 2px;
    background: #ff5500;
  }
  .trilho .corte-rotulo {
    position: absolute;
    bottom: calc(100% + 5px);
    white-space: nowrap;
    font-size: .74rem;
    color: #ff5500;
    font-variant-numeric: tabular-nums;
  }
  .trilho-legenda {
    display: flex;
    justify-content: space-between;
    font-size: .76rem;
    opacity: .6;
    font-variant-numeric: tabular-nums;
  }

  .riscado { text-decoration: line-through; opacity: .45; }
</style>
"""


def barra_probabilidade(probabilidade: float, threshold: float, alerta: bool) -> str:
    """Desenha a probabilidade contra o threshold de operação."""
    classe = "alerta" if alerta else "liberado"

    # Perto da borda direita o rótulo sairia da tela, então ele vira para dentro.
    ancora = "right:4px" if threshold > 0.6 else "left:4px"

    return f"""
    <div class="trilho">
      <div class="preenchido {classe}" style="width:{probabilidade * 100:.4f}%"></div>
      <div class="corte" style="left:{threshold * 100:.4f}%">
        <span class="corte-rotulo" style="{ancora}">corte {threshold:.2%}</span>
      </div>
    </div>
    <div class="trilho-legenda">
      <span>0%</span>
      <span>100%</span>
    </div>
    """


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Detecção de fraude — PaySim",
    page_icon="◆",
    layout="wide",
)

st.markdown(CSS, unsafe_allow_html=True)

try:
    artefato = carregar_artefato()
except FileNotFoundError:
    st.error(
        f"Artefato não encontrado em `{ARTIFACT_PATH}`. "
        "Ele é versionado junto do repositório — confira se o arquivo veio no clone."
    )
    st.stop()

st.title("Detecção de fraude em tempo real")
st.markdown(
    "Modelo treinado sobre transações do **PaySim**, usando apenas informação "
    "disponível no momento da autorização. O objetivo do projeto não foi maximizar "
    "a acurácia reportada, e sim medir **quanto dela vem de vazamento** — por isso "
    "o split é cronológico e a métrica é PR-AUC, não acurácia."
)

col_esq, col_dir = st.columns([1, 1.15], gap="large")

# --------------------------------------------------------------------------- #
# Entrada
# --------------------------------------------------------------------------- #

with col_esq:
    st.subheader("Transação")

    cenario = st.selectbox(
        "Partir de um cenário",
        list(CENARIOS.keys()),
        help="Preenche o formulário abaixo. Você pode alterar qualquer campo depois.",
    )
    base = CENARIOS[cenario]

    tipo = st.selectbox(
        "Tipo", TIPOS, index=TIPOS.index(base["tipo"]),
    )
    amount = st.number_input(
        "Valor", min_value=0.0, value=base["amount"], step=100.0, format="%.2f",
    )
    oldbalance_org = st.number_input(
        "Saldo da origem antes da transação",
        min_value=0.0, value=base["oldbalanceOrg"], step=100.0, format="%.2f",
    )
    oldbalance_dest = st.number_input(
        "Saldo do destino antes da transação",
        min_value=0.0, value=base["oldbalanceDest"], step=100.0, format="%.2f",
    )

    is_merchant = st.checkbox(
        "Destinatário é comerciante",
        value=bool(base["is_merchant_dest"]),
        help="No PaySim, contas de comerciante têm o prefixo M no campo nameDest.",
    )

    c1, c2 = st.columns(2)
    with c1:
        hora = st.slider("Hora do dia", 0, 23, base["hour_of_day"])
    with c2:
        dia = st.slider(
            "Dia da semana", 0, 6, base["day_of_week"],
            help="Derivado de (step // 24) % 7. O PaySim não traz calendário real, "
                 "então o número é um índice cíclico, não segunda-feira.",
        )

    with st.expander("Por que não há campos de saldo final aqui"):
        st.markdown(
            "Estes quatro campos existem no dataset e elevam muito a métrica, "
            "mas **não existem no momento da autorização** — só depois que a "
            "transação liquida. No PaySim eles codificam o rótulo de forma quase "
            "tautológica, e é daí que vem boa parte da acurácia que se costuma "
            "reportar nesse dataset."
        )
        st.markdown(
            "".join(f"<div class='riscado'>{c}</div>" for c in POS_LIQUIDACAO),
            unsafe_allow_html=True,
        )
        st.caption(
            "Um modelo com esses campos é válido para revisão forense, nunca para "
            "prevenção. O artefato carregado aqui os declara como excluídos, e a "
            "função de pontuação recusa artefatos que os contenham."
        )

# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #

entrada = pd.DataFrame([{
    "type": tipo,
    "amount": amount,
    "oldbalanceOrg": oldbalance_org,
    "oldbalanceDest": oldbalance_dest,
    "is_merchant_dest": int(is_merchant),
    "hour_of_day": hora,
    "day_of_week": dia,
}])

resultado = pontuar(entrada, artefato)
probabilidade = float(resultado.loc[0, "fraud_probability"])
threshold = float(artefato["threshold"])
alerta = probabilidade >= threshold

with col_dir:
    st.subheader("Decisão")

    if alerta:
        st.markdown(
            "<div class='bloco-veredito alerta'>"
            "<div class='veredito-titulo'>Transação retida para revisão</div>"
            "<p class='veredito-texto'>A probabilidade ficou acima do ponto de "
            "operação escolhido no projeto.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='bloco-veredito liberado'>"
            "<div class='veredito-titulo'>Transação autorizada</div>"
            "<p class='veredito-texto'>A probabilidade ficou abaixo do ponto de "
            "operação escolhido no projeto.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        barra_probabilidade(probabilidade, threshold, alerta),
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)
    m1.metric("Probabilidade de fraude", f"{probabilidade:.2%}")
    m2.metric("Threshold de operação", f"{threshold:.2%}")

    st.caption(
        "O threshold não é 0,5. Ele foi escolhido no conjunto de validação para "
        "equilibrar recall e precisão, e vai salvo dentro do artefato — um modelo "
        "distribuído sem o seu ponto de operação não é utilizável."
    )

    st.divider()

    st.subheader("Desempenho no teste")
    st.caption(
        f"Medido no recorte cronológico final — treino nos steps "
        f"{artefato['train_steps'][0]}–{artefato['train_steps'][1]}, "
        f"teste nos steps {artefato['test_steps'][0]}–{artefato['test_steps'][1]}. "
        "Nenhuma transação de teste antecede as de treino."
    )

    d1, d2, d3 = st.columns(3)
    d1.metric("PR-AUC", f"{artefato['test_pr_auc']:.3f}")
    d2.metric("Recall", f"{artefato['test_recall']:.1%}")
    d3.metric("Precisão", f"{artefato['test_precision']:.1%}")

    st.markdown(
        f"Como referência, o motor de regras que acompanha o dataset sinaliza "
        f"{INCUMBENTE_FLAG_RATE:.6%} das transações. Mesmo que **todo** alerta dele "
        f"fosse correto, ele não passaria de **{INCUMBENTE_TETO_RECALL:.2%}** de "
        f"recall. O modelo acima alcança {artefato['test_recall']:.1%}."
    )

# --------------------------------------------------------------------------- #
# Pontuação em lote
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Pontuar um arquivo")
st.markdown(
    "Envie um CSV com as colunas `type`, `amount`, `oldbalanceOrg` e "
    "`oldbalanceDest`. As três features derivadas são calculadas aqui se o "
    "arquivo trouxer `step` e `nameDest`, exatamente como no pré-processamento "
    "do projeto."
)

arquivo = st.file_uploader("Arquivo CSV", type="csv", label_visibility="collapsed")

if arquivo is not None:
    try:
        dados = pd.read_csv(arquivo)

        if "is_merchant_dest" not in dados and "nameDest" in dados:
            dados["is_merchant_dest"] = (
                dados["nameDest"].astype(str).str.startswith("M").astype(int)
            )
        if "hour_of_day" not in dados and "step" in dados:
            dados["hour_of_day"] = dados["step"] % 24
        if "day_of_week" not in dados and "step" in dados:
            dados["day_of_week"] = (dados["step"] // 24) % 7

        pontuado = pontuar(dados, artefato)

        total = len(pontuado)
        retidas = int(pontuado["flagged"].sum())

        r1, r2, r3 = st.columns(3)
        r1.metric("Transações", f"{total:,}".replace(",", "."))
        r2.metric("Retidas", f"{retidas:,}".replace(",", "."))
        r3.metric("Taxa de retenção", f"{retidas / total:.3%}" if total else "—")

        st.dataframe(
            pontuado.sort_values("fraud_probability", ascending=False).head(200),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Baixar resultado completo",
            pontuado.to_csv(index=False).encode("utf-8"),
            file_name="transacoes_pontuadas.csv",
            mime="text/csv",
        )

    except ValueError as erro:
        st.error(str(erro))
    except Exception as erro:  # noqa: BLE001 — a mensagem vai para a tela
        st.error(f"Não consegui ler o arquivo: {erro}")

# --------------------------------------------------------------------------- #
# Rodapé
# --------------------------------------------------------------------------- #

st.divider()
st.caption(
    f"Modelo treinado em {artefato['trained_at'][:10]} · "
    f"scikit-learn {artefato['sklearn_version']} · "
    f"XGBoost {artefato['xgboost_version']} · "
    "Código em github.com/diogo-moitinho/fraud-detection-project"
)
