"""Estudo de caso interativo: detecção de fraude em transações do PaySim.

A página apresenta o projeto em abas — problema, método, resultado, impacto
financeiro, teste do modelo e limitações. Todos os números vêm do artefato
salvo pelo notebook 02_model ou das contagens registradas no README; nada é
retreinado nem estimado aqui.

Rodar localmente:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Localização do artefato
# --------------------------------------------------------------------------- #

ARTIFACT_RELATIVO = os.path.join("notebooks", "models", "fraud_realtime_v1.joblib")


def localizar_artefato() -> str:
    """Encontra o artefato sem depender de onde o app foi iniciado."""
    aqui = os.path.dirname(os.path.abspath(__file__))

    for _ in range(5):
        candidato = os.path.join(aqui, ARTIFACT_RELATIVO)
        if os.path.exists(candidato):
            return candidato
        pai = os.path.dirname(aqui)
        if pai == aqui:
            break
        aqui = pai

    return ARTIFACT_RELATIVO


ARTIFACT_PATH = localizar_artefato()

# --------------------------------------------------------------------------- #
# Constantes do projeto  (README + config.py)
# --------------------------------------------------------------------------- #

TIPOS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

POS_LIQUIDACAO = [
    "newbalanceOrig",
    "newbalanceDest",
    "errorBalanceOrig",
    "errorBalanceDest",
]

# Dataset completo
LINHAS_TOTAL = 6_362_620
FRAUDES_TOTAL = 8_213
HORAS_TOTAL = 743

# Partições cronológicas
PARTICOES = [
    #  nome,     steps,      dias,     linhas,      fraudes
    ("TREINO", "1–301",   "1–13",   4_104_531,  3_405),
    ("VALIDAÇÃO", "302–355", "13–15", 1_009_353,    558),
    ("TESTE",  "356–743",  "15–31",  1_248_736,  4_250),
]

LINHAS_TESTE = 1_248_736
FRAUDES_TESTE = 4_250
HORAS_TESTE = 743 - 356 + 1          # 388 horas
DIAS_TESTE = HORAS_TESTE / 24        # ~16,2 dias

# Motor de regras incumbente, medido na EDA
INCUMBENTE_FLAG_RATE = 0.000003
DATASET_FRAUD_RATE = 0.001291
INCUMBENTE_TETO_RECALL = INCUMBENTE_FLAG_RATE / DATASET_FRAUD_RATE

# Triagem de modelos sob CV idêntica (README)
TRIAGEM = [
    ("Regressão logística", 0.6404, 0.0434, "8,1s"),
    ("Árvore de decisão",   0.6697, 0.2649, "5,9s"),
    ("Random Forest",       0.9884, 0.0102, "79,7s"),
    ("XGBoost",             0.9693, 0.0233, "6,6s"),
]

CENARIOS = {
    "Transferência esvaziando a conta, de madrugada": {
        "tipo": "TRANSFER", "amount": 181_000.00, "oldbalanceOrg": 181_000.00,
        "oldbalanceDest": 0.00, "is_merchant_dest": 0, "hour_of_day": 3, "day_of_week": 2,
    },
    "Pagamento cotidiano a comerciante": {
        "tipo": "PAYMENT", "amount": 95.50, "oldbalanceOrg": 4_300.00,
        "oldbalanceDest": 0.00, "is_merchant_dest": 1, "hour_of_day": 14, "day_of_week": 3,
    },
    "Saque de rotina": {
        "tipo": "CASH_OUT", "amount": 1_200.00, "oldbalanceOrg": 8_500.00,
        "oldbalanceDest": 25_000.00, "is_merchant_dest": 0, "hour_of_day": 11, "day_of_week": 5,
    },
}

# Cores de estado — nunca sozinhas: sempre acompanhadas de ícone e rótulo.
COR_BOM = "#0ca30c"
COR_ATENCAO = "#ec835a"
COR_CRITICO = "#d03b3b"


# --------------------------------------------------------------------------- #
# Formatação em português
# --------------------------------------------------------------------------- #

def num(valor: float, casas: int = 0) -> str:
    """Formata número no padrão brasileiro: ponto de milhar, vírgula decimal."""
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def pct(valor: float, casas: int = 1) -> str:
    """Porcentagem com vírgula decimal."""
    return f"{valor * 100:.{casas}f}".replace(".", ",") + "%"


def moeda(valor: float) -> str:
    sinal = "−" if valor < 0 else ""
    v = abs(valor)
    if v >= 1_000_000:
        return f"{sinal}R$ {num(v / 1_000_000, 1)} mi"
    return f"{sinal}R$ {num(v)}"


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Carregando o modelo…")
def carregar_artefato(caminho: str = ARTIFACT_PATH) -> dict:
    """Lê o artefato salvo pelo notebook de modelagem.

    O threshold vem junto do pipeline de propósito: um modelo salvo sem o seu
    ponto de operação não é utilizável.
    """
    return joblib.load(caminho)


def pontuar(frame: pd.DataFrame, artefato: dict) -> pd.DataFrame:
    """Pontua transações e aplica o threshold guardado no artefato."""
    vazadas = [c for c in POS_LIQUIDACAO if c in artefato["features"]]
    if vazadas:
        raise ValueError(
            f"o artefato declara features pós-liquidação {vazadas}; "
            "ele é forense, não serve para pontuação em tempo real"
        )

    faltando = [c for c in artefato["features"] if c not in frame.columns]
    if faltando:
        raise ValueError(f"faltam as colunas obrigatórias: {faltando}")

    probabilidades = artefato["pipeline"].predict_proba(frame[artefato["features"]])[:, 1]

    saida = frame.copy()
    saida["fraud_probability"] = probabilidades
    saida["flagged"] = (probabilidades >= artefato["threshold"]).astype(int)
    return saida


def matriz_confusao(artefato: dict) -> dict:
    """Reconstrói a matriz de confusão do teste a partir de recall e precisão.

    As contagens do teste estão no README; recall e precisão, no artefato. Com
    isso os quatro quadrantes saem por aritmética exata — e de fato resultam em
    inteiros, o que confirma que as duas fontes descrevem a mesma avaliação.
    """
    vp = artefato["test_recall"] * FRAUDES_TESTE
    alertas = vp / artefato["test_precision"]
    return {
        "vp": round(vp),
        "fp": round(alertas - vp),
        "fn": round(FRAUDES_TESTE - vp),
        "vn": round(LINHAS_TESTE - alertas - (FRAUDES_TESTE - vp)),
        "alertas": round(alertas),
    }


# --------------------------------------------------------------------------- #
# Estilo
# --------------------------------------------------------------------------- #

CSS = f"""
<style>
  .painel {{
    border: 1px solid rgba(128,128,128,.25);
    border-left-width: 3px;
    padding: 16px 18px;
    margin-bottom: 14px;
  }}
  .painel.bom      {{ border-left-color: {COR_BOM}; }}
  .painel.atencao  {{ border-left-color: {COR_ATENCAO}; }}
  .painel.critico  {{ border-left-color: {COR_CRITICO}; }}
  .painel-titulo   {{ font-size: 1.1rem; font-weight: 600; margin-bottom: 3px; }}
  .painel-texto    {{ opacity: .75; font-size: .9rem; margin: 0; }}

  /* barra de probabilidade */
  .trilho {{ position: relative; height: 10px;
             background: rgba(128,128,128,.18); margin: 28px 0 8px; }}
  .trilho .preenchido {{ position: absolute; inset: 0 auto 0 0; border-radius: 0 4px 4px 0; }}
  .trilho .corte {{ position: absolute; top: -7px; bottom: -7px; width: 2px; background: #6b46d6; }}
  .trilho .corte-rotulo {{ position: absolute; bottom: calc(100% + 5px); white-space: nowrap;
                           font-size: .74rem; color: #6b46d6; font-variant-numeric: tabular-nums; }}
  .trilho-legenda {{ display: flex; justify-content: space-between;
                     font-size: .76rem; opacity: .6; font-variant-numeric: tabular-nums; }}

  /* barras horizontais rotuladas */
  .barras {{ display: grid; gap: 14px; margin: 6px 0 4px; }}
  .barra-linha {{ display: grid; grid-template-columns: 190px 1fr; gap: 14px; align-items: center; }}
  .barra-nome {{ font-size: .86rem; opacity: .85; }}
  .barra-trilho {{ position: relative; height: 22px; background: rgba(128,128,128,.12); }}
  .barra-fill {{ height: 100%; border-radius: 0 4px 4px 0; }}
  .barra-valor {{ position: absolute; top: 50%; transform: translateY(-50%);
                  font-size: .82rem; font-variant-numeric: tabular-nums; white-space: nowrap; }}

  .riscado {{ text-decoration: line-through; opacity: .45; }}

  .quadrantes {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .quadrante {{ border: 1px solid rgba(128,128,128,.25); padding: 14px 16px; }}
  .quadrante .rotulo {{ font-size: .78rem; opacity: .7; display: flex; gap: 6px; align-items: center; }}
  .quadrante .valor {{ font-size: 1.7rem; font-weight: 600; font-variant-numeric: tabular-nums;
                       line-height: 1.2; margin-top: 2px; }}
  .quadrante .nota {{ font-size: .76rem; opacity: .6; }}
</style>
"""


def barra_probabilidade(probabilidade: float, threshold: float, alerta: bool) -> str:
    cor = COR_CRITICO if alerta else COR_BOM
    ancora = "right:4px" if threshold > 0.6 else "left:4px"
    return f"""
    <div class="trilho">
      <div class="preenchido" style="width:{probabilidade * 100:.4f}%;background:{cor}"></div>
      <div class="corte" style="left:{threshold * 100:.4f}%">
        <span class="corte-rotulo" style="{ancora}">corte {pct(threshold, 2)}</span>
      </div>
    </div>
    <div class="trilho-legenda"><span>0%</span><span>100%</span></div>
    """


def barras(itens: list[tuple[str, float, str, str]]) -> str:
    """Barras horizontais com rótulo direto em cada uma.

    itens: (nome, valor, texto_do_rotulo, cor)
    """
    maximo = max((v for _, v, _, _ in itens), default=0) or 1

    # A maior barra ocupa 72% do trilho, deixando espaço fixo para o rótulo à
    # direita de todas elas. Assim o texto nunca fica sobre a cor — ele usa a
    # cor de texto do tema, e a barra colorida ao lado é que carrega a identidade.
    linhas = []
    for nome, valor, rotulo, cor in itens:
        largura = max(valor, 0) / maximo * 72
        linhas.append(
            f"<div class='barra-linha'>"
            f"<div class='barra-nome'>{nome}</div>"
            f"<div class='barra-trilho'>"
            f"<div class='barra-fill' style='width:max({largura:.3f}%, 2px);background:{cor}'></div>"
            f"<span class='barra-valor' style='left:calc({largura:.3f}% + 10px)'>{rotulo}</span>"
            f"</div></div>"
        )
    return f"<div class='barras'>{''.join(linhas)}</div>"


# --------------------------------------------------------------------------- #
# Página
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Detecção de fraude — PaySim", page_icon="◆", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

try:
    artefato = carregar_artefato()
except FileNotFoundError:
    st.error(
        f"Artefato não encontrado em `{ARTIFACT_PATH}`. "
        "Ele é versionado junto do repositório — confira se o arquivo veio no clone."
    )
    st.stop()

mc = matriz_confusao(artefato)
taxa_alerta = mc["alertas"] / LINHAS_TESTE

st.title("Detecção de fraude em tempo real")
st.markdown(
    "Um modelo que decide, **no momento da autorização**, se uma transação deve ser "
    "retida. O projeto também mede quanto da acurácia normalmente reportada no PaySim "
    "vem de vazamento de dados, e não de sinal."
)

aba_problema, aba_metodo, aba_resultado, aba_impacto, aba_teste, aba_limites = st.tabs([
    "O problema",
    "Método",
    "Resultado",
    "Impacto financeiro",
    "Testar o modelo",
    "Limitações",
])

# ═══════════════════════════════════════════════════════════════════════════ #
# 1. O PROBLEMA
# ═══════════════════════════════════════════════════════════════════════════ #

with aba_problema:
    st.subheader("O que precisa ser resolvido")
    st.markdown(
        "Uma instituição financeira precisa sinalizar transações fraudulentas **na hora "
        "da autorização**, capturando o máximo de fraude possível sem bloquear tanto "
        "cliente legítimo a ponto de inviabilizar a operação."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transações", num(LINHAS_TOTAL))
    c2.metric("Fraudes", num(FRAUDES_TOTAL))
    c3.metric("Taxa de fraude", f"{pct(FRAUDES_TOTAL / LINHAS_TOTAL, 4)}")
    c4.metric("Período", f"{HORAS_TOTAL} horas")

    st.caption(
        "Dataset PaySim, de dinheiro móvel. A fraude ocorre apenas em TRANSFER e "
        "CASH_OUT — PAYMENT, CASH_IN e DEBIT não registram nenhum caso."
    )

    st.divider()

    st.subheader("O controle que existe hoje")
    st.markdown(
        "O controle incumbente é uma regra única: sinalizar transferências acima de "
        "200.000. Ela dispara em **0,0003%** das transações, contra uma taxa real de "
        "fraude de 0,1291%."
    )

    st.markdown(
        f"<div class='painel critico'>"
        f"<div class='painel-titulo'>Mesmo que toda regra acertasse, ela capturaria no "
        f"máximo {pct(INCUMBENTE_TETO_RECALL, 2)} da fraude</div>"
        f"<p class='painel-texto'>Não é uma estimativa de desempenho: é um teto "
        f"aritmético. A regra não sinaliza transações suficientes para alcançar mais "
        f"do que isso, acertando ou errando.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Essa é a barra a ser batida** — e é contra ela que o caso de negócio deve "
        "ser montado, não contra zero. Um modelo que captura 40% da fraude parece "
        "modesto no vácuo; contra 0,23%, é outra conversa."
    )

    st.divider()

    st.subheader("Por que a acurácia alta do PaySim engana")
    st.markdown(
        "A maior parte do trabalho publicado sobre esse dataset reporta ROC-AUC acima "
        "de 0,99. Com uma base de 0,13% de fraude, responder *\"não é fraude\"* para "
        "tudo já garante 99,87% de acurácia — e o ROC-AUC continua lisonjeiro porque a "
        "taxa de falso positivo mal se move quando os negativos superam os positivos "
        "em três ordens de grandeza."
    )
    st.markdown(
        "Este projeto reproduz aquele número, mostra que ele é em boa parte artefato de "
        "três erros de modelagem, e reporta o que sobra depois de removê-los. A métrica "
        "usada é **PR-AUC**, que é função da taxa base e não se deixa inflar por um mar "
        "de negativos fáceis."
    )

# ═══════════════════════════════════════════════════════════════════════════ #
# 2. MÉTODO
# ═══════════════════════════════════════════════════════════════════════════ #

with aba_metodo:
    st.subheader("Os três vazamentos removidos")

    v1, v2, v3 = st.columns(3)

    with v1:
        st.markdown("**1. Reamostragem antes da validação cruzada**")
        st.markdown(
            "O SMOTE interpola entre um registro real e seus vizinhos. Aplicado antes "
            "de sortear as dobras, o ponto sintético cai na validação enquanto o "
            "registro que o originou fica no treino — o modelo é avaliado em uma "
            "combinação linear de linhas que memorizou."
        )
        st.caption("Correção: reamostragem como etapa de um pipeline imblearn, refeita a cada dobra.")

    with v2:
        st.markdown("**2. Divisão aleatória em dado sequencial**")
        st.markdown(
            "O campo `step` percorre 743 horas consecutivas. Uma divisão aleatória põe "
            "o futuro no treino e separa os pares de fraude do PaySim — um TRANSFER "
            "seguido de um CASH_OUT do mesmo valor — entre as partições, deixando o "
            "modelo casar um valor que já viu."
        )
        st.caption("Correção: divisão cronológica. `stratify` não resolve — equilibra o rótulo, não a entidade.")

    with v3:
        st.markdown("**3. Campos pós-liquidação num problema de tempo real**")
        st.markdown(
            "`newbalanceOrig`, `newbalanceDest` e os erros de saldo derivados deles não "
            "existem quando a autorização é decidida. No PaySim o fraudador esvazia a "
            "conta, então `errorBalanceOrig` resolve para exatamente 0,00 numa fatia "
            "grande das fraudes."
        )
        st.caption("Não é preditor: é o rótulo disfarçado.")

    st.divider()

    st.subheader("Partição cronológica")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Partição": nome, "Steps": steps, "Dias": dias,
                    "Linhas": num(linhas), "Fraudes": num(fraudes),
                    "Prevalência": f"{pct(fraudes / linhas, 4)}",
                }
                for nome, steps, dias, linhas, fraudes in PARTICOES
            ]
        ),
        hide_index=True, use_container_width=True,
    )

    st.markdown(
        f"O período de teste é **{(4250 / 1_248_736) / (3405 / 4_104_531):.1f}× mais "
        "hostil** que o de treino, e concentra 51,7% de toda a fraude em 19,6% das "
        "linhas. Isso é propriedade do dado, não defeito de amostragem: o modelo é "
        "ajustado na parte calma do mês e avaliado na parte movimentada."
    )
    st.caption(
        "O corte cai perto do dia 13, não do 25, porque o volume é concentrado no "
        "início — metade das transações acontece nos dez primeiros dias."
    )

    st.divider()

    st.subheader("Escolha do modelo")
    st.markdown("Quatro candidatos sob validação cruzada idêntica:")

    st.dataframe(
        pd.DataFrame(
            [
                {"Modelo": m, "PR-AUC (CV)": f"{p:.4f}".replace(".", ","),
                 "Desvio padrão": f"{s:.4f}".replace(".", ","), "Tempo de ajuste": t}
                for m, p, s, t in TRIAGEM
            ]
        ),
        hide_index=True, use_container_width=True,
    )

    st.markdown(
        "A vantagem do Random Forest cabe dentro do desvio padrão do próprio XGBoost — "
        "estatisticamente, empate. E o XGBoost ajusta **12× mais rápido**, o que pesa "
        "quando o número é multiplicado por tentativas vezes dobras."
    )
    st.markdown(
        "O resultado mais interessante da tabela é o desvio de 0,2649 da árvore de "
        "decisão: uma árvore só memoriza a rajada de fraude que calhar de cair na sua "
        "janela de treino. **Uma divisão aleatória teria escondido isso por completo.**"
    )

    st.divider()

    st.subheader("Ponto de operação")
    st.markdown(
        f"O threshold foi escolhido na partição de validação — nunca no teste — e vale "
        f"**{pct(artefato['threshold'], 2)}**, não 0,5. Ele é serializado junto do modelo: "
        "um modelo distribuído sem o seu ponto de operação não é utilizável, porque "
        "quem o carregar vai chamar `predict` em 0,5 e obter um recall sem relação com "
        "o reportado."
    )

# ═══════════════════════════════════════════════════════════════════════════ #
# 3. RESULTADO
# ═══════════════════════════════════════════════════════════════════════════ #

with aba_resultado:
    st.subheader("Desempenho no mês de teste")
    st.caption(
        f"Steps 356–743 · {num(LINHAS_TESTE)} transações · {num(FRAUDES_TESTE)} fraudes "
        f"· prevalência {pct(FRAUDES_TESTE / LINHAS_TESTE, 4)}. Nenhuma transação de teste "
        "antecede as de treino."
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("PR-AUC", f"{artefato['test_pr_auc']:.3f}".replace(".", ","))
    r2.metric("Recall", f"{pct(artefato['test_recall'], 1)}")
    r3.metric("Precisão", f"{pct(artefato['test_precision'], 1)}")
    r4.metric("Taxa de alerta", f"{pct(taxa_alerta, 3)}")

    st.divider()

    st.subheader("O que aconteceu com as 4.250 fraudes e com o resto")

    st.markdown(
        f"""
        <div class="quadrantes">
          <div class="quadrante" style="border-left:3px solid {COR_BOM}">
            <div class="rotulo">✔ Fraude capturada</div>
            <div class="valor" style="color:{COR_BOM}">{num(mc['vp'])}</div>
            <div class="nota">{pct(mc['vp'] / FRAUDES_TESTE, 1)} de toda a fraude do período</div>
          </div>
          <div class="quadrante" style="border-left:3px solid {COR_CRITICO}">
            <div class="rotulo">✘ Fraude perdida</div>
            <div class="valor" style="color:{COR_CRITICO}">{num(mc['fn'])}</div>
            <div class="nota">passou pela autorização sem ser retida</div>
          </div>
          <div class="quadrante" style="border-left:3px solid {COR_ATENCAO}">
            <div class="rotulo">⚠ Alerta falso</div>
            <div class="valor" style="color:{COR_ATENCAO}">{num(mc['fp'])}</div>
            <div class="nota">cliente legítimo retido — 1 a cada {num(LINHAS_TESTE / mc['fp'])} transações</div>
          </div>
          <div class="quadrante">
            <div class="rotulo">Liberado corretamente</div>
            <div class="valor">{num(mc['vn'])}</div>
            <div class="nota">nenhum atrito para o cliente</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Estes quatro números são derivados do recall e da precisão guardados no "
        "artefato, combinados com as contagens da partição de teste. A conta fecha em "
        "inteiros exatos, o que confirma que as duas fontes descrevem a mesma avaliação."
    )

    st.divider()

    st.subheader("Contra o controle que existe hoje")

    fraudes_incumbente = FRAUDES_TESTE * INCUMBENTE_TETO_RECALL

    st.markdown(
        barras([
            ("Modelo — capturadas", mc["vp"],
             f"{num(mc['vp'])} fraudes · {pct(artefato['test_recall'], 1)}", COR_BOM),
            ("Regra atual — teto", fraudes_incumbente,
             f"{num(fraudes_incumbente, 1)} fraudes · {pct(INCUMBENTE_TETO_RECALL, 2)}", COR_CRITICO),
        ]),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"No mesmo mês, sobre as mesmas {num(LINHAS_TESTE)} transações: o modelo captura "
        f"**{num(mc['vp'])}** fraudes; a regra atual, no melhor cenário aritmeticamente "
        f"possível, capturaria **{num(fraudes_incumbente, 1)}**. "
        f"Um ganho de **{artefato['test_recall'] / INCUMBENTE_TETO_RECALL:.0f}×** em recall."
    )

    st.info(
        "O modelo forense — aquele que mantém os campos pós-liquidação — não está "
        "versionado neste repositório, apenas o de tempo real. Por isso o custo do "
        "vazamento em PR-AUC, que é a diferença entre os dois, não aparece aqui. "
        "Para publicá-lo, salve também o artefato forense no notebook 02_model.",
        icon="ℹ",
    )

# ═══════════════════════════════════════════════════════════════════════════ #
# 4. IMPACTO FINANCEIRO
# ═══════════════════════════════════════════════════════════════════════════ #

with aba_impacto:
    st.subheader("Quanto o modelo economiza")

    st.warning(
        "Os dois valores abaixo são **premissas suas, não medições do dado**. O PaySim "
        "não traz custo de operação, e o projeto assume custo simétrico entre bloquear "
        "um cliente legítimo e perder uma fraude. Ajuste para a realidade da operação "
        "antes de citar qualquer número daqui.",
        icon="⚠",
    )

    p1, p2 = st.columns(2)
    with p1:
        perda_fraude = st.number_input(
            "Perda média por fraude não detectada (R$)",
            min_value=0.0, value=3_000.0, step=100.0, format="%.2f",
        )
    with p2:
        custo_revisao = st.number_input(
            "Custo de revisar um alerta (R$)",
            min_value=0.0, value=25.0, step=5.0, format="%.2f",
            help="Tempo do analista, atrito com o cliente legítimo, reprocessamento.",
        )

    perda_evitada = mc["vp"] * perda_fraude
    custo_operacao = mc["alertas"] * custo_revisao
    perda_residual = mc["fn"] * perda_fraude
    liquido = perda_evitada - custo_operacao

    sem_modelo = FRAUDES_TESTE * perda_fraude

    fraudes_incumbente = FRAUDES_TESTE * INCUMBENTE_TETO_RECALL
    incumbente_evitada = fraudes_incumbente * perda_fraude
    incumbente_custo = LINHAS_TESTE * INCUMBENTE_FLAG_RATE * custo_revisao

    # Prejuízo total do período em cada cenário: o que a fraude levou, mais o
    # que custou revisar os alertas. Negativo, porque é saída de caixa.
    prejuizo_sem_nada = -sem_modelo
    prejuizo_incumbente = -((sem_modelo - incumbente_evitada) + incumbente_custo)
    prejuizo_modelo = -(perda_residual + custo_operacao)

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Perda evitada no mês", moeda(perda_evitada))
    m2.metric("Custo de revisar os alertas", moeda(custo_operacao))
    m3.metric("Resultado líquido", moeda(liquido))

    st.markdown(
        barras([
            ("Perda evitada", perda_evitada, moeda(perda_evitada), COR_BOM),
            ("Perda residual", perda_residual, moeda(perda_residual), COR_CRITICO),
            ("Custo de revisão", custo_operacao, moeda(custo_operacao), COR_ATENCAO),
        ]),
        unsafe_allow_html=True,
    )

    st.caption(
        f"Sobre o período de teste: {num(DIAS_TESTE, 1)} dias, "
        f"{num(LINHAS_TESTE)} transações."
    )

    st.divider()

    st.subheader("Comparação")

    st.dataframe(
        pd.DataFrame([
            {
                "Cenário": "Sem nenhum controle",
                "Fraude capturada": "0",
                "Perda evitada": moeda(0),
                "Custo de revisão": moeda(0),
                "Prejuízo do período": moeda(prejuizo_sem_nada),
            },
            {
                "Cenário": "Regra atual (teto teórico)",
                "Fraude capturada": num(fraudes_incumbente, 1),
                "Perda evitada": moeda(incumbente_evitada),
                "Custo de revisão": moeda(incumbente_custo),
                "Prejuízo do período": moeda(prejuizo_incumbente),
            },
            {
                "Cenário": "Modelo de tempo real",
                "Fraude capturada": num(mc["vp"]),
                "Perda evitada": moeda(perda_evitada),
                "Custo de revisão": moeda(custo_operacao),
                "Prejuízo do período": moeda(prejuizo_modelo),
            },
        ]),
        hide_index=True, use_container_width=True,
    )

    st.caption(
        "O prejuízo do período soma o que a fraude levou e o que custou revisar os "
        "alertas. Quanto mais próximo de zero, melhor."
    )

    ganho = prejuizo_modelo - prejuizo_incumbente
    st.markdown(
        f"Trocar a regra atual pelo modelo muda o resultado do mês em **{moeda(ganho)}**. "
        f"Projetado para um ano, mantidas as mesmas premissas e o mesmo volume: "
        f"**{moeda(ganho * 365 / DIAS_TESTE)}**."
    )
    st.caption(
        "A projeção anual multiplica um mês simulado por 22,6. Ela supõe volume e "
        "perfil de fraude constantes, o que não se sustenta na prática — trate como "
        "ordem de grandeza, não como previsão."
    )

    st.divider()

    st.subheader("A partir de que ponto o modelo se paga")

    razao = mc["alertas"] / mc["vp"]
    st.markdown(
        f"O modelo gera {num(mc['alertas'])} alertas para capturar {num(mc['vp'])} "
        f"fraudes. Ele se paga enquanto **uma fraude custar mais do que "
        f"{num(razao, 2)}× uma revisão** — ou seja, praticamente sempre."
    )
    st.markdown(
        f"Com as premissas atuais, a razão é de {num(perda_fraude / custo_revisao, 1)}× "
        f"— {'bem acima' if perda_fraude / custo_revisao > razao else 'abaixo'} do ponto "
        f"de equilíbrio."
    )
    st.caption(
        "Essa razão não depende das premissas de custo: ela sai direto da precisão do "
        "modelo. É o número mais robusto desta aba."
    )

# ═══════════════════════════════════════════════════════════════════════════ #
# 5. TESTAR O MODELO
# ═══════════════════════════════════════════════════════════════════════════ #

with aba_teste:
    st.subheader("Pontuar uma transação")

    col_esq, col_dir = st.columns([1, 1.15], gap="large")

    with col_esq:
        cenario = st.selectbox("Partir de um cenário", list(CENARIOS.keys()))
        base = CENARIOS[cenario]

        tipo = st.selectbox("Tipo", TIPOS, index=TIPOS.index(base["tipo"]))
        amount = st.number_input("Valor", min_value=0.0, value=base["amount"],
                                 step=100.0, format="%.2f")
        oldbalance_org = st.number_input("Saldo da origem antes da transação",
                                         min_value=0.0, value=base["oldbalanceOrg"],
                                         step=100.0, format="%.2f")
        oldbalance_dest = st.number_input("Saldo do destino antes da transação",
                                          min_value=0.0, value=base["oldbalanceDest"],
                                          step=100.0, format="%.2f")
        is_merchant = st.checkbox(
            "Destinatário é comerciante", value=bool(base["is_merchant_dest"]),
            help="No PaySim, contas de comerciante têm o prefixo M em nameDest.",
        )

        h1, h2 = st.columns(2)
        with h1:
            hora = st.slider("Hora do dia", 0, 23, base["hour_of_day"])
        with h2:
            dia = st.slider("Dia da semana", 0, 6, base["day_of_week"],
                            help="Derivado de (step // 24) % 7 — índice cíclico, "
                                 "não um dia de calendário real.")

        with st.expander("Por que não há campos de saldo final aqui"):
            st.markdown(
                "Estes quatro campos existem no dataset e elevam muito a métrica, mas "
                "**não existem no momento da autorização**."
            )
            st.markdown(
                "".join(f"<div class='riscado'>{c}</div>" for c in POS_LIQUIDACAO),
                unsafe_allow_html=True,
            )
            st.caption(
                "Um modelo com esses campos vale para revisão forense, nunca para "
                "prevenção. A função de pontuação recusa artefatos que os contenham."
            )

    entrada = pd.DataFrame([{
        "type": tipo, "amount": amount,
        "oldbalanceOrg": oldbalance_org, "oldbalanceDest": oldbalance_dest,
        "is_merchant_dest": int(is_merchant),
        "hour_of_day": hora, "day_of_week": dia,
    }])

    resultado = pontuar(entrada, artefato)
    probabilidade = float(resultado.loc[0, "fraud_probability"])
    threshold = float(artefato["threshold"])
    alerta = probabilidade >= threshold

    with col_dir:
        classe = "critico" if alerta else "bom"
        titulo = "Transação retida para revisão" if alerta else "Transação autorizada"
        icone = "✘" if alerta else "✔"
        lado = "acima" if alerta else "abaixo"

        st.markdown(
            f"<div class='painel {classe}'>"
            f"<div class='painel-titulo'>{icone} {titulo}</div>"
            f"<p class='painel-texto'>A probabilidade ficou {lado} do ponto de "
            f"operação definido no projeto.</p></div>",
            unsafe_allow_html=True,
        )

        st.markdown(barra_probabilidade(probabilidade, threshold, alerta),
                    unsafe_allow_html=True)

        t1, t2 = st.columns(2)
        t1.metric("Probabilidade de fraude", f"{pct(probabilidade, 2)}")
        t2.metric("Threshold de operação", f"{pct(threshold, 2)}")

    st.divider()

    st.subheader("Pontuar um arquivo")
    st.markdown(
        "Envie um CSV com `type`, `amount`, `oldbalanceOrg` e `oldbalanceDest`. As três "
        "features derivadas são calculadas aqui se o arquivo trouxer `step` e "
        "`nameDest`, do mesmo jeito que no pré-processamento do projeto."
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

            a1, a2, a3 = st.columns(3)
            a1.metric("Transações", num(total))
            a2.metric("Retidas", num(retidas))
            a3.metric("Taxa de retenção", f"{pct(retidas / total, 3)}" if total else "—")

            st.dataframe(
                pontuado.sort_values("fraud_probability", ascending=False).head(200),
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "Baixar resultado completo",
                pontuado.to_csv(index=False).encode("utf-8"),
                file_name="transacoes_pontuadas.csv", mime="text/csv",
            )
        except ValueError as erro:
            st.error(str(erro))
        except Exception as erro:  # noqa: BLE001
            st.error(f"Não consegui ler o arquivo: {erro}")

# ═══════════════════════════════════════════════════════════════════════════ #
# 6. LIMITAÇÕES
# ═══════════════════════════════════════════════════════════════════════════ #

with aba_limites:
    st.subheader("O que este número não prova")

    st.markdown(
        "**Dado sintético.** Os agentes de fraude do PaySim seguem regras programadas, "
        "aprendíveis de um jeito que adversários reais não são. Toda métrica aqui é um "
        "limite superior do desempenho no mundo real."
    )
    st.markdown(
        "**Sem deriva adversarial.** A validação cobre um mês de uma simulação estática. "
        "Padrões reais de fraude mudam em resposta à detecção, então o desempenho em "
        "produção decai sem retreino."
    )
    st.markdown(
        "**Escopo de transação única.** Cada transação é pontuada isoladamente. O modelo "
        "não enxerga que uma conta recebeu três transferências nos dez minutos "
        "anteriores, que é como quadrilhas de fato aparecem. Agregados de velocidade "
        "por conta são a extensão de maior valor disponível — e a razão de a busca de "
        "hiperparâmetros estabilizar cedo: a restrição é o conjunto de features, não o "
        "modelo."
    )
    st.markdown(
        "**Custo simétrico.** O threshold padrão maximiza F1, o que precifica um cliente "
        "legítimo bloqueado e uma fraude perdida como se fossem a mesma coisa. Não são. "
        "A aba de impacto financeiro existe justamente para tornar essa assimetria "
        "explícita."
    )

    st.divider()

    st.subheader("Próximos passos")
    st.markdown(
        "1. Agregados de velocidade por conta sobre uma janela móvel de `step`, "
        "calculados estritamente a partir de linhas passadas, preservando a ordem causal.\n"
        "2. Rederivar o ponto de operação contra custos explícitos de falso positivo e "
        "falso negativo.\n"
        "3. Linha de base de monitoramento sobre a distribuição de predições e a taxa de "
        "alerta, para que a deriva seja detectada antes de o recall degradar em silêncio.\n"
        "4. Validação contra dado transacional real antes de qualquer consideração "
        "sobre produção."
    )

# --------------------------------------------------------------------------- #
# Rodapé
# --------------------------------------------------------------------------- #

st.divider()
st.caption(
    f"Modelo treinado em {artefato['trained_at'][:10]} · "
    f"scikit-learn {artefato['sklearn_version']} · XGBoost {artefato['xgboost_version']} · "
    "Código em github.com/diogo-moitinho/fraud-detection-project"
)
