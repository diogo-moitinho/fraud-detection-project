"""Estudo de caso: detecção de fraude em transações do PaySim.

Seis abas — problema, método, resultado, impacto financeiro, teste e limitações.
Os números vêm do artefato salvo pelo notebook 02_model e das contagens da
partição de teste registradas no README. Nada é retreinado aqui.

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #

TIPOS = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

POS_LIQUIDACAO = ["newbalanceOrig", "newbalanceDest",
                  "errorBalanceOrig", "errorBalanceDest"]

LINHAS_TOTAL, FRAUDES_TOTAL, HORAS_TOTAL = 6_362_620, 8_213, 743
LINHAS_TESTE, FRAUDES_TESTE = 1_248_736, 4_250
DIAS_TESTE = (743 - 356 + 1) / 24

PARTICOES = pd.DataFrame([
    {"Partição": "Treino",    "Steps": "1–301",   "Linhas": 4_104_531, "Fraudes": 3_405},
    {"Partição": "Validação", "Steps": "302–355", "Linhas": 1_009_353, "Fraudes":   558},
    {"Partição": "Teste",     "Steps": "356–743", "Linhas": 1_248_736, "Fraudes": 4_250},
])
PARTICOES["Prevalência"] = PARTICOES["Fraudes"] / PARTICOES["Linhas"] * 100

TRIAGEM = pd.DataFrame([
    {"Modelo": "Regressão logística", "PR-AUC": 0.6404, "Desvio": 0.0434, "Ajuste": "8,1s"},
    {"Modelo": "Árvore de decisão",   "PR-AUC": 0.6697, "Desvio": 0.2649, "Ajuste": "5,9s"},
    {"Modelo": "Random Forest",       "PR-AUC": 0.9884, "Desvio": 0.0102, "Ajuste": "79,7s"},
    {"Modelo": "XGBoost",             "PR-AUC": 0.9693, "Desvio": 0.0233, "Ajuste": "6,6s"},
])

INCUMBENTE_FLAG_RATE = 0.000003
DATASET_FRAUD_RATE = 0.001291
INCUMBENTE_TETO = INCUMBENTE_FLAG_RATE / DATASET_FRAUD_RATE

CENARIOS = {
    "Transferência esvaziando a conta, de madrugada":
        ("TRANSFER", 181_000.0, 181_000.0, 0.0, 0, 3, 2),
    "Pagamento cotidiano a comerciante":
        ("PAYMENT", 95.5, 4_300.0, 0.0, 1, 14, 3),
    "Saque de rotina":
        ("CASH_OUT", 1_200.0, 8_500.0, 25_000.0, 0, 11, 5),
}

VERDE, VERMELHO, LARANJA, AZUL = "#0ca30c", "#d03b3b", "#ec835a", "#2a78d6"

# --------------------------------------------------------------------------- #
# Formatação em português
# --------------------------------------------------------------------------- #

def num(v: float, casas: int = 0) -> str:
    return f"{v:,.{casas}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(v: float, casas: int = 1) -> str:
    return f"{v * 100:.{casas}f}".replace(".", ",") + "%"


def moeda(v: float) -> str:
    sinal, v = ("−" if v < 0 else ""), abs(v)
    return f"{sinal}R$ {num(v / 1e6, 1)} mi" if v >= 1e6 else f"{sinal}R$ {num(v)}"


def moeda_md(v: float) -> str:
    """Igual a moeda(), mas com o cifrão escapado.

    Em markdown o Streamlit lê $...$ como fórmula LaTeX, e dois valores em R$ na
    mesma frase viram uma equação.
    """
    return moeda(v).replace("$", r"\$")


def mostrar(fig, altura: int = 300) -> None:
    """Aplica o enxoval comum a todos os gráficos e desenha."""
    # separators=",." põe vírgula no decimal e ponto no milhar em todo o gráfico
    fig.update_layout(height=altura, margin=dict(l=0, r=10, t=10, b=0),
                      separators=",.")
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    st.plotly_chart(fig, width="stretch")


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #

def caminho_artefato() -> str:
    """Acha o .joblib subindo a partir deste arquivo até a raiz do repositório."""
    relativo = os.path.join("notebooks", "models", "fraud_realtime_v1.joblib")
    pasta = os.path.dirname(os.path.abspath(__file__))

    for _ in range(5):
        if os.path.exists(os.path.join(pasta, relativo)):
            return os.path.join(pasta, relativo)
        pai = os.path.dirname(pasta)
        if pai == pasta:
            break
        pasta = pai
    return relativo


@st.cache_resource(show_spinner="Carregando o modelo…")
def carregar():
    return joblib.load(caminho_artefato())


def pontuar(frame: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Pontua transações e aplica o threshold guardado no artefato."""
    faltando = [c for c in art["features"] if c not in frame.columns]
    if faltando:
        raise ValueError(f"faltam as colunas obrigatórias: {faltando}")

    prob = art["pipeline"].predict_proba(frame[art["features"]])[:, 1]
    return frame.assign(
        fraud_probability=prob,
        flagged=(prob >= art["threshold"]).astype(int)
    )


# --------------------------------------------------------------------------- #
# Página
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="Detecção de fraude — PaySim",
                   page_icon="◆", layout="wide")

try:
    art = carregar()
except FileNotFoundError:
    st.error("Artefato `fraud_realtime_v1.joblib` não encontrado no repositório.")
    st.stop()

# Matriz de confusão do teste, derivada de recall e precisão. Fecha em inteiros.
VP = round(art["test_recall"] * FRAUDES_TESTE)
ALERTAS = round(VP / art["test_precision"])
FP = ALERTAS - VP
FN = FRAUDES_TESTE - VP
VN = LINHAS_TESTE - ALERTAS - FN

st.title("Detecção de fraude em tempo real")
st.caption(
    "Um modelo que decide, no momento da autorização, se uma transação deve ser "
    "retida — e a medida de quanto da acurácia usualmente reportada no PaySim vem "
    "de vazamento, e não de sinal."
)

abas = st.tabs(["O problema", "Método", "Resultado",
                "Impacto financeiro", "Testar o modelo", "Limitações"])

# ═══ 1. O PROBLEMA ═════════════════════════════════════════════════════════ #

with abas[0]:
    c = st.columns(4)
    c[0].metric("Transações", num(LINHAS_TOTAL))
    c[1].metric("Fraudes", num(FRAUDES_TOTAL))
    c[2].metric("Taxa de fraude", pct(FRAUDES_TOTAL / LINHAS_TOTAL, 4))
    c[3].metric("Período", f"{HORAS_TOTAL} horas")

    st.markdown(
        "Uma instituição financeira precisa sinalizar transações fraudulentas **na "
        "hora da autorização**, capturando o máximo de fraude sem bloquear tanto "
        "cliente legítimo a ponto de inviabilizar a operação. No PaySim a fraude "
        "ocorre apenas em `TRANSFER` e `CASH_OUT`."
    )

    st.divider()

    esq, dir_ = st.columns(2)

    with esq:
        st.subheader("Por que a acurácia engana")
        acuracia = pd.DataFrame({
            "Medida": ["Acurácia", "Fraude capturada"],
            "Valor": [99.87, 0.0],
        })
        fig = px.bar(acuracia, x="Valor", y="Medida", orientation="h",
                     text=["99,87%", "0%"], range_x=[0, 108],
                     color="Medida", color_discrete_map={
                         "Acurácia": AZUL, "Fraude capturada": VERMELHO})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, xaxis_title="%", yaxis_title=None)
        mostrar(fig, 220)
        st.caption(
            "Um modelo que responde *“não é fraude”* para tudo acerta 99,87% das "
            "vezes e não captura fraude nenhuma. Por isso a métrica do projeto é "
            "PR-AUC, que é função da taxa base."
        )

    with dir_:
        st.subheader("O controle que existe hoje")
        teto = pd.DataFrame({
            "Controle": ["Regra atual (teto)", "Todo o resto"],
            "Fraude": [INCUMBENTE_TETO * 100, 100 - INCUMBENTE_TETO * 100],
        })
        fig = px.pie(teto, names="Controle", values="Fraude", hole=0.62,
                     color="Controle", color_discrete_map={
                         "Regra atual (teto)": VERMELHO, "Todo o resto": "#c9ccd1"})
        fig.update_traces(textinfo="none", sort=False)
        fig.update_layout(legend=dict(orientation="h", y=-0.08, x=0.5,
                                      xanchor="center", title=None))
        fig.add_annotation(text=f"<b>{pct(INCUMBENTE_TETO, 2)}</b><br>da fraude",
                           showarrow=False, font=dict(size=20))
        mostrar(fig, 220)
        st.caption(
            "A regra única — transferências acima de 200.000 — dispara em 0,0003% "
            "das transações. **Mesmo que todo alerta dela estivesse certo**, não "
            "alcançaria mais que isso. É um teto aritmético, não uma estimativa."
        )

    st.info(
        "Essa é a barra a ser batida, e é contra ela que o caso de negócio deve ser "
        "montado — não contra zero."
    )

# ═══ 2. MÉTODO ═════════════════════════════════════════════════════════════ #

with abas[1]:
    st.subheader("Os três vazamentos removidos")

    for titulo, texto, correcao in [
        ("1. Reamostragem antes da validação cruzada",
         "O SMOTE interpola entre um registro real e seus vizinhos. Aplicado antes "
         "de sortear as dobras, o ponto sintético cai na validação enquanto o "
         "registro que o originou fica no treino.",
         "Reamostragem como etapa de um pipeline imblearn, refeita a cada dobra."),
        ("2. Divisão aleatória em dado sequencial",
         "`step` percorre 743 horas consecutivas. Uma divisão aleatória põe o futuro "
         "no treino e separa os pares de fraude do PaySim entre as partições.",
         "Divisão cronológica. `stratify` equilibra o rótulo, não a entidade."),
        ("3. Campos pós-liquidação num problema de tempo real",
         "Os saldos finais não existem quando a autorização é decidida. No PaySim o "
         "fraudador esvazia a conta, então `errorBalanceOrig` resolve para 0,00 numa "
         "fatia grande das fraudes.",
         "Não é preditor: é o rótulo disfarçado."),
    ]:
        with st.container(border=True):
            st.markdown(f"**{titulo}**")
            st.markdown(texto)
            st.caption(correcao)

    st.divider()
    st.subheader("Partição cronológica")

    esq, dir_ = st.columns(2)

    with esq:
        fig = px.bar(PARTICOES, x="Partição", y="Linhas",
                     text=[num(v) for v in PARTICOES["Linhas"]])
        fig.update_traces(marker_color=AZUL, textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_title="transações", xaxis_title=None)
        mostrar(fig, 290)
        st.caption("Metade das transações acontece nos dez primeiros dias.")

    with dir_:
        fig = px.bar(PARTICOES, x="Partição", y="Prevalência",
                     text=[pct(v / 100, 4) for v in PARTICOES["Prevalência"]])
        fig.update_traces(marker_color=VERMELHO, textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_title="fraude (%)", xaxis_title=None)
        mostrar(fig, 290)
        st.caption("O teste é 4,1× mais hostil que o treino — propriedade do dado.")

    st.markdown(
        "O modelo é ajustado na parte calma do mês e avaliado na movimentada. O teste "
        "concentra 51,7% de toda a fraude em 19,6% das linhas, e nenhuma transação de "
        "teste antecede as de treino."
    )

    st.divider()
    st.subheader("Escolha do modelo")

    fig = px.bar(TRIAGEM.sort_values("PR-AUC"), x="PR-AUC", y="Modelo",
                 orientation="h", error_x="Desvio", range_x=[0, 1.3])
    fig.update_traces(marker_color=AZUL)
    fig.update_layout(yaxis_title=None)
    mostrar(fig, 280)

    st.caption("Barras de erro = desvio padrão entre as dobras da validação cruzada.")
    st.markdown(
        "A vantagem do Random Forest **cabe dentro do desvio do próprio XGBoost** — "
        "estatisticamente, empate — e o XGBoost ajusta 12× mais rápido. O resultado "
        "mais revelador é a barra da árvore de decisão: uma árvore só memoriza a "
        "rajada de fraude que calhar de cair na sua janela. Uma divisão aleatória "
        "teria escondido isso."
    )

    st.metric("Ponto de operação escolhido na validação", pct(art["threshold"], 2))
    st.caption(
        "Não é 0,5, e vai serializado junto do modelo: quem carregar um modelo sem o "
        "ponto de operação vai chamar `predict` em 0,5 e obter um recall sem relação "
        "com o reportado."
    )

# ═══ 3. RESULTADO ══════════════════════════════════════════════════════════ #

with abas[2]:
    c = st.columns(4)
    c[0].metric("PR-AUC", num(art["test_pr_auc"], 3))
    c[1].metric("Recall", pct(art["test_recall"], 1))
    c[2].metric("Precisão", pct(art["test_precision"], 1))
    c[3].metric("Taxa de alerta", pct(ALERTAS / LINHAS_TESTE, 3))

    st.caption(
        f"Período de teste: steps 356–743 ({num(DIAS_TESTE, 1)} dias) · {num(LINHAS_TESTE)} transações · "
        f"{num(FRAUDES_TESTE)} fraudes · prevalência {pct(FRAUDES_TESTE / LINHAS_TESTE, 4)}."
    )

    st.divider()
    st.subheader("Onde foram parar as fraudes e os alertas")

    fluxo = pd.DataFrame([
        {"Grupo": f"{num(FRAUDES_TESTE)} fraudes", "Desfecho": "Fraude capturada", "n": VP},
        {"Grupo": f"{num(FRAUDES_TESTE)} fraudes", "Desfecho": "Fraude perdida", "n": FN},
        {"Grupo": f"{num(ALERTAS)} alertas", "Desfecho": "Fraude capturada", "n": VP},
        {"Grupo": f"{num(ALERTAS)} alertas", "Desfecho": "Alerta falso", "n": FP},
    ])
    fluxo["rotulo"] = [num(v) for v in fluxo["n"]]
    fig = px.bar(fluxo, x="n", y="Grupo", color="Desfecho", orientation="h",
                 text="rotulo", color_discrete_map={
                     "Fraude capturada": VERDE,
                     "Fraude perdida": VERMELHO,
                     "Alerta falso": LARANJA})
    fig.update_traces(textposition="inside")
    fig.update_layout(barmode="stack", xaxis_title="transações", yaxis_title=None,
                      legend_title=None, bargap=0.45,
                      legend=dict(orientation="h", y=1.12, x=0))
    mostrar(fig, 250)

    c = st.columns(4)
    c[0].metric("✔ Fraude capturada", num(VP), pct(VP / FRAUDES_TESTE, 1))
    c[1].metric("✘ Fraude perdida", num(FN), f"-{pct(FN / FRAUDES_TESTE, 1)}")
    c[2].metric("⚠ Alerta falso", num(FP))
    c[3].metric("Liberado corretamente", num(VN))

    st.caption(
        f"Um alerta falso a cada {num(LINHAS_TESTE / FP)} transações. Os quatro "
        "números são derivados do recall e da precisão do artefato com as contagens "
        "do teste — a conta fecha em inteiros exatos."
    )

    st.divider()
    st.subheader("Contra o controle que existe hoje")

    comparacao = pd.DataFrame({
        "Controle": ["Modelo de tempo real", "Regra atual (teto)"],
        "Recall": [art["test_recall"] * 100, INCUMBENTE_TETO * 100],
    })
    fig = px.bar(comparacao, x="Recall", y="Controle", orientation="h",
                 text=[pct(art["test_recall"], 1), pct(INCUMBENTE_TETO, 2)],
                 range_x=[0, 92], color="Controle",
                 color_discrete_map={"Modelo de tempo real": VERDE,
                                     "Regra atual (teto)": VERMELHO})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, xaxis_title="fraude capturada (%)",
                      yaxis_title=None)
    mostrar(fig, 220)

    st.markdown(
        f"Sobre as mesmas {num(LINHAS_TESTE)} transações, o modelo captura "
        f"**{num(VP)}** fraudes; a regra atual, no melhor cenário aritmético, "
        f"capturaria **{num(FRAUDES_TESTE * INCUMBENTE_TETO, 1)}**. Ganho de "
        f"**{art['test_recall'] / INCUMBENTE_TETO:.0f}×** em recall."
    )

    st.info(
        "O modelo forense não está versionado no repositório, só o de tempo real. "
        "Por isso o custo do vazamento em PR-AUC — a diferença entre os dois — não "
        "aparece aqui. Salvar o artefato forense no `02_model` destravaria isso."
    )

# ═══ 4. IMPACTO FINANCEIRO ═════════════════════════════════════════════════ #

with abas[3]:
    st.warning(
        "Os dois valores abaixo são **premissas suas, não medições do dado**. O "
        "PaySim não traz custo de operação. Ajuste para a realidade da operação "
        "antes de citar qualquer número daqui."
    )

    esq, dir_ = st.columns(2)
    perda_fraude = esq.number_input("Perda média por fraude não detectada (R$)",
                                    min_value=0.0, value=3_000.0, step=100.0)
    custo_revisao = dir_.number_input("Custo de revisar um alerta (R$)",
                                      min_value=0.0, value=25.0, step=5.0)

    evitada = VP * perda_fraude
    operacao = ALERTAS * custo_revisao
    residual = FN * perda_fraude

    fraudes_inc = FRAUDES_TESTE * INCUMBENTE_TETO
    prejuizo = {
        "Sem nenhum controle": -FRAUDES_TESTE * perda_fraude,
        "Regra atual (teto)": -((FRAUDES_TESTE - fraudes_inc) * perda_fraude
                                + LINHAS_TESTE * INCUMBENTE_FLAG_RATE * custo_revisao),
        "Modelo de tempo real": -(residual + operacao),
    }

    c = st.columns(3)
    c[0].metric("Perda evitada no período", moeda(evitada))
    c[1].metric("Custo de revisar os alertas", moeda(operacao))
    c[2].metric("Resultado líquido", moeda(evitada - operacao))

    st.divider()
    st.subheader("Prejuízo do período em cada cenário")

    dados = pd.DataFrame({"Cenário": list(prejuizo), "Prejuízo": list(prejuizo.values())})
    fig = px.bar(dados, x="Prejuízo", y="Cenário", orientation="h",
                 text=[moeda(v) for v in prejuizo.values()],
                 color="Cenário", color_discrete_map={
                     "Sem nenhum controle": VERMELHO,
                     "Regra atual (teto)": LARANJA,
                     "Modelo de tempo real": VERDE})
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_title=None,
                      xaxis_title="prejuízo do período (R$) — quanto mais perto de zero, melhor")
    mostrar(fig, 260)

    st.caption(
        f"Soma o que a fraude levou e o que custou revisar os alertas, sobre "
        f"{num(DIAS_TESTE, 1)} dias e {num(LINHAS_TESTE)} transações."
    )

    ganho = prejuizo["Modelo de tempo real"] - prejuizo["Regra atual (teto)"]
    st.markdown(
        f"Trocar a regra atual pelo modelo muda o resultado do período em "
        f"**{moeda_md(ganho)}**. Projetado para um ano ao mesmo volume: "
        f"**{moeda_md(ganho * 365 / DIAS_TESTE)}**."
    )
    st.caption(
        "A projeção multiplica o período de teste por 22,6, supondo volume e perfil de "
        "fraude constantes. Ordem de grandeza, não previsão."
    )

    st.divider()
    st.subheader("Sensibilidade à premissa de perda")

    faixa = pd.DataFrame({"Perda por fraude": range(0, 10_001, 250)})
    faixa["Resultado"] = faixa["Perda por fraude"] * VP - operacao

    fig = px.line(faixa, x="Perda por fraude", y="Resultado")
    fig.update_traces(line=dict(color=AZUL, width=2))
    fig.add_vline(x=perda_fraude, line_dash="dot", line_color=LARANJA,
                  annotation_text="sua premissa", annotation_position="top left")
    fig.add_hline(y=0, line_width=1, line_color="#9aa0a6")
    fig.update_layout(xaxis_title="perda média por fraude (R$)",
                      yaxis_title="resultado líquido do período (R$)")
    mostrar(fig, 300)

    razao = ALERTAS / VP
    st.markdown(
        f"O modelo gera {num(ALERTAS)} alertas para capturar {num(VP)} fraudes, "
        f"então **se paga enquanto uma fraude custar mais que {num(razao, 2)}× uma "
        f"revisão**. Essa razão sai direto da precisão e não depende de nenhuma "
        f"premissa — é o número mais robusto desta aba."
    )

# ═══ 5. TESTAR O MODELO ════════════════════════════════════════════════════ #

with abas[4]:
    esq, dir_ = st.columns([1, 1.1], gap="large")

    with esq:
        nome = st.selectbox("Partir de um cenário", list(CENARIOS))
        b_tipo, b_val, b_org, b_dest, b_merc, b_hora, b_dia = CENARIOS[nome]

        tipo = st.selectbox("Tipo", TIPOS, index=TIPOS.index(b_tipo))
        valor = st.number_input("Valor", min_value=0.0, value=b_val, step=100.0)
        saldo_org = st.number_input("Saldo da origem antes", min_value=0.0,
                                    value=b_org, step=100.0)
        saldo_dest = st.number_input("Saldo do destino antes", min_value=0.0,
                                     value=b_dest, step=100.0)
        comerciante = st.checkbox("Destinatário é comerciante", value=bool(b_merc),
                                  help="No PaySim, essas contas têm prefixo M em nameDest.")

        h, d = st.columns(2)
        hora = h.slider("Hora do dia", 0, 23, b_hora)
        dia = d.slider("Dia da semana", 0, 6, b_dia,
                       help="(step // 24) % 7 — índice cíclico, não dia de calendário.")

        with st.expander("Por que não há campos de saldo final"):
            st.markdown(
                "Estes campos existem no dataset e elevam muito a métrica, mas **não "
                "existem no momento da autorização**:"
            )
            st.code("\n".join(POS_LIQUIDACAO), language=None)
            st.caption(
                "Um modelo com eles vale para revisão forense, nunca para prevenção."
            )

    entrada = pd.DataFrame([{
        "type": tipo, "amount": valor,
        "oldbalanceOrg": saldo_org, "oldbalanceDest": saldo_dest,
        "is_merchant_dest": int(comerciante), "hour_of_day": hora, "day_of_week": dia,
    }])
    prob = float(pontuar(entrada, art).loc[0, "fraud_probability"])
    corte = float(art["threshold"])
    retida = prob >= corte

    with dir_:
        if retida:
            st.error("**Transação retida para revisão** — acima do ponto de operação.")
        else:
            st.success("**Transação autorizada** — abaixo do ponto de operação.")

        medidor = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".2f"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": VERMELHO if retida else VERDE},
                "steps": [{"range": [0, corte * 100], "color": "rgba(128,128,128,.15)"}],
                "threshold": {"line": {"color": LARANJA, "width": 3},
                              "value": corte * 100},
            }
        ))
        mostrar(medidor, 260)
        st.caption(f"A marca laranja é o threshold de operação: {pct(corte, 2)}.")

    st.divider()
    st.subheader("Pontuar um arquivo")
    st.markdown(
        "CSV com `type`, `amount`, `oldbalanceOrg` e `oldbalanceDest`. As features "
        "derivadas são calculadas aqui se o arquivo trouxer `step` e `nameDest`."
    )

    arquivo = st.file_uploader("CSV", type="csv", label_visibility="collapsed")

    if arquivo:
        try:
            dados = pd.read_csv(arquivo)
            if "is_merchant_dest" not in dados and "nameDest" in dados:
                dados["is_merchant_dest"] = (
                    dados["nameDest"].astype(str).str.startswith("M").astype(int))
            if "step" in dados:
                if "hour_of_day" not in dados:
                    dados["hour_of_day"] = dados["step"] % 24
                if "day_of_week" not in dados:
                    dados["day_of_week"] = (dados["step"] // 24) % 7

            pontuado = pontuar(dados, art)
            retidas = int(pontuado["flagged"].sum())

            c = st.columns(3)
            c[0].metric("Transações", num(len(pontuado)))
            c[1].metric("Retidas", num(retidas))
            c[2].metric("Taxa de retenção", pct(retidas / len(pontuado), 3))

            fig = px.histogram(pontuado, x="fraud_probability", nbins=50, log_y=True)
            fig.update_traces(marker_color=AZUL)
            fig.add_vline(x=corte, line_dash="dot", line_color=LARANJA,
                          annotation_text="corte")
            fig.update_layout(xaxis_title="probabilidade de fraude",
                              yaxis_title="transações (escala log)")
            mostrar(fig, 280)

            st.dataframe(
                pontuado.sort_values("fraud_probability", ascending=False).head(200),
                width="stretch", hide_index=True)
            st.download_button("Baixar resultado completo",
                               pontuado.to_csv(index=False).encode("utf-8"),
                               file_name="transacoes_pontuadas.csv", mime="text/csv")
        except Exception as erro:  # noqa: BLE001 — a mensagem vai para a tela
            st.error(f"Não consegui processar o arquivo: {erro}")

# ═══ 6. LIMITAÇÕES ═════════════════════════════════════════════════════════ #

with abas[5]:
    st.subheader("O que estes números não provam")

    for titulo, texto in [
        ("Dado sintético",
         "Os agentes de fraude do PaySim seguem regras programadas, aprendíveis de um "
         "jeito que adversários reais não são. Toda métrica aqui é um limite superior "
         "do desempenho no mundo real."),
        ("Sem deriva adversarial",
         "A validação cobre um mês de uma simulação estática. Padrões reais mudam em "
         "resposta à detecção, então o desempenho em produção decai sem retreino."),
        ("Escopo de transação única",
         "Cada transação é pontuada isoladamente. O modelo não enxerga que uma conta "
         "recebeu três transferências nos dez minutos anteriores, que é como "
         "quadrilhas de fato aparecem. A restrição é o conjunto de features, não o "
         "modelo — por isso a busca de hiperparâmetros estabiliza cedo."),
        ("Custo simétrico",
         "O threshold padrão maximiza F1, precificando um cliente bloqueado e uma "
         "fraude perdida como se fossem a mesma coisa. A aba de impacto financeiro "
         "existe para tornar essa assimetria explícita."),
    ]:
        with st.container(border=True):
            st.markdown(f"**{titulo}**")
            st.markdown(texto)

    st.subheader("Próximos passos")
    st.markdown(
        "1. Agregados de velocidade por conta sobre janela móvel de `step`, calculados "
        "só a partir de linhas passadas.\n"
        "2. Rederivar o ponto de operação contra custos explícitos de falso positivo e "
        "falso negativo.\n"
        "3. Monitoramento da distribuição de predições e da taxa de alerta, para pegar "
        "deriva antes de o recall degradar em silêncio.\n"
        "4. Validação contra dado transacional real antes de qualquer produção."
    )

st.divider()
st.caption(
    f"Modelo treinado em {art['trained_at'][:10]} · scikit-learn "
    f"{art['sklearn_version']} · XGBoost {art['xgboost_version']} · "
    "github.com/diogo-moitinho/fraud-detection-project"
)
