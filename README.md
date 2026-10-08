# JARVIS AI

Assistente pessoal multimodelo inspirado no JARVIS. Conversa com vários agentes de IA
**100% gratuitos** e troca de agente sozinho quando um atinge o limite ou falha.

> **Regra absoluta: custo zero.** `COST_MODE=FREE_ONLY` é o padrão e só pode ser mudado
> pelo ambiente. Um agente pago fica bloqueado por padrão, e qualquer custo maior que zero
> informado por um provedor bloqueia esse agente na hora.

## Estado

| Fase | Status |
|---|---|
| 1. Pesquisa e arquitetura | ✅ |
| 2. Core: orchestrator, adapters, fallback, status | ✅ |
| 3. Multi-agent: limites, ranking, health checks, persistência | ✅ (aguardando revisão) |
| 4–9. Voz, interface, memória, tools, IA local, polimento | ⏳ |

## Como rodar (backend, texto)

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m jarvis status          # saúde dos agentes (não gasta cota); --json
.venv/bin/python -m jarvis chat            # conversa: /status, /limpar, /sair
.venv/bin/python -m jarvis ask "Explique Docker em uma frase"
.venv/bin/python -m jarvis unblock <id>    # libera um agente bloqueado por custo
```

Testes: `.venv/bin/pytest`. O teste real e gratuito com o OpenCode é opcional:
`JARVIS_LIVE_TESTS=1 .venv/bin/pytest -m live`.

### Agentes online opcionais

Copie `.env.example` para `.env` e preencha só o que for usar. As contas devem estar no
**plano grátis, sem cartão cadastrado**:

- `GROQ_API_KEY`: plano Free do Groq, com limite diário que reinicia.
- `OPENROUTER_API_KEY`: só modelos `:free`, e nunca comprar créditos.
- `MISTRAL_API_KEY`: plano Experiment (verificação por telefone, sem cartão). Por
  padrão os dados podem ser usados para treino; dá para desligar no console.
- `CLOUDFLARE_ACCOUNT_ID` e `CLOUDFLARE_API_TOKEN`: plano Workers Free, com 10.000
  Neurons por dia. Passado o limite, os pedidos falham em vez de cobrar.

Sem chave, o agente aparece como ⚪ `unconfigured` e o JARVIS segue com os outros.

## Arquitetura

```text
Usuário ─► Orchestrator ─► classifica a tarefa (chat / code / math / research)
                │
                ├─► Agent Manager: quem está disponível? (cooldown, cota, erros, bloqueios)
                ├─► Router: pontua os candidatos (capacidade > prioridade, penaliza falhas/lentidão)
                ├─► Cost Guard: só local / free / free_with_limits; custo > 0 ⇒ bloqueio
                └─► tenta o melhor; se falhar, registra e tenta o próximo
                         │
        ┌────────────────┼──────────────────────┐
   OpenCode (Zen free)  Groq (free plan)   OpenRouter (:free)   … novos adapters
```

- `backend/jarvis/core/`: tipos, erros, cost guard, agent manager, classifier, router,
  conversation e orchestrator.
- `backend/jarvis/agents/`:
  - `opencode/`: sandbox, servidor privado, runtime e adapter.
  - `openai_compat/`: adapter genérico para APIs no formato OpenAI.
  - `factory.py`: monta os agentes a partir da configuração.
- `config/agents.toml`: agentes, prioridades e capacidades. **Não guarda segredos.**
- O histórico da conversa pertence ao JARVIS, e não ao agente. Por isso trocar de agente
  no meio da conversa não perde o contexto.

### Limites, ranking e saúde (Fase 3)

- **Limites:**
  - por dia (`daily_limit`) e por minuto (`rpm_limit`), respeitados localmente antes
    do provedor responder com 429;
  - a cota informada pelo próprio provedor nos cabeçalhos (`x-ratelimit-*`) tem
    prioridade;
  - `quota_group`: modelos que dividem a cota da mesma conta (por exemplo, os
    gratuitos do OpenRouter) dividem também os contadores.
- **Persistência** (`data/jarvis.db`, só contadores e estados, nunca conversas nem
  chaves):
  - o uso do dia e os cooldowns sobrevivem a um reinício;
  - **um agente bloqueado por custo continua bloqueado** até um
    `jarvis unblock <id>` explícito.
- **Ranking:**
  - prioridade, mais qualidade por tarefa (`quality` no `agents.toml`);
  - menos penalidades por taxa de sucesso recente, falhas seguidas e latência;
  - bônus de privacidade;
  - agentes cuja janela de contexto não comporta a conversa ficam de fora.
- **Saúde:** no `jarvis chat`, um monitor em segundo plano verifica todos os
  agentes a cada `health_interval` segundos, sem gastar cota. Um agente offline
  volta sozinho quando a verificação passa.

### Como o OpenCode é usado (e por quê)

O OpenCode Zen só libera o plano grátis para **o próprio cliente OpenCode, com as
ferramentas do agente declaradas**. Pedidos feitos pela API HTTP recebem
`FreeTierError: can only be used from within OpenCode`. O JARVIS respeita isso e não
tenta contornar. Por isso ele usa o cliente oficial, `opencode run --server`, preso
numa sandbox:

1. **Casa própria do OpenCode** (XDG config/data/state temporários). O JARVIS nunca
   vê as suas credenciais nem o seu histórico. O único provedor é o Zen, com a chave
   anônima `public`, então não há o que cobrar. A verificação recusa qualquer outra
   situação.
2. **Ambiente mínimo** (allowlist). Chaves como `OPENAI_API_KEY` e variáveis
   `OPENCODE_*` do seu shell não chegam ao OpenCode.
3. **Pasta de trabalho vazia**, também definida como `PWD`. O OpenCode usa o `PWD`
   para escolher a pasta do projeto.
4. **Toda ferramenta pede aprovação** (`ask` como última regra do agente `jarvis`).
   No modo não interativo, o próprio OpenCode rejeita cada uma. O `--auto` nunca é
   usado. Se alguma ferramenta chegar a rodar, a resposta é descartada e o agente,
   bloqueado.
5. **Só modelos grátis**: nome `*-free` ou `big-pickle` **e** preço 0 em todas as faixas.

Verificado na prática:
- um pedido para ler um arquivo-isca e outro para rodar `ls ~` foram rejeitados pelo
  OpenCode;
- todas as respostas tiveram custo 0.
