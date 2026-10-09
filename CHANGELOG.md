# Changelog

## 1.1.0 — Wake word

- "Hey Jarvis" com openWakeWord (local, ~9 MB): `jarvis voice --wake` ou o interruptor
  "Ativar por Hey Jarvis" na interface.
- Ciclo dormindo → acordado ("Sim?") → conversa → volta a dormir depois de um silêncio
  ou de "tchau JARVIS". Enquanto dorme, nada é gravado nem enviado.
- Estado "dormindo" no orb e no WebSocket; o modelo aparece no `jarvis doctor`.
- Testado com áudio sintetizado: "Hey Jarvis" pontua 0,99; frases comuns em português
  ficam abaixo de 0,02.

## 1.0.0 — Fase 9: polimento

- **Segurança:**
  - verificação do cabeçalho `Host` contra DNS rebinding;
  - cabeçalhos de segurança (CSP, `frame-ancestors 'none'`, `X-Frame-Options`,
    `nosniff`, `Referrer-Policy`, `Permissions-Policy`);
  - o `fetch_url` passa a verificar o IP realmente conectado.
- **Recursos:** fechados os vazamentos de bancos SQLite e de pipes de processo.
- **Experiência:**
  - `jarvis doctor`, um diagnóstico que nunca mostra chaves;
  - `jarvis serve --open`;
  - inicializador `JARVIS.command`, para abrir com dois cliques no Finder.
- **Qualidade:**
  - testes da CLI;
  - cobertura de 79% para 85% (266 testes no backend, 18 no frontend);
  - `pip-audit` e `npm audit` sem vulnerabilidades.
- **Documentação:** README novo, `SECURITY.md` (modelo de ameaças),
  `docs/architecture.md`.

## Fase 8: IA local

- Modelo offline Qwen3-4B (MLX-LM, Apache-2.0), que assume sozinho sem internet.
- Modo "🔒 só local" na CLI (`--local`) e na interface. As mensagens desse modo ficam
  privadas, persistidas, e nunca vão para agentes online nem para os resumos.
- Servidor MLX travado: só local, sem origens web e com o Hugging Face offline.

## Fase 7: ferramentas

- Protocolo de ferramentas em texto, executor com política, confirmação,
  "taint" contra prompt injection, limites e auditoria.
- Calculadora (AST), data e hora, Wikipédia, leitura de páginas (anti-SSRF), tarefas,
  arquivos (pastas liberadas) e Python na sandbox do macOS.
- Confirmação pelo terminal ou por um diálogo na web.

## Fase 6: memória

- Histórico em SQLite (continua depois de recarregar), resumo das mensagens antigas e
  lembranças de longo prazo com comandos locais.

## Fase 5: interface

- FastAPI + WebSocket; React com orb, voz no navegador, painel de agentes; verificação
  de `Origin`.

## Fase 4: voz

- whisper.cpp, `say`/Piper, VAD, conversa contínua half-duplex e estilo curto para voz.

## Fase 3: multiagente

- Cotas por minuto e por dia, grupos de cota, cabeçalhos de cota dos provedores,
  ranking v2, monitor de saúde, persistência e mais agentes gratuitos.

## Fase 2: núcleo

- Orchestrator, AgentManager, CostGuard e Router; adapters OpenCode (via
  `opencode run` em sandbox) e OpenAI-compatível.

## Fase 1: pesquisa e arquitetura

- Levantamento dos provedores realmente gratuitos e escolha da stack.
