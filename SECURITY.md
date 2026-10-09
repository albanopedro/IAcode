# Segurança e privacidade do JARVIS

O JARVIS roda no seu Mac e fala com agentes de IA de terceiros. Este documento lista as
ameaças consideradas, como cada uma é mitigada e qual teste automatizado comprova isso.

## Princípios

1. **Custo zero é uma regra de segurança.** Nenhuma chamada que possa gerar cobrança.
2. **Os agentes não têm acesso direto a nada.** Eles pedem; o JARVIS decide.
3. **Todo conteúdo externo é dado, nunca instrução.** Isso vale para páginas,
   arquivos, resultados de ferramentas e a própria conversa no resumo.
4. **Na dúvida, negar.** Erros de cobrança bloqueiam o agente, confirmações sem
   resposta são negadas e ferramentas indisponíveis nem são oferecidas.

## Ameaças e mitigações

### Custo

| Ameaça | Mitigação | Teste |
|---|---|---|
| Um agente pago ser usado | `COST_MODE=FREE_ONLY` vem só do ambiente; um valor inválido vira `FREE_ONLY`; agentes `paid` ficam bloqueados no registro | `test_cost_guard.py`, `test_config.py` |
| Um provedor começar a cobrar | `cost > 0` informado, HTTP 402 ou mensagem de cobrança bloqueiam o agente, e o bloqueio **persiste** até `jarvis unblock` | `test_phase3_limits.py`, `test_orchestrator.py` |
| Modelo pago com nome parecido | OpenRouter só `:free`; OpenCode só `*-free` com preço 0 em todas as faixas | `test_openai_compat.py`, `test_opencode_adapter.py` |
| Credencial paga no OpenCode | O OpenCode roda numa casa própria (XDG isolado): só o Zen, com a chave anônima `public`; qualquer outra coisa bloqueia | `test_opencode_adapter.py` |

### Acesso ao computador

| Ameaça | Mitigação | Teste |
|---|---|---|
| O agente OpenCode usar as ferramentas nativas dele (shell, arquivos) | Agente `jarvis` com "pedir aprovação para tudo" como última regra; o modo não interativo rejeita; `--auto` nunca é usado; uma ferramenta executada descarta a resposta e bloqueia o agente | `test_opencode_adapter.py` e teste manual (`ls ~` rejeitado) |
| Código malicioso via `run_python` | Sandbox do macOS (`sandbox-exec`): sem arquivos do usuário, sem rede, sem fork e sem exec; limite de CPU (5 s) e de tempo (10 s); ambiente vazio; **sempre** pede confirmação | `test_tools.py` (6 ataques reais bloqueados) |
| Ler segredos com `read_file` | Desligada sem `allowed_dirs`; caminhos resolvidos (symlinks incluídos) precisam estar na pasta liberada; `.env`, chaves SSH/GPG/nuvem e `.pem` são recusados; só texto até 200 KB | `test_tools.py` |
| Calculadora usada para executar código | Avaliação por AST, sem `eval`; só números, operadores e funções `math` | `test_tools.py` |

### Prompt injection

| Ameaça | Mitigação | Teste |
|---|---|---|
| Uma página ou arquivo mandar o agente fazer algo | O resultado vai marcado como "CONTEÚDO EXTERNO NÃO CONFIÁVEL"; depois dele, **toda** ferramenta sensível exige sua confirmação | `test_tools.py` (`test_untrusted_content…`) |
| Exfiltrar dados por URL (`fetch_url`) | `fetch_url` sempre pede confirmação e mostra o endereço exato | `test_tools.py` |
| Comando de voz mal transcrito apagar tudo | "Esqueça tudo" é recusado por voz e texto; apagar tudo exige um botão com confirmação ou digitar "SIM" | `test_memory.py`, `test_cli.py` |

### Rede

| Ameaça | Mitigação | Teste |
|---|---|---|
| Outro site no navegador usar a API local | Verificação de `Origin`: REST responde 403 e WebSocket fecha com 1008 | `test_server.py` |
| DNS rebinding contra o servidor local | O cabeçalho `Host` precisa ser `localhost` ou `127.0.0.1` (requisições GET da mesma origem não mandam `Origin`) | `test_server.py` |
| Clickjacking do botão "Permitir" | `X-Frame-Options: DENY` e CSP com `frame-ancestors 'none'` | `test_server.py` |
| SSRF via `fetch_url` (roteador, localhost) | Só http(s) em IPs públicos; cada redirecionamento é revalidado; **o IP realmente conectado** é verificado antes de ler a resposta | `test_tools.py` |
| Servidor exposto na rede | `jarvis serve` só escuta em `127.0.0.1`; os servidores internos (OpenCode, MLX) também, em portas aleatórias e com senha ou sem aceitar origens | `test_local.py` |
| Servidor MLX aceitando qualquer site (o padrão é `*`) | Iniciado com uma origem impossível e com o Hugging Face offline | `test_local.py` |

### Privacidade

| Ameaça | Mitigação | Teste |
|---|---|---|
| Lembranças pessoais irem para provedores que treinam com dados | Por padrão vão só para agentes de retenção zero ou locais | `test_memory.py` |
| Mensagens do modo "só local" vazarem depois | Ficam marcadas como privadas (gravado no banco) e nunca entram no contexto de agentes online nem nos resumos | `test_local.py`, `test_server.py` e teste manual |
| Chaves vazarem para o navegador, logs ou diagnóstico | Lidas só na hora do uso; nunca aparecem em status, eventos, `doctor` ou erros | `test_server.py`, `test_cli.py` |
| Chaves do shell chegarem ao OpenCode ou ao MLX | Ambiente por allowlist | `test_opencode_adapter.py`, `test_local.py` |

## Riscos aceitos

- **Qualquer processo do seu usuário** pode falar com `127.0.0.1:8300`. Proteger contra
  malware já instalado na máquina está fora do escopo.
- Os provedores gratuitos podem registrar as conversas (veja a coluna de privacidade no
  painel de agentes). Para assuntos sensíveis, use o modo "🔒 só local".
- A sandbox depende do `sandbox-exec` do macOS, que a Apple considera obsoleto, mas que
  ainda funciona (verificado no macOS 27). Sem ele, `run_python` fica desligado.

## Verificações feitas na Fase 9

- `pip-audit` nas 64 dependências Python: nenhuma vulnerabilidade conhecida.
- `npm audit` no frontend: 0 vulnerabilidades.
- Cobertura de testes do backend: 85% (266 testes), mais 18 testes no frontend.
- Nenhum recurso (banco SQLite ou pipe de processo) fica aberto pelo código do JARVIS.

## Reportar um problema

Este é um projeto pessoal. Se encontrar uma falha, descreva o cenário e como reproduzir,
**sem** incluir chaves ou dados pessoais.
