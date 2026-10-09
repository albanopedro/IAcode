"""Internet tools: Wikipedia (official free API) and reading a web page.

Fetching pages is protected against SSRF: only http/https on public addresses —
never localhost, your router, or anything on your local network, including after
redirects. Pages are size-limited and reduced to plain text.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from jarvis.tools.base import Param, Policy, Risk, Tool, ToolContext, ToolError, ToolResult

USER_AGENT = "JARVIS-local/0.1 (personal assistant; contact: local user)"
MAX_PAGE_BYTES = 1_000_000
MAX_TEXT_CHARS = 6000
MAX_REDIRECTS = 3
WIKI_LANGS = frozenset({"pt", "en", "es"})


class _TextExtractor(HTMLParser):
    SKIP = frozenset({"script", "style", "noscript", "svg", "head", "nav", "footer", "form"})
    BLOCK = frozenset({"p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skipping = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self.skipping += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP and self.skipping:
            self.skipping -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self.skipping:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    text = "\n".join(line for line in lines if line)
    return parser.title.strip(), text


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    return ip.is_global and not ip.is_multicast


async def check_public_url(url: str) -> str:
    """Raise ToolError unless ``url`` is http(s) and every address of its host is public."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ToolError("só endereços http:// ou https://")
    if not parts.hostname or parts.username or parts.password:
        raise ToolError("endereço inválido")
    host = parts.hostname
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, parts.port or 443)
    except socket.gaierror as exc:
        raise ToolError(f"não encontrei o site {host}") from exc
    addresses = {info[4][0] for info in infos}
    if not addresses or not all(_is_public(a) for a in addresses):
        raise ToolError("endereço bloqueado: só sites públicos da internet (nada local/privado)")
    return url


class FetchUrlTool(Tool):
    name = "fetch_url"
    title = "Ler página da web"
    description = (
        "Baixa uma página pública da internet e devolve o texto dela (sem imagens). "
        "Use só com endereços que o usuário deu ou que vieram de uma busca."
    )
    params = (Param("url", "endereço completo, começando com https://"),)
    risk = Risk.SENSITIVE
    default_policy = Policy.CONFIRM
    untrusted_output = True
    timeout = 20.0

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    def describe_call(self, args: dict[str, Any]) -> str:
        return f"Abrir {str(args.get('url', ''))[:200]}"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        url = self.arg(args, "url", max_len=2000)
        async with httpx.AsyncClient(
            timeout=15,
            follow_redirects=False,
            transport=self._transport,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                await check_public_url(url)
                async with client.stream("GET", url) as response:
                    if response.is_redirect and "location" in response.headers:
                        url = urljoin(url, response.headers["location"])
                        continue
                    if response.status_code >= 400:
                        raise ToolError(f"o site respondeu HTTP {response.status_code}")
                    kind = response.headers.get("content-type", "").split(";")[0].strip()
                    if kind not in ("text/html", "text/plain", "application/json", ""):
                        raise ToolError(f"tipo de conteúdo não suportado: {kind}")
                    body = b""
                    async for chunk in response.aiter_bytes():
                        body += chunk
                        if len(body) > MAX_PAGE_BYTES:
                            break
                    text = body[:MAX_PAGE_BYTES].decode(response.encoding or "utf-8", "replace")
                    break
            else:
                raise ToolError("redirecionamentos demais")
        title, content = html_to_text(text) if kind in ("text/html", "") else ("", text)
        if len(content) > MAX_TEXT_CHARS:
            content = content[:MAX_TEXT_CHARS] + "\n[…página cortada]"
        header = f"Página: {title}\n" if title else ""
        return ToolResult(f"{header}Endereço: {url}\n\n{content}")


class WikipediaTool(Tool):
    name = "wikipedia"
    title = "Wikipédia"
    description = (
        "Pesquisa na Wikipédia (API oficial e gratuita) e devolve o resumo do artigo mais "
        "relevante. Bom para fatos, definições, pessoas, lugares e história."
    )
    params = (
        Param("query", "o que pesquisar"),
        Param("lang", "idioma: pt (padrão), en ou es", required=False),
    )
    risk = Risk.SENSITIVE  # your question goes to wikipedia.org
    default_policy = Policy.ALLOW
    untrusted_output = True
    timeout = 20.0

    def __init__(self, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._transport = transport

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = self.arg(args, "query", max_len=200)
        lang = str(args.get("lang") or "pt").lower()
        if lang not in WIKI_LANGS:
            lang = "pt"
        api = f"https://{lang}.wikipedia.org/w/api.php"
        async with httpx.AsyncClient(
            timeout=15, transport=self._transport, headers={"User-Agent": USER_AGENT}
        ) as client:
            try:
                found = await client.get(
                    api,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": 3,
                        "format": "json",
                        "formatversion": 2,
                    },
                )
                hits = found.json()["query"]["search"]
                if not hits:
                    return ToolResult(f"Nada encontrado na Wikipédia ({lang}) para: {query}")
                title = hits[0]["title"]
                page = await client.get(
                    api,
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "exintro": 1,
                        "explaintext": 1,
                        "titles": title,
                        "format": "json",
                        "formatversion": 2,
                        "redirects": 1,
                    },
                )
                extract = page.json()["query"]["pages"][0].get("extract", "")
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                raise ToolError(
                    f"a Wikipédia não respondeu como esperado ({type(exc).__name__})"
                ) from exc
        others = ", ".join(h["title"] for h in hits[1:])
        url = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
        text = extract[:MAX_TEXT_CHARS] or "(artigo sem resumo)"
        more = f"\nOutros artigos: {others}" if others else ""
        return ToolResult(f"{title} — {url}\n\n{text}{more}")
