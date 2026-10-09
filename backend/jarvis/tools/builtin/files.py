"""Read-only file tools, limited to the folders you list in config/agents.toml.

- Nothing works until you configure ``allowed_dirs`` (empty by default).
- Paths are resolved (symlinks included) and must stay inside an allowed folder,
  so "../" tricks and links pointing outside are refused.
- Secrets are never read: .env files, SSH/GPG/cloud credentials, keys and certificates.
- Only text files up to 200 KB. Reading asks for confirmation by default, because
  the content is sent to the agent that answers.
"""

from __future__ import annotations

import fnmatch
from datetime import datetime
from pathlib import Path
from typing import Any

from jarvis.tools.base import Param, Policy, Risk, Tool, ToolContext, ToolError, ToolResult

MAX_FILE_BYTES = 200_000
MAX_ENTRIES = 100
SECRET_NAMES = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.keychain*",
    "id_rsa*",
    "id_ed25519*",
    "id_ecdsa*",
    "*.kdbx",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials*",
    "*.sqlite",
    "*.db",
)
SECRET_DIRS = frozenset({".ssh", ".gnupg", ".aws", ".azure", ".kube", ".docker", ".config/gcloud"})


class FileAccess:
    def __init__(self, allowed_dirs: list[str | Path]) -> None:
        self.roots = [Path(d).expanduser().resolve() for d in allowed_dirs if str(d).strip()]

    def unavailable(self) -> str | None:
        if not self.roots:
            return "nenhuma pasta liberada (configure [tools] allowed_dirs no agents.toml)"
        return None

    def resolve(self, raw: str) -> Path:
        if not self.roots:
            raise ToolError(self.unavailable() or "")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.roots[0] / candidate
        path = candidate.resolve()
        root = next((r for r in self.roots if path == r or path.is_relative_to(r)), None)
        if root is None:
            allowed = ", ".join(str(r) for r in self.roots)
            raise ToolError(f"fora das pastas liberadas ({allowed})")
        relative = path.relative_to(root).as_posix()
        parts = set(Path(relative).parts)
        if any(d in parts or relative.startswith(d + "/") for d in SECRET_DIRS) or any(
            fnmatch.fnmatch(path.name.lower(), pattern) for pattern in SECRET_NAMES
        ):
            raise ToolError("arquivo protegido (pode conter senhas ou chaves)")
        return path


class ReadFileTool(Tool):
    name = "read_file"
    title = "Ler arquivo"
    description = "Lê um arquivo de texto (até 200 KB) de uma pasta liberada pelo usuário."
    params = (Param("path", "caminho do arquivo (absoluto ou relativo à primeira pasta liberada)"),)
    risk = Risk.SENSITIVE
    default_policy = Policy.CONFIRM
    untrusted_output = True

    def __init__(self, access: FileAccess) -> None:
        self.access = access

    def available(self) -> str | None:
        return self.access.unavailable()

    def describe_call(self, args: dict[str, Any]) -> str:
        return f"Ler o arquivo {args.get('path', '')} (o conteúdo vai para o agente)"

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        path = self.access.resolve(self.arg(args, "path", max_len=1000))
        if not path.is_file():
            raise ToolError("arquivo não encontrado")
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ToolError(f"arquivo grande demais ({size // 1000} KB; máximo 200 KB)")
        data = path.read_bytes()
        if b"\x00" in data[:4096]:
            raise ToolError("parece um arquivo binário; só leio texto")
        return ToolResult(f"Arquivo: {path}\n\n{data.decode('utf-8', 'replace')}")


class ListDirTool(Tool):
    name = "list_dir"
    title = "Listar pasta"
    description = "Lista arquivos e subpastas de uma pasta liberada pelo usuário."
    params = (Param("path", "caminho da pasta (vazio = primeira pasta liberada)", required=False),)
    risk = Risk.SENSITIVE
    default_policy = Policy.ALLOW

    def __init__(self, access: FileAccess) -> None:
        self.access = access

    def available(self) -> str | None:
        return self.access.unavailable()

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw = str(args.get("path") or "").strip()
        path = self.access.resolve(raw) if raw else self.access.roots[0]
        if not path.is_dir():
            raise ToolError("pasta não encontrada")
        entries = sorted(
            (p for p in path.iterdir() if not p.name.startswith(".")),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
        lines = []
        for entry in entries[:MAX_ENTRIES]:
            if entry.is_dir():
                lines.append(f"[pasta] {entry.name}/")
            else:
                stat = entry.stat()
                when = datetime.fromtimestamp(stat.st_mtime).strftime("%d/%m/%Y")
                lines.append(f"{entry.name} ({stat.st_size:,} bytes, {when})")
        more = f"\n… e mais {len(entries) - MAX_ENTRIES}" if len(entries) > MAX_ENTRIES else ""
        return ToolResult(f"Pasta: {path}\n" + ("\n".join(lines) or "(vazia)") + more)
