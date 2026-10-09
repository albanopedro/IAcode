"""Task list (to-dos), stored locally in data/memory.db."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from jarvis.memory.store import MemoryStore
from jarvis.tools.base import Param, Policy, Risk, Tool, ToolContext, ToolError, ToolResult


def _task_id(args: dict[str, Any]) -> int:
    try:
        return int(str(args.get("id", "")).strip())
    except ValueError as exc:
        raise ToolError("informe o número da tarefa (veja list_tasks)") from exc


class _TaskTool(Tool):
    risk = Risk.SENSITIVE  # changes JARVIS's own data, nothing outside it

    def __init__(self, store: MemoryStore | None) -> None:
        self.store = store

    def available(self) -> str | None:
        return None if self.store is not None else "a memória está desligada"


class AddTaskTool(_TaskTool):
    name = "add_task"
    title = "Adicionar tarefa"
    description = "Adiciona uma tarefa à lista de tarefas do usuário."
    params = (Param("text", "a tarefa, por exemplo 'comprar pão amanhã'"),)
    default_policy = Policy.ALLOW

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        task = self.store.add_task(self.arg(args, "text", max_len=300))
        return ToolResult(f"Tarefa {task.id} adicionada: {task.text}")


class ListTasksTool(_TaskTool):
    name = "list_tasks"
    title = "Listar tarefas"
    description = "Mostra as tarefas pendentes do usuário (com os números de cada uma)."
    default_policy = Policy.ALLOW

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tasks = self.store.tasks()
        if not tasks:
            return ToolResult("Nenhuma tarefa pendente.")
        lines = [
            f"{t.id}. {t.text} (desde {datetime.fromtimestamp(t.created_at):%d/%m})" for t in tasks
        ]
        return ToolResult("Tarefas pendentes:\n" + "\n".join(lines))


class CompleteTaskTool(_TaskTool):
    name = "complete_task"
    title = "Concluir tarefa"
    description = "Marca uma tarefa como concluída, pelo número."
    params = (Param("id", "número da tarefa"),)
    default_policy = Policy.ALLOW

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        task_id = _task_id(args)
        if not self.store.complete_task(task_id):
            raise ToolError(f"não há tarefa pendente com o número {task_id}")
        return ToolResult(f"Tarefa {task_id} concluída.")


class DeleteTaskTool(_TaskTool):
    name = "delete_task"
    title = "Apagar tarefa"
    description = "Apaga uma tarefa da lista, pelo número."
    params = (Param("id", "número da tarefa"),)
    default_policy = Policy.CONFIRM

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        task_id = _task_id(args)
        if not self.store.delete_task(task_id):
            raise ToolError(f"não há tarefa com o número {task_id}")
        return ToolResult(f"Tarefa {task_id} apagada.")
