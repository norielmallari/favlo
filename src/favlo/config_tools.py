"""The configuration tool group. Favlo web enables it on the Favlo agent."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

GROUP_ID = "favlo-config"
GROUP_NAME = "Configuration tools"

INSTRUCTIONS = (
    "When someone asks how to connect a client, which address or model id to use, "
    "how to download or quantize a supported model, or how to register one, "
    "call read_guide before you answer. "
    "Use clients for the server address, the model field, reload, "
    "and stopping the server with quit. "
    "Use models for which checkpoints can be downloaded and how registration works. "
    "When someone wants to set up Qwen3.6-35B-A3B, call read_setup with qwen36-35b-a3b "
    "before you answer. When someone wants to set up Qwen3.8-27B, call read_setup with "
    "qwen38-27b before you answer. When someone wants to set up Z-Image-Turbo, call "
    "read_setup with z-image-turbo before you answer. When someone wants to set up "
    "Wan 2.2 I2V, call read_setup with wan22-i2v before you answer. When someone wants to "
    "set up ACE-Step, call read_setup with ace-step-15-turbo before you answer. For any other "
    "pipeline "
    "they want to set up, download, quantize, or delete, call read_setup with that pipeline id. "
    "Answer from that guide. "
    "When they want you to run a step from it, call run_setup with that command. "
    "run_setup only runs inside this install. Do not run a command the guide does not contain. "
    "When someone asks which models are registered, or which model id to send, "
    "call list_models before you answer. Report only the lines it returns. "
    "When someone asks where your name came from, or who Amanda and Celestia are, "
    "call read_guide with history and tell that story. Do not add events it does not contain. "
    "Answer from that guide. If the guide does not say it, say you do not know."
)


def tools(base: str, home: Path | None) -> list[object]:
    """The configuration tools, in the order the group offers them."""
    return [
        _guide_tool(),
        _models_tool(),
        _suggestions_tool(home),
        _setup_guide_tool(),
        _setup_run_tool(base),
    ]


def registered_models(root: Path | None = None) -> str:
    """The registry listing. An empty registry says so."""
    from favlo.registry import format_listing

    listing = format_listing(root).strip()
    if not listing:
        return "no models are registered"
    return listing


def _guide_tool() -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    from favlo.guides import read_guide as load_guide

    @tool(approval_mode="never_require")
    def read_guide(
        name: Annotated[str, Field(description="clients, models, or history.")],
    ) -> str:
        """Read one guide.

        clients points a program at this server, covers ``reload``,
        and ``quit`` stops the server. models covers
        download, quantize, and registration. history is where the name Favlo came from.
        """
        return load_guide(name)

    return read_guide


def _suggestions_tool(home: Path | None) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    from favlo.machine import set_suggestions

    @tool(approval_mode="never_require")
    def download_suggestions(
        enabled: Annotated[
            bool,
            Field(description="False stores download reminders as off. True stores them as on."),
        ],
    ) -> str:
        """Store whether download reminders are enabled.

        The terminal does not start a reminder on its own. This records the preference.
        """
        return set_suggestions(enabled, home)

    return download_suggestions


def _setup_guide_tool() -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    from favlo.guides import read_setup as load_setup

    @tool(approval_mode="never_require")
    def read_setup(
        pipeline: Annotated[
            str,
            Field(description="Pipeline id, such as wan22-i2v."),
        ],
    ) -> str:
        """Read one pipeline's setup guide.

        Call this when someone wants to set up, download, quantize, or delete a model.
        qwen36-35b-a3b is Qwen3.6-35B-A3B. qwen38-27b is Qwen3.8-27B.
        z-image-turbo is Z-Image-Turbo. wan22-i2v is Wan 2.2 I2V.
        ace-step-15-turbo is ACE-Step.
        The result is that pipeline's setup page.
        A pipeline with no setup page names the ones that have one.
        """
        return load_setup(pipeline)

    return read_setup


def _setup_run_tool(base: str) -> object:
    from typing import Annotated

    from agent_framework import tool
    from pydantic import Field

    from favlo.setup_run import accepted_argv, frees_model, run_setup_command

    @tool(approval_mode="never_require")
    def run_setup(
        command: Annotated[
            str,
            Field(description="One command copied from the setup guide."),
        ],
    ) -> str:
        """Run one setup-guide command in this install.

        The working directory is the folder where Favlo is installed. A path cannot
        leave that folder. This is not a shell: pipes, cd, and other programs are refused.
        """
        try:
            argv = accepted_argv(command, _install())
        except ValueError as exc:
            return str(exc)
        if frees_model(argv):
            closed = _close_resident(base)
            if closed is not None:
                return closed
        try:
            return run_setup_command(command)
        except ValueError as exc:
            return str(exc)

    return run_setup


def _install() -> Path:
    from favlo.cli import repo_root

    return repo_root()


def _close_resident(base: str) -> str | None:
    request = urllib.request.Request(base + "/pipelines/close", data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
    except (OSError, urllib.error.URLError) as exc:
        return f"could not close the open model: {exc}"
    return None


def _models_tool() -> object:
    from agent_framework import tool

    @tool(approval_mode="never_require")
    def list_models() -> str:
        """List the registered models.

        Each line is one record: folder, pipeline, alias, and key/value mode.
        An unregistered complete record is named too. This is the live registry,
        not the download guide.
        """
        return registered_models()

    return list_models
