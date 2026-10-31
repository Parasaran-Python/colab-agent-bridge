from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from colab_agent_bridge.config import get_config
from colab_agent_bridge.core.session import SessionManager
from colab_agent_bridge.core.hardware import recommend_model_strategy
from colab_agent_bridge.client.bridge import ColabBridge
from colab_agent_bridge.client.harness import generate_agent_config
from colab_agent_bridge.client.health import check_health

app = typer.Typer(
    name="colab-bridge",
    help="Bridge Google Colab GPU/TPU model serving to local coding agent harnesses.",
    add_completion=False,
)
console = Console()


@app.command()
def status(
    session_name: Optional[str] = typer.Option(None, "-s", "--session", help="Session name to inspect"),
):
    """Check active Colab sessions, hardware status, and CLI authentication."""
    config = get_config()
    mgr = SessionManager(config)
    auth_check = mgr.verify_auth()

    if not auth_check["authenticated"]:
        console.print(
            Panel(
                f"[red]Authentication Issue:[/red] {auth_check.get('error')}\n\n"
                f"[bold yellow]Remediation:[/bold yellow]\n{auth_check.get('remediation')}",
                title="Colab Auth Status",
            )
        )
        raise typer.Exit(code=1)

    console.print("[green]✔ Google Colab CLI authenticated.[/green]")
    stat = mgr.get_session_status(session_name)
    if stat["stdout"]:
        console.print(Panel(stat["stdout"], title=f"Session Status: {session_name or 'Default'}"))
    else:
        console.print("[yellow]No active session found or VM is currently stopped.[/yellow]")


@app.command()
def recommend(
    model: str = typer.Argument(..., help="Model name or HuggingFace repo ID"),
    gpu: str = typer.Option("T4", "-g", "--gpu", help="Accelerator (T4, L4, A100, H100)"),
):
    """Analyze VRAM requirements and recommend serving backend and quantization."""
    try:
        rec = recommend_model_strategy(model, gpu)
    except ValueError as e:
        console.print(f"[bold red]Recommendation error:[/bold red] {e}")
        raise typer.Exit(code=1)

    table = Table(title=f"Hardware Strategy for {model}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Target Accelerator", gpu)
    table.add_row("Needs Quantization", str(rec["needs_quantization"]))
    table.add_row("Recommended Backend", rec.get("recommended_backend", "vllm"))
    table.add_row(
        "Quantization Formats",
        ", ".join(rec.get("recommended_quantization", [])) or "None (FP16)",
    )
    if rec.get("warning"):
        table.add_row("Advisory", f"[yellow]{rec['warning']}[/yellow]")

    console.print(table)


@app.command()
def up(
    model: str = typer.Argument(..., help="Model identifier (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)"),
    session: str = typer.Option("agent-bridge", "-s", "--session", help="Colab session name"),
    gpu: Optional[str] = typer.Option("T4", "-g", "--gpu", help="GPU accelerator tier (T4, L4, A100, H100)"),
    tpu: Optional[str] = typer.Option(None, "--tpu", help="TPU accelerator tier (v5e1, v6e1)"),
    high_mem: bool = typer.Option(False, "--high-mem", help="Request high-RAM machine shape"),
    backend: str = typer.Option("vllm", "-b", "--backend", help="Serving backend (vllm, ollama, llama_cpp)"),
    tunnel: str = typer.Option("ssh", "-t", "--tunnel", help="Tunnel type (ssh, cloudflare)"),
    port: int = typer.Option(8000, "-p", "--port", help="Local port to bind"),
    timeout: int = typer.Option(600, "--timeout", help="Timeout in seconds for model readiness"),
):
    """Provision a Colab VM, launch model server, and expose OpenAI endpoint."""
    console.print(f"[bold cyan]Deploying {model} on Google Colab ({gpu or tpu or 'CPU'})...[/bold cyan]")
    bridge = ColabBridge(
        session_name=session,
        gpu=gpu,
        tpu=tpu,
        high_mem=high_mem,
        backend=backend,
        tunnel_type=tunnel,
        local_port=port,
    )

    try:
        res = bridge.deploy(model_name=model, wait_for_ready=True, timeout=timeout)
        console.print(
            Panel(
                f"[bold green]✔ Model is READY and live![/bold green]\n\n"
                f"[cyan]Endpoint URL:[/cyan] {res['endpoint_url']}\n"
                f"[cyan]Model Name:[/cyan] {res['model_name']}\n\n"
                f"[bold]To connect coding harnesses:[/bold]\n"
                f'export OPENAI_BASE_URL="{res["endpoint_url"]}"\n'
                f'export OPENAI_API_KEY="colab"\n\n'
                f"[yellow]Aider command:[/yellow] {res['harness']['aider_command']}",
                title="Colab Model Bridge Activated",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Deployment failed:[/bold red] {e}")
        console.print(
            f"[yellow]Note: Session '{session}' may still be active on Colab.\n"
            f"Run 'colab-bridge down -s {session}' to release compute units.[/yellow]"
        )
        raise typer.Exit(code=1)


@app.command()
def down(
    session: str = typer.Option("agent-bridge", "-s", "--session", help="Session to terminate"),
):
    """Terminate the tunnel and release Colab VM compute resources."""
    config = get_config()
    mgr = SessionManager(config)
    console.print(f"[yellow]Terminating session {session}...[/yellow]")
    from colab_agent_bridge.tunnels.ssh_tunnel import SSHTunnel
    SSHTunnel.cleanup_session_tunnels(session)

    res = mgr.stop_session(session)
    if res["success"]:
        console.print(f"[green]✔ Session {session} successfully released and associated tunnels terminated.[/green]")
    else:
        console.print(f"[red]Error stopping session:[/red] {res['stderr']}")
        raise typer.Exit(code=1)


@app.command()
def harness(
    endpoint: str = typer.Option("http://localhost:8000/v1", "--endpoint", "-e", help="Active model endpoint URL"),
    model: str = typer.Option("Qwen/Qwen2.5-Coder-7B-Instruct", "--model", "-m", help="Served model name"),
):
    """Print configuration snippets for Aider, Cline, Continue, and OpenAI SDK."""
    conf = generate_agent_config(endpoint, model)
    console.print(Panel(conf["env_snippet"], title="Environment Variables (.env)"))
    console.print(Panel(conf["aider_command"], title="Aider CLI Launch Command"))
    console.print(Panel(conf["continue_config"], title="Continue Config Block"))


@app.command(name="test")
def test_endpoint(
    endpoint: str = typer.Option("http://localhost:8000/v1", "--endpoint", "-e", help="Model endpoint URL to test"),
    timeout: int = typer.Option(10, "--timeout", "-t", help="Timeout in seconds for model readiness"),
):
    """Test connectivity and health of a deployed model endpoint."""
    console.print(f"[bold cyan]Testing endpoint {endpoint}...[/bold cyan]")
    res = check_health(endpoint, timeout_seconds=timeout)
    if res.get("ready"):
        console.print(
            f"[bold green]✔ Endpoint is healthy and responsive (elapsed: {res.get('elapsed')}s)[/bold green]"
        )
    else:
        console.print(f"[bold red]✖ Health check failed:[/bold red] {res.get('error', 'Timed out')}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
