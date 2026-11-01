import os
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from colab_agent_bridge.config import get_config
from colab_agent_bridge.core.session import SessionManager
from colab_agent_bridge.core.hardware import recommend_model_strategy
from colab_agent_bridge.core.profile import ProfileManager
from colab_agent_bridge.client.bridge import ColabBridge
from colab_agent_bridge.client.harness import generate_agent_config
from colab_agent_bridge.client.health import check_health

app = typer.Typer(
    name="colab-bridge",
    help="Bridge Google Colab GPU/TPU model serving to local coding agent harnesses.",
    add_completion=False,
)
profile_app = typer.Typer(
    name="profile",
    help="Manage multi-account profiles to isolate and pool multiple Google Colab accounts.",
)
app.add_typer(profile_app, name="profile")
console = Console()


# --- Profile Subcommands ---

@profile_app.command("list")
def profile_list():
    """List all configured Google Colab account profiles and active compute."""
    mgr = ProfileManager()
    profiles = mgr.list_profiles()
    active_profile = mgr.get_active_profile()

    table = Table(title="Google Colab Account Profiles")
    table.add_column("Profile", style="cyan")
    table.add_column("Google Account", style="magenta")
    table.add_column("CU Balance", justify="right", style="green")
    table.add_column("Usage Rate", justify="right", style="yellow")
    table.add_column("Active Sessions", style="blue")
    table.add_column("Status", style="bold")

    total_active_sessions = 0
    for p in profiles:
        is_active_marker = " [bold green](active)[/bold green]" if p.name == active_profile else ""
        name_display = f"{p.name}{is_active_marker}"
        email_display = p.email or "[dim]unknown / unauth[/dim]"
        cu_display = f"{p.cu_balance:.2f} CU"
        rate_display = f"{p.cu_usage_rate:.2f}/hr"
        sessions_display = ", ".join(p.active_sessions) if p.active_sessions else "[dim]None[/dim]"
        status_display = "[green]AUTH[/green]" if p.authenticated else "[red]UNAUTH[/red]"

        total_active_sessions += len(p.active_sessions)
        table.add_row(
            name_display,
            email_display,
            cu_display,
            rate_display,
            sessions_display,
            status_display,
        )

    console.print(table)
    if total_active_sessions > 0:
        console.print(
            f"[bold yellow]⚠️  Active Compute Alert: {total_active_sessions} VM(s) currently active across accounts.[/bold yellow]"
        )


@profile_app.command("add")
def profile_add(
    name: str = typer.Argument(..., help="Name for the new account profile (e.g. personal, work)"),
):
    """Add and authenticate a new Google Colab account profile."""
    mgr = ProfileManager()
    try:
        p_path = mgr.create_profile(name)
        console.print(f"[green]✔ Profile directory initialized:[/green] {p_path}")
        console.print("[cyan]Launching Google authentication flow...[/cyan]")
        import subprocess
        env = mgr.get_profile_env(name)
        res = subprocess.run(
            [mgr.colab_cli_path, "--auth=oauth2", "usage"],
            env=env,
            check=False,
        )
        if res.returncode == 0:
            console.print(f"[bold green]✔ Successfully authenticated profile '{name}'![/bold green]")
        else:
            console.print(f"[yellow]Authentication flow finished with code {res.returncode}.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Failed to create profile:[/bold red] {e}")
        raise typer.Exit(code=1)


@profile_app.command("switch")
def profile_switch(
    name: str = typer.Argument(..., help="Profile name to set as active default"),
):
    """Switch the active default Colab account profile."""
    mgr = ProfileManager()
    try:
        mgr.set_active_profile(name)
        console.print(f"[bold green]✔ Active profile switched to:[/bold green] {name}")
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@profile_app.command("remove")
def profile_remove(
    name: str = typer.Argument(..., help="Profile name to remove"),
):
    """Remove a configured account profile and its credentials."""
    mgr = ProfileManager()
    try:
        removed = mgr.remove_profile(name)
        if removed:
            console.print(f"[green]✔ Profile '{name}' removed successfully.[/green]")
        else:
            console.print(f"[yellow]Profile '{name}' not found.[/yellow]")
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# --- Core Commands ---

@app.command()
def status(
    session_name: Optional[str] = typer.Option(None, "-s", "--session", help="Session name to inspect"),
    all_profiles: bool = typer.Option(False, "-a", "--all", help="Display unified dashboard across all accounts"),
    profile: Optional[str] = typer.Option(None, "-p", "--profile", help="Account profile to inspect"),
):
    """Check active Colab sessions, hardware status, and compute units across accounts."""
    if all_profiles:
        profile_list()
        return

    config = get_config()
    mgr = SessionManager(config, profile=profile)
    auth_check = mgr.verify_auth()

    active_label = profile or ProfileManager().get_active_profile()
    if not auth_check["authenticated"]:
        console.print(
            Panel(
                f"[red]Authentication Issue ({active_label}):[/red] {auth_check.get('error')}\n\n"
                f"[bold yellow]Remediation:[/bold yellow]\n{auth_check.get('remediation')}",
                title=f"Colab Auth Status: {active_label}",
            )
        )
        raise typer.Exit(code=1)

    console.print(f"[green]✔ Google Colab CLI authenticated ({active_label}).[/green]")
    stat = mgr.get_session_status(session_name)
    if stat["stdout"]:
        console.print(Panel(stat["stdout"], title=f"Session Status [{active_label}]: {session_name or 'Default'}"))
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
    profile: Optional[str] = typer.Option(None, "-p", "--profile", help="Account profile to use"),
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
        profile=profile,
    )

    try:
        res = bridge.deploy(model_name=model, wait_for_ready=True, timeout=timeout)
        if bridge.tunnel_type == "ssh":
            if getattr(bridge, "adapter_server", None):
                try:
                    bridge.adapter_server.shutdown()
                    bridge.adapter_server.server_close()
                except Exception:
                    pass
            import subprocess
            import sys
            subprocess.Popen(
                [sys.executable, "-m", "colab_agent_bridge.adapter", str(port), str(port + 1000)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
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
    all_sessions: bool = typer.Option(False, "-a", "--all", help="Terminate all sessions across all accounts"),
    profile: Optional[str] = typer.Option(None, "-p", "--profile", help="Account profile to target"),
):
    """Terminate tunnels and release Colab VM compute resources (supports --all)."""
    from colab_agent_bridge.tunnels.ssh_tunnel import SSHTunnel
    import subprocess

    if all_sessions:
        console.print("[bold yellow]Scanning and terminating all sessions across all accounts...[/bold yellow]")
        try:
            subprocess.run(["pkill", "-f", "colab_agent_bridge.adapter"], capture_output=True)
        except Exception:
            pass

        pm = ProfileManager()
        profiles = pm.list_profiles()
        total_killed = 0

        for p in profiles:
            if not p.active_sessions:
                continue
            sm = SessionManager(profile=p.name)
            for s in p.active_sessions:
                console.print(f"  - Releasing session '{s}' on profile '{p.name}'...")
                SSHTunnel.cleanup_session_tunnels(s)
                res = sm.stop_session(s)
                if res["success"]:
                    total_killed += 1
                else:
                    console.print(f"    [red]Warning stopping {s}: {res['stderr']}[/red]")

        console.print(f"[bold green]✔ Terminated {total_killed} session(s) across all accounts. Compute preserved.[/bold green]")
        return

    config = get_config()
    mgr = SessionManager(config, profile=profile)
    active_label = profile or ProfileManager().get_active_profile()
    console.print(f"[yellow]Terminating session {session} on profile [{active_label}]...[/yellow]")
    SSHTunnel.cleanup_session_tunnels(session)
    try:
        subprocess.run(["pkill", "-f", "colab_agent_bridge.adapter"], capture_output=True)
    except Exception:
        pass

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
