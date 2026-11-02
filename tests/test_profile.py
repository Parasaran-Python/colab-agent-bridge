import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from colab_agent_bridge.cli import app
from colab_agent_bridge.core.profile import ProfileInfo, ProfileManager
from colab_agent_bridge.core.session import SessionManager

runner = CliRunner()


@pytest.fixture
def temp_profile_mgr(tmp_path):
    profiles_dir = tmp_path / "profiles"
    mgr = ProfileManager(profiles_dir=profiles_dir)
    return mgr


def test_profile_create_and_remove(temp_profile_mgr):
    mgr = temp_profile_mgr
    # Create profile
    p_path = mgr.create_profile("work_account")
    assert p_path.exists()
    assert (p_path / ".config" / "colab-cli").exists()

    # Re-creating or creating 'default' should raise ValueError
    with pytest.raises(ValueError, match="reserved"):
        mgr.create_profile("default")

    # Invalid names should raise ValueError
    with pytest.raises(ValueError, match="invalid"):
        mgr.create_profile("invalid/name")
    with pytest.raises(ValueError, match="invalid"):
        mgr.create_profile("../traversal")
    with pytest.raises(ValueError, match="invalid"):
        mgr.create_profile("bad name with spaces")

    # Remove profile
    removed = mgr.remove_profile("work_account")
    assert removed is True
    assert not p_path.exists()

    # Removing non-existent
    assert mgr.remove_profile("non_existent") is False

    # Removing default raises ValueError
    with pytest.raises(ValueError, match="Cannot remove"):
        mgr.remove_profile("default")


def test_profile_switch_active(temp_profile_mgr, tmp_path):
    mgr = temp_profile_mgr
    mgr.create_profile("personal")
    mgr.create_profile("research")

    assert mgr.get_active_profile() == "default"

    mgr.set_active_profile("personal")
    assert mgr.get_active_profile() == "personal"

    mgr.set_active_profile("research")
    assert mgr.get_active_profile() == "research"

    mgr.set_active_profile("default")
    assert mgr.get_active_profile() == "default"

    with pytest.raises(ValueError, match="does not exist"):
        mgr.set_active_profile("unknown_profile")


def test_profile_env_isolation(temp_profile_mgr):
    mgr = temp_profile_mgr
    mgr.create_profile("account_a")

    env_default = mgr.get_profile_env("default")
    assert env_default.get("HOME") == os.environ.get("HOME")

    env_a = mgr.get_profile_env("account_a")
    assert env_a.get("HOME") == str(mgr.profiles_dir / "account_a")


def test_profile_query_usage_parsing(temp_profile_mgr):
    mgr = temp_profile_mgr
    mgr.create_profile("account_b")

    mock_stdout = "Current balance: 24.50 compute units\nUsage rate: 1.96/hr\nActive assignments: 2\n"

    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=0, stdout=mock_stdout)
        usage = mgr.query_usage("account_b")
        assert usage["success"] is True
        assert usage["balance"] == 24.50
        assert usage["rate"] == 1.96
        assert usage["assignments"] == 2


def test_session_manager_uses_profile_env(temp_profile_mgr):
    mgr = temp_profile_mgr
    mgr.create_profile("acc1")

    with patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=0, stdout="Session list ok", stderr="")
        sm = SessionManager(profile="acc1")
        sm.profile_mgr = mgr
        sm.list_sessions()

        mock_sub.assert_called_once()
        _, kwargs = mock_sub.call_args
        assert "env" in kwargs
        assert kwargs["env"].get("HOME") == str(mgr.profiles_dir / "acc1")


def test_cli_profile_list():
    with patch("colab_agent_bridge.cli.ProfileManager") as mock_pm_cls:
        mock_pm = MagicMock()
        mock_pm.get_active_profile.return_value = "default"
        mock_pm.list_profiles.return_value = [
            ProfileInfo(
                name="default",
                path=Path("/home/user"),
                email="default@gmail.com",
                authenticated=True,
                cu_balance=0.0,
                cu_usage_rate=0.0,
                active_sessions=[],
                is_active=True,
            ),
            ProfileInfo(
                name="work",
                path=Path("/tmp/work"),
                email="worker@company.com",
                authenticated=True,
                cu_balance=15.0,
                cu_usage_rate=1.96,
                active_sessions=["work-vm-01"],
                is_active=False,
            ),
        ]
        mock_pm_cls.return_value = mock_pm

        result = runner.invoke(app, ["profile", "list"], env={"COLUMNS": "160"})
        assert result.exit_code == 0
        assert "Google Colab Account Profiles" in result.stdout
        assert "default" in result.stdout
        assert "worker@company.com" in result.stdout
        assert "15.00 CU" in result.stdout
        assert "work-vm-01" in result.stdout


def test_cli_profile_switch():
    with patch("colab_agent_bridge.cli.ProfileManager") as mock_pm_cls:
        mock_pm = MagicMock()
        mock_pm_cls.return_value = mock_pm

        result = runner.invoke(app, ["profile", "switch", "work"])
        assert result.exit_code == 0
        assert "Active profile switched to: work" in result.stdout
        mock_pm.set_active_profile.assert_called_once_with("work")


def test_cli_status_all():
    with patch("colab_agent_bridge.cli.ProfileManager") as mock_pm_cls:
        mock_pm = MagicMock()
        mock_pm.get_active_profile.return_value = "default"
        mock_pm.list_profiles.return_value = [
            ProfileInfo(
                name="default",
                path=Path("/home/user"),
                email="test@gmail.com",
                authenticated=True,
                cu_balance=5.0,
                cu_usage_rate=0.0,
                active_sessions=["vm-test"],
                is_active=True,
            )
        ]
        mock_pm_cls.return_value = mock_pm

        result = runner.invoke(app, ["status", "--all"])
        assert result.exit_code == 0
        assert "Google Colab Account Profiles" in result.stdout
        assert "vm-test" in result.stdout


def test_cli_down_all():
    with (
        patch("colab_agent_bridge.cli.ProfileManager") as mock_pm_cls,
        patch("colab_agent_bridge.cli.SessionManager") as mock_sm_cls,
        patch("colab_agent_bridge.cli.ClusterManager.teardown_all"),
        patch("colab_agent_bridge.tunnels.ssh_tunnel.SSHTunnel.cleanup_session_tunnels") as mock_cleanup,
    ):
        mock_pm = MagicMock()
        mock_pm.list_profiles.return_value = [
            ProfileInfo(
                name="default",
                path=Path("/home/user"),
                active_sessions=["sess-1", "sess-2"],
            ),
            ProfileInfo(
                name="work",
                path=Path("/tmp/work"),
                active_sessions=["sess-3"],
            ),
        ]
        mock_pm_cls.return_value = mock_pm

        mock_sm = MagicMock()
        mock_sm.stop_session.return_value = {"success": True}
        mock_sm_cls.return_value = mock_sm

        result = runner.invoke(app, ["down", "--all"])
        assert result.exit_code == 0
        assert "Scanning and terminating all sessions across all accounts" in result.stdout
        assert "Terminated 3 session(s) across all accounts" in result.stdout
        assert mock_cleanup.call_count == 3


def test_ssh_tunnel_profile_command_and_auth(temp_profile_mgr):
    from colab_agent_bridge.config import BridgeConfig
    from colab_agent_bridge.tunnels.ssh_tunnel import SSHTunnel

    mgr = temp_profile_mgr
    mgr.create_profile("work")
    config = BridgeConfig(colab_cli_path="/mock/colab", auth_strategy="adc")

    # Explicit profile
    tunnel = SSHTunnel(
        session_name="work-sess",
        local_port=8000,
        remote_port=8000,
        config=config,
        profile="work",
        profile_mgr=mgr,
    )
    cmd = " ".join(tunnel.build_ssh_command())
    assert f"HOME={mgr.profiles_dir / 'work'}" in cmd
    assert "--auth=oauth2" in cmd

    # Active profile inheritance
    mgr.set_active_profile("work")
    tunnel_auto = SSHTunnel(
        session_name="work-sess",
        local_port=8000,
        remote_port=8000,
        config=config,
        profile=None,
        profile_mgr=mgr,
    )
    cmd_auto = " ".join(tunnel_auto.build_ssh_command())
    assert f"HOME={mgr.profiles_dir / 'work'}" in cmd_auto
    assert "--auth=oauth2" in cmd_auto


def test_executor_and_session_profile_auth(temp_profile_mgr):
    from colab_agent_bridge.config import BridgeConfig
    from colab_agent_bridge.core.executor import RemoteExecutor

    mgr = temp_profile_mgr
    mgr.create_profile("dev")
    config = BridgeConfig(colab_cli_path="/mock/colab", auth_strategy="adc")

    sm = SessionManager(config=config, profile="dev", profile_mgr=mgr)
    assert sm._get_auth_strategy() == "oauth2"
    assert sm._build_base_cmd() == ["/mock/colab", "--auth=oauth2"]

    rem = RemoteExecutor(config=config, profile="dev", profile_mgr=mgr)
    assert rem._get_auth_strategy() == "oauth2"
    assert rem._build_base_cmd() == ["/mock/colab", "--auth=oauth2"]


def test_cloudflare_tunnel_profile_forwarding():
    from colab_agent_bridge.tunnels.cloudflare import CloudflareTunnel

    tunnel = CloudflareTunnel(session_name="test-cf", profile="work")
    assert tunnel.profile == "work"
    assert tunnel.executor.profile == "work"
