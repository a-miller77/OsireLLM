# Tests for functions in app.core.shell_commands

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Assuming your module is app.core.shell_commands
# Adjust if necessary
from app.core import shell_commands

# Define a reusable mock process result
@pytest.fixture
def mock_process_result():
    def _factory(stdout="", stderr="", exit_status=0):
        mock = MagicMock()
        mock.stdout = stdout
        mock.stderr = stderr
        mock.exit_status = exit_status
        return mock
    return _factory

@pytest.mark.asyncio
async def test_run_async_command_success(mocker, mock_process_result):
    """Test run_async_command successful execution."""
    test_command = "echo hello"
    expected_stdout = "hello\n"
    expected_stderr = ""
    expected_exit_status = 0

    # Mock asyncssh.connect
    mock_conn = AsyncMock()
    mock_conn.run = AsyncMock(return_value=mock_process_result(
        stdout=expected_stdout,
        stderr=expected_stderr,
        exit_status=expected_exit_status
    ))
    mock_conn.close = MagicMock() # Mock close method

    mock_connect = mocker.patch("app.core.shell_commands.asyncssh.connect", return_value=mock_conn)

    # Call the function
    stdout, stderr, exit_status = await shell_commands.run_async_command(test_command)

    # Assertions
    mock_connect.assert_awaited_once()
    mock_conn.run.assert_awaited_once_with(test_command, check=False)
    mock_conn.close.assert_called_once()
    assert stdout == expected_stdout
    assert stderr == expected_stderr
    assert exit_status == expected_exit_status

@pytest.mark.asyncio
async def test_run_async_command_failure(mocker, mock_process_result):
    """Test run_async_command when the command fails (non-zero exit status)."""
    test_command = "false"
    expected_stdout = ""
    expected_stderr = "Command failed"
    expected_exit_status = 1

    # Mock asyncssh.connect
    mock_conn = AsyncMock()
    mock_conn.run = AsyncMock(return_value=mock_process_result(
        stdout=expected_stdout,
        stderr=expected_stderr,
        exit_status=expected_exit_status
    ))
    mock_conn.close = MagicMock()

    mock_connect = mocker.patch("app.core.shell_commands.asyncssh.connect", return_value=mock_conn)

    # Call the function
    stdout, stderr, exit_status = await shell_commands.run_async_command(test_command)

    # Assertions
    mock_connect.assert_awaited_once()
    mock_conn.run.assert_awaited_once_with(test_command, check=False)
    mock_conn.close.assert_called_once()
    assert stdout == expected_stdout
    assert stderr == expected_stderr
    assert exit_status == expected_exit_status

@pytest.mark.asyncio
async def test_run_async_command_ssh_error(mocker):
    """Test run_async_command when the SSH connection itself fails."""
    test_command = "echo hello"
    ssh_error = Exception("SSH Connection Failed")

    # Mock asyncssh.connect to raise an error
    mock_connect = mocker.patch("app.core.shell_commands.asyncssh.connect", side_effect=ssh_error)

    # Call the function
    stdout, stderr, exit_status = await shell_commands.run_async_command(test_command)

    # Assertions
    mock_connect.assert_awaited_once()
    assert stdout == ""
    assert str(ssh_error) in stderr # Check if the exception message is in stderr
    assert exit_status == 1 # Should return 1 for failure

@pytest.mark.parametrize(
    "func_to_test, func_args, expected_cmd_fragment",
    [
        (shell_commands.run_async_sbatch, ["/path/to/script.sh"], "sbatch /path/to/script.sh"),
        (shell_commands.run_async_scancel, ["12345"], "scancel 12345"),
        (shell_commands.run_async_scontrol, ["54321"], "scontrol show job 54321"),
        (shell_commands.run_async_sacct, ["98765"], "sacct -j 98765"),
    ]
)
@pytest.mark.asyncio
async def test_async_wrapper_functions(mocker, func_to_test, func_args, expected_cmd_fragment):
    """Test that wrapper functions call run_async_command with the correct command."""
    # Mock the underlying run_async_command
    mock_run_async = mocker.patch(
        "app.core.shell_commands.run_async_command",
        return_value=AsyncMock(return_value=("stdout", "stderr", 0)) # Mock return is async
    )
    mock_run_async.return_value = ("mock_stdout", "mock_stderr", 0) # Actual return of await

    # Call the wrapper function
    await func_to_test(*func_args, host="test_host", user="test_user")

    # Assert run_async_command was called correctly
    mock_run_async.assert_awaited_once()
    call_args, call_kwargs = mock_run_async.call_args
    # Check that the expected fragment is in the command string
    assert expected_cmd_fragment in call_args[0]
    # Check that host and user are passed through
    assert call_args[1] == "test_host"
    assert call_args[2] == "test_user"

# Special case for run_async_scontrol to test dictionary parsing
@pytest.mark.asyncio
async def test_run_async_scontrol_parsing(mocker):
    """Test the dictionary parsing in run_async_scontrol."""
    scontrol_output = "JobId=123 State=RUNNING NodeList=node01 Partition=gpu User=test"
    mock_run_async = mocker.patch(
        "app.core.shell_commands.run_async_command",
        return_value=AsyncMock(return_value=(scontrol_output, "", 0))
    )
    mock_run_async.return_value = (scontrol_output, "", 0)

    job_info, stderr, exit_code = await shell_commands.run_async_scontrol("123")

    assert exit_code == 0
    assert stderr == ""
    assert job_info == {
        "JobId": "123",
        "State": "RUNNING",
        "NodeList": "node01",
        "Partition": "gpu",
        "User": "test"
    }

# Special case for run_async_sacct to test dictionary parsing
@pytest.mark.asyncio
async def test_run_async_sacct_parsing(mocker):
    """Test the dictionary parsing in run_async_sacct."""
    sacct_output = "123|COMPLETED|0:0\n123.batch|COMPLETED|0:0"
    mock_run_async = mocker.patch(
        "app.core.shell_commands.run_async_command",
        return_value=AsyncMock(return_value=(sacct_output, "", 0))
    )
    mock_run_async.return_value = (sacct_output, "", 0)

    job_info, stderr, exit_code = await shell_commands.run_async_sacct("123")

    assert exit_code == 0
    assert stderr == ""
    assert job_info == {
        "JobID": "123",
        "State": "COMPLETED",
        "ExitCode": "0:0"
    }

@pytest.mark.asyncio
async def test_run_async_sacct_parsing_not_found(mocker):
    """Test run_async_sacct when the job ID is not in the output."""
    sacct_output = "456|RUNNING|0:0\n456.batch|RUNNING|0:0" # Different job ID
    mock_run_async = mocker.patch(
        "app.core.shell_commands.run_async_command",
        return_value=AsyncMock(return_value=(sacct_output, "", 0))
    )
    mock_run_async.return_value = (sacct_output, "", 0)

    job_info, stderr, exit_code = await shell_commands.run_async_sacct("123") # Looking for 123

    assert exit_code == 0 # Command succeeded
    assert stderr == ""
    assert job_info is None # Job info should be None

@pytest.mark.asyncio
async def test_setup_ssh_keys_for_local_access_keys_exist(mocker):
    """Test setup_ssh_keys_for_local_access when keys already exist."""
    # Mock os.path.exists to return True for both key and pub key
    mock_exists = mocker.patch("app.core.shell_commands.os.path.exists", return_value=True)
    mock_expanduser = mocker.patch("app.core.shell_commands.os.path.expanduser", return_value="/home/user/.ssh")
    mock_makedirs = mocker.patch("app.core.shell_commands.os.makedirs")
    mock_run = mocker.patch("app.core.shell_commands.sp.run")

    key_name = "id_test"
    key_dir = "~/.ssh"
    expected_key_path = "/home/user/.ssh/id_test"

    success, message = await shell_commands.setup_ssh_keys_for_local_access(key_name, key_dir)

    assert success is True
    assert f"SSH keys already exist at {expected_key_path}" in message
    mock_expanduser.assert_called_once_with(key_dir)
    # Check os.path.exists was called for both private and public keys
    assert mock_exists.call_count == 2
    mock_exists.assert_any_call(expected_key_path)
    mock_exists.assert_any_call(expected_key_path + ".pub")
    mock_makedirs.assert_not_called() # Shouldn't be called if exists
    mock_run.assert_not_called() # Shouldn't generate if exists

@pytest.mark.asyncio
async def test_setup_ssh_keys_for_local_access_generate_keys(mocker):
    """Test setup_ssh_keys_for_local_access when keys need to be generated."""
    # Mock os.path.exists to return False
    mock_exists = mocker.patch("app.core.shell_commands.os.path.exists", return_value=False)
    mock_expanduser = mocker.patch("app.core.shell_commands.os.path.expanduser", return_value="/home/user/.ssh")
    mock_makedirs = mocker.patch("app.core.shell_commands.os.makedirs")
    # Mock subprocess.run to simulate successful key generation
    mock_run = mocker.patch("app.core.shell_commands.sp.run", return_value=MagicMock(returncode=0))
    # Mock adding key to authorized_keys
    mock_add_key = mocker.patch("app.core.shell_commands.add_key_to_authorized_keys", return_value=(True, "Added"))

    key_name = "id_gen"
    key_dir = "~/.ssh"
    expected_key_path = "/home/user/.ssh/id_gen"

    success, message = await shell_commands.setup_ssh_keys_for_local_access(key_name, key_dir)

    assert success is True
    assert "Successfully generated and added SSH key" in message
    mock_expanduser.assert_called_once_with(key_dir)
    mock_exists.assert_called_once_with("/home/user/.ssh") # Check if dir exists
    mock_makedirs.assert_called_once_with("/home/user/.ssh", mode=0o700) # Create dir
    # Check ssh-keygen command was called
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert "ssh-keygen" in args[0]
    assert expected_key_path in args[0]
    mock_add_key.assert_awaited_once_with(expected_key_path + ".pub", key_dir + "/authorized_keys")

@pytest.mark.asyncio
async def test_setup_ssh_keys_for_local_access_generation_fails(mocker):
    """Test setup_ssh_keys_for_local_access when key generation fails."""
    mock_exists = mocker.patch("app.core.shell_commands.os.path.exists", return_value=False)
    mock_expanduser = mocker.patch("app.core.shell_commands.os.path.expanduser", return_value="/home/user/.ssh")
    mock_makedirs = mocker.patch("app.core.shell_commands.os.makedirs")
    # Mock subprocess.run to simulate failed key generation
    mock_run = mocker.patch("app.core.shell_commands.sp.run", return_value=MagicMock(returncode=1, stderr="keygen error"))
    mock_add_key = mocker.patch("app.core.shell_commands.add_key_to_authorized_keys")

    key_name = "id_fail"
    key_dir = "~/.ssh"

    success, message = await shell_commands.setup_ssh_keys_for_local_access(key_name, key_dir)

    assert success is False
    assert "Failed to generate SSH key pair" in message
    assert "keygen error" in message
    mock_run.assert_called_once()
    mock_add_key.assert_not_called() # Should not attempt to add key if generation failed

# Need a test for add_key_to_authorized_keys as well
@pytest.mark.asyncio
async def test_add_key_to_authorized_keys(mocker, tmp_path):
    """Test adding a public key to the authorized_keys file."""
    pub_key_content = "ssh-rsa AAAATESTKEY user@host"
    auth_keys_path = tmp_path / "authorized_keys"

    # Simulate pub key file
    pub_key_path = tmp_path / "id_test.pub"
    with open(pub_key_path, "w") as f:
        f.write(pub_key_content)

    # Simulate authorized_keys already exists but doesn't contain the key
    with open(auth_keys_path, "w") as f:
        f.write("ssh-rsa AAAANOTHERKEY other@host\n")

    # Use real file operations within the temp directory
    success, msg = await shell_commands.add_key_to_authorized_keys(str(pub_key_path), str(auth_keys_path))

    assert success is True
    assert "Key added to authorized_keys" in msg

    # Verify content
    with open(auth_keys_path, "r") as f:
        content = f.read()
        assert "ssh-rsa AAAANOTHERKEY other@host" in content
        assert pub_key_content in content

@pytest.mark.asyncio
async def test_add_key_to_authorized_keys_already_exists(mocker, tmp_path):
    """Test adding a public key when it's already in authorized_keys."""
    pub_key_content = "ssh-rsa AAAATESTKEY user@host"
    auth_keys_path = tmp_path / "authorized_keys"
    pub_key_path = tmp_path / "id_test.pub"
    with open(pub_key_path, "w") as f:
        f.write(pub_key_content)

    # Simulate authorized_keys already containing the key
    with open(auth_keys_path, "w") as f:
        f.write(f"{pub_key_content}\n")
        f.write("ssh-rsa AAAANOTHERKEY other@host\n")

    # Initial content length
    initial_content_len = len(auth_keys_path.read_text())

    success, msg = await shell_commands.add_key_to_authorized_keys(str(pub_key_path), str(auth_keys_path))

    assert success is True
    assert "Key already exists in authorized_keys" in msg

    # Verify content hasn't changed (key not added again)
    assert len(auth_keys_path.read_text()) == initial_content_len