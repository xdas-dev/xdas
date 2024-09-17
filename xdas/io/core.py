import socket


def get_free_port():
    """
    Find and return a free port on the host machine.

    This function creates a temporary socket, binds it to an available port
    provided by the host, retrieves the port number, and then closes the socket.
    This is useful for finding an available port for network communication.

    Returns
    -------
    int:
        A free port number on the host machine.

    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
