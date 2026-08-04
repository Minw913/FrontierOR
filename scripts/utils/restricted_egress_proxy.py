"""Compatibility entrypoint for the WLS egress proxy."""

from frontieror.infra.wls_proxy import *  # noqa: F401,F403
from frontieror.infra.wls_proxy import main


if __name__ == "__main__":
    main()
