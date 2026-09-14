#!/usr/bin/env python3
"""Launch GetContacts with a macOS-compatible multiprocessing method."""

import multiprocessing as mp


def main() -> None:
    # GetContacts passes open file handles to worker processes. On macOS,
    # Python's default "spawn" method cannot pickle those handles, whereas
    # the POSIX "fork" method used here supports the existing implementation.
    mp.set_start_method("fork", force=True)

    from get_dynamic_contacts import main as getcontacts_main

    getcontacts_main()


if __name__ == "__main__":
    main()
