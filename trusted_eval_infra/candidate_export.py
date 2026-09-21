"""Export one bounded regular output from a stopped candidate's tmpfs volume."""

import os
import stat
import sys


def main():
    limit = int(sys.argv[1])
    fd = os.open("/output/solution.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > limit:
            raise ValueError("output must be a bounded, single-link regular file")
        total = 0
        while chunk := os.read(fd, min(1024 * 1024, limit - total + 1)):
            total += len(chunk)
            if total > limit:
                raise ValueError("output exceeds size limit")
            sys.stdout.buffer.write(chunk)
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
