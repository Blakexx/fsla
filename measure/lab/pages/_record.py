# Copyright 2026 Blake Bottum
# SPDX-License-Identifier: Apache-2.0
"""A record's rows by name, for the page generators: a row a page names and the record lacks is refused by name."""


class Record(dict):
    def __missing__(self, name):
        raise SystemExit(
            f"lab page: the record holds no row {name!r} -- the page names a row of the experiment that has not been "
            "run and recorded"
        )
