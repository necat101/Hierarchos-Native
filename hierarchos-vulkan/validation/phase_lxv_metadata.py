#!/usr/bin/env python3
"""Minimal reader for the AMDPAL compute metadata note in a captured code object.

Only the subset needed by the Phase LXV gate is implemented: enough MessagePack to
reach `amdpal.pipelines[0].hardware_stages[".cs"]`, whose fields are the driver's
own record of VGPR/SGPR counts, LDS bytes, scratch size and wavefront size.

Anything unexpected raises rather than guessing, so a malformed or forged code
object cannot silently satisfy a resource check.
"""
from __future__ import annotations

import struct


def _unpack(b, i=0):
    c = b[i]
    if c <= 0x7F:
        return c, i + 1
    if 0x80 <= c <= 0x8F or c in (0xDE, 0xDF):
        if c == 0xDE:
            n = struct.unpack_from(">H", b, i + 1)[0]
            i += 3
        elif c == 0xDF:
            n = struct.unpack_from(">I", b, i + 1)[0]
            i += 5
        else:
            n = c & 0x0F
            i += 1
        out = {}
        for _ in range(n):
            key, i = _unpack(b, i)
            value, i = _unpack(b, i)
            out[key] = value
        return out, i
    if 0x90 <= c <= 0x9F or c in (0xDC, 0xDD):
        if c == 0xDC:
            n = struct.unpack_from(">H", b, i + 1)[0]
            i += 3
        elif c == 0xDD:
            n = struct.unpack_from(">I", b, i + 1)[0]
            i += 5
        else:
            n = c & 0x0F
            i += 1
        out = []
        for _ in range(n):
            value, i = _unpack(b, i)
            out.append(value)
        return out, i
    if 0xA0 <= c <= 0xBF:
        n = c & 0x1F
        return b[i + 1:i + 1 + n].decode("utf-8", "replace"), i + 1 + n
    if c == 0xC0:
        return None, i + 1
    if c == 0xC2:
        return False, i + 1
    if c == 0xC3:
        return True, i + 1
    if c in (0xC4, 0xC5):
        n = b[i + 1] if c == 0xC4 else struct.unpack_from(">H", b, i + 1)[0]
        head = 2 if c == 0xC4 else 3
        return b[i + head:i + head + n].decode("utf-8", "replace"), i + head + n
    if c in (0xD9, 0xDA, 0xDB):
        head = {0xD9: 2, 0xDA: 3, 0xDB: 5}[c]
        n = {0xD9: b[i + 1], 0xDA: struct.unpack_from(">H", b, i + 1)[0]}.get(
            c, struct.unpack_from(">I", b, i + 1)[0]
        )
        return b[i + head:i + head + n].decode("utf-8", "replace"), i + head + n
    sizes = {0xCC: (1, ">B"), 0xCD: (2, ">H"), 0xCE: (4, ">I"), 0xCF: (8, ">Q"),
             0xD0: (1, ">b"), 0xD1: (2, ">h"), 0xD2: (4, ">i"), 0xD3: (8, ">q"),
             0xCA: (4, ">f"), 0xCB: (8, ">d")}
    if c in sizes:
        width, fmt = sizes[c]
        return struct.unpack_from(fmt, b, i + 1)[0], i + 1 + width
    raise ValueError("unhandled msgpack 0x%02x at %d" % (c, i))


def compute_stage(path) -> dict:
    """Return the `.cs` hardware stage of the first pipeline recorded in `path`."""
    with open(path, "rb") as handle:
        blob = handle.read()
    shoff = struct.unpack_from("<Q", blob, 40)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 58)
    sections = []
    for index in range(shnum):
        base = shoff + index * shentsize
        name_off, _t, _f, _a, off, size, _l, _i, _al, _e = struct.unpack_from(
            "<IIQQQQIIQQ", blob, base
        )
        sections.append((name_off, off, size))
    names = {}
    stroff = sections[shstrndx][1]
    for name_off, off, size in sections:
        end = blob.index(b"\0", stroff + name_off)
        names[blob[stroff + name_off:end].decode("utf-8", "replace")] = (off, size)
    if ".note" not in names:
        raise ValueError("%s carries no .note section" % path)
    off, size = names[".note"]
    note = blob[off:off + size]
    pos = 0
    while pos + 12 <= len(note):
        namesz, descsz, ntype = struct.unpack_from("<III", note, pos)
        name = note[pos + 12:pos + 12 + namesz].rstrip(b"\0").decode("utf-8", "replace")
        start = pos + 12 + ((namesz + 3) // 4) * 4
        desc = note[start:start + descsz]
        pos = start + ((descsz + 3) // 4) * 4
        if name != "AMDGPU" or ntype != 32:
            continue
        data, _ = _unpack(desc)
        pipeline = data["amdpal.pipelines"][0]
        stages = pipeline.get("hardware_stages") or pipeline.get(".hardware_stages")
        return (stages or {})[".cs"]
    raise ValueError("%s carries no AMDGPU metadata note" % path)
