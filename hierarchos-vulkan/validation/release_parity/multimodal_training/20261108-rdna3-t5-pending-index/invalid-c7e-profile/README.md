# Invalid historical profile match

`c7e096d.exe` is 308,437,504 bytes with a 51,456,036-byte `.text` section and
runs the full benchmark in 5.92 s. The indexed release binary is 254,914,048
bytes with a 35,780,773-byte `.text` section and runs it in 2.74 s. The two
artifacts are different build profiles, so their 2.50x T5 ratio is retained for
traceability only and is not a speedup claim.
