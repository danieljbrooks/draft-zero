"""After a killed `dzgorge play`: keep a chunk's complete games only.

    python gorge/recipe/trim.py CHUNK.jsonl CHUNK.gz  -> prints k, the number of complete games kept

gorge writes games in order through a buffer, so a killed run's games file can end mid-line, while its corpus (one
gzip member per game, written unbuffered) can hold a game or two more. Keeps the first k complete lines and only the
corpus members of those k games.
"""
import re
import sys
import zlib

jl, gz = sys.argv[1], sys.argv[2]
lines = open(jl, "rb").read().split(b"\n")
done = [l for l in lines[:-1]]                     # complete lines end with "\n"; the last piece is partial or empty
open(jl, "wb").write(b"".join(l + b"\n" for l in done))
# the corpus names a game p<pair>g<leg>; "game" in the games file restarts at 0 in a -first-pair run, "pair" doesn't
games = {(int(re.search(rb'"pair":(\d+)', l).group(1)), int(re.search(rb'"leg":(\d+)', l).group(1))) for l in done}
data = open(gz, "rb").read()
keep, pos = [], 0
while pos < len(data):
    d = zlib.decompressobj(31)
    try:
        out = d.decompress(data[pos:])
    except zlib.error:
        break
    if not d.eof:
        break
    end = len(data) - len(d.unused_data)
    m = re.search(rb'"game_id":"p(\d+)g(\d+)"', out)
    if m and (int(m.group(1)), int(m.group(2))) in games:
        keep.append(data[pos:end])
    pos = end
open(gz, "wb").write(b"".join(keep))
print(len(done))
