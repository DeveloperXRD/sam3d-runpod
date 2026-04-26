"""Generate paste-friendly commands to upload handler.py to a RunPod pod."""
import gzip, base64

data = open("handler.py", "rb").read()
compressed = gzip.compress(data, 9)
b64 = base64.b64encode(compressed).decode()

# Split into chunks of ~2200 chars (safe for web terminal paste)
chunk_size = 2200
chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]

print(f"Handler: {len(data)} bytes -> {len(b64)} base64 chars -> {len(chunks)} chunks")
print()
print("=" * 60)
print("PASTE THESE COMMANDS ONE AT A TIME IN THE WEB TERMINAL:")
print("=" * 60)
print()

for i, c in enumerate(chunks):
    op = ">" if i == 0 else ">>"
    print(f'echo -n "{c}" {op} /tmp/h.b64')
    print()

print('python3 -c "import gzip,base64; open(chr(47)+chr(119)+chr(111)+chr(114)+chr(107)+chr(115)+chr(112)+chr(97)+chr(99)+chr(101)+chr(47)+chr(104)+chr(97)+chr(110)+chr(100)+chr(108)+chr(101)+chr(114)+chr(46)+chr(112)+chr(121),chr(119)+chr(98)).write(gzip.decompress(base64.b64decode(open(chr(47)+chr(116)+chr(109)+chr(112)+chr(47)+chr(104)+chr(46)+chr(98)+chr(54)+chr(52)).read())))"')
print()
print("echo 'handler.py created!' && wc -l /workspace/handler.py")
print()
print("python -u /workspace/handler.py")

# Also save to file
with open("upload_cmds.txt", "w") as f:
    for i, c in enumerate(chunks):
        op = ">" if i == 0 else ">>"
        f.write(f'echo -n "{c}" {op} /tmp/h.b64\n\n')
    f.write('python3 -c "import gzip,base64; open(chr(47)+chr(119)+chr(111)+chr(114)+chr(107)+chr(115)+chr(112)+chr(97)+chr(99)+chr(101)+chr(47)+chr(104)+chr(97)+chr(110)+chr(100)+chr(108)+chr(101)+chr(114)+chr(46)+chr(112)+chr(121),chr(119)+chr(98)).write(gzip.decompress(base64.b64decode(open(chr(47)+chr(116)+chr(109)+chr(112)+chr(47)+chr(104)+chr(46)+chr(98)+chr(54)+chr(52)).read())))"\n\n')
    f.write("echo 'handler.py created!' && wc -l /workspace/handler.py\n\n")
    f.write("python -u /workspace/handler.py\n")

print()
print("Commands also saved to upload_cmds.txt")
