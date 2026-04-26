import base64

# Encode boot.py to base64
with open("boot.py", "r") as f:
    boot_code = f.read()

b64 = base64.b64encode(boot_code.encode()).decode()

cmd = f"""bash -c 'echo "{b64}" | python -m base64 -d > /workspace/boot.py && python -u /workspace/boot.py'"""
with open("start_cmd.txt", "w") as f:
    f.write(cmd)

print("Done! Open start_cmd.txt and copy the full line as Docker Start Command.")
print(f"Base64 length: {len(b64)} chars")
