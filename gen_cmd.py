b = open("boot.b64").read().strip()
cmd = f"""bash -c 'echo "{b}" | python -m base64 -d > /workspace/boot.py && python -u /workspace/boot.py'"""
open("start_cmd.txt", "w").write(cmd)
print("Done! Open start_cmd.txt and copy the full line as Docker Start Command.")
