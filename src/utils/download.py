import config as c
from dotenv import load_dotenv

load_dotenv()
c.os.environ["HF_TOKEN"] = c.os.environ.get("HF_TOKEN")
c.os.environ["USE_TORCH"] = "0"

from datasets import load_dataset

print("Connecting to Hugging Face dataset stream...")
dataset = load_dataset(
    "allenai/peS2o", split="train", streaming=True, trust_remote_code=True
)

bytes_written = 0
line_count = 0
print(f"Writing to file {c.DATA[0]}...")

with open(c.DATA[0], "w", encoding="utf-8") as f:
    for r in dataset:
        raw = r.get("text", "").strip()

        if raw:
            text = " ".join(raw.splitlines())
            to_write = text + "\n"
            f.write(to_write)

            bytes_written += len(to_write.encode("utf-8"))
            line_count += 1

            if line_count % 15000 == 0:
                mb_written = bytes_written / (1024 * 1024)
                print(f"Written: {mb_written:.2f} MB ({line_count} documents)")

            if bytes_written >= c.DATA[1]:
                print("Max bytes reached. Disconnecting stream")
                break

print(f"Done.\nFile saved as {c.DATA[0]}, {bytes_written / (1024 * 1024):.2f} MB")
