from wrapper import ADIFF
import os

if __name__ == "__main__":
    adiff = ADIFF(
                    config_path="base.yaml",
                    model_path = f'adiff_base.ckpt'
                    )

    examples = [
        ["<path1>", "<path2>", "explain the difference between the two audio in detail"],
    ]

    response = adiff.generate(examples=examples, max_len=300, temperature=1.0)
    print(f"\noutput: {response[0]}")