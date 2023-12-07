import json
from os.path import join
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data-dir", type=str, required=True, help="Dataset root directory")
parser.add_argument("-v", "--vocab-size", type=int, required=True, help="Vocab size")
args = parser.parse_args()


def text_iterator():
    with open(join(args.data_dir, "train.jsonl"), "r") as file:
        for row in file:
            yield json.loads(row)["src"]
            yield json.loads(row)["trg"]
    # for i in range(0, len(dataset), 1000):
    #     yield dataset[i: i + 1000]["text"]


special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
trainer = trainers.WordPieceTrainer(vocab_size=args.vocab_size,
                                    special_tokens=special_tokens)
tokenizer.train_from_iterator(text_iterator(), trainer=trainer)

print("EXAMPLES:")
it = text_iterator()
for i in range(5):
    ex = next(it)
    print(ex)
    print(tokenizer.encode(ex).tokens)
    print()

tokenizer.save(join(args.data_dir, f"tokenizer_{args.vocab_size}.json"))

# new_tokenizer = Tokenizer.from_file("tokenizer.json")
# wrapped_tokenizer = PreTrainedTokenizerFast(
#     tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     cls_token="[CLS]",
#     sep_token="[SEP]",
#     mask_token="[MASK]",
# )
# BertTokenizerFast(tokenizer_object=tokenizer)

