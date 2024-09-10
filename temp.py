import transformers
import pdb


config = transformers.GPT2Config(
    vocab_size=1,  # doesn't matter -- we don't use the vocab
    n_embd=1024,
    )
    
pdb.set_trace()