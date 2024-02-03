from pretrain.modeling_duplex import DupMAEForPretraining
from pretrained_args import ModelArguments
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("amazingvince/bert_DupMAE-minipile_1024-vN")
model = DupMAEForPretraining.from_pretrained(
    pretrained_model_name_or_path="amazingvince/bert_DupMAE-minipile_1024-vN",
    model_args=ModelArguments(),
)
m = model.lm
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
output = m(**encoded_input)
output
