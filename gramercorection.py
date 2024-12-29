import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("prithivida/grammar_error_correcter_v1")
model = AutoModelForSeq2SeqLM.from_pretrained("prithivida/grammar_error_correcter_v1")


def correct_grammar(sentence):
    inputs = tokenizer.encode("gec: " + sentence, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs, max_length=128, num_beams=5, early_stopping=True)
    corrected_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_sentence


interface = gr.Interface(
    fn=correct_grammar,
    inputs=gr.Textbox(lines=2, placeholder="Enter a sentence to correct..."),
    outputs=gr.Textbox(label="Corrected Sentence"),
    title="Grammar Correction",
    description="Enter a sentence, and the model will correct its grammar.",
)


interface.launch()
