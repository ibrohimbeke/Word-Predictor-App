from transformers import pipeline
import gradio as gr

text_generator = pipeline("text-generation", model="distilgpt2")

def predict(prompt):
    result = text_generator(prompt, max_length=50, num_return_sequences=1)
    generated_text = result[0]['generated_text']
    return generated_text

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Enter your prompt:"),
    outputs=gr.Textbox(label="Generated Text:")
)

if __name__ == "__main__":
    iface.launch()
