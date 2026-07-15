from __future__ import annotations

import os
import requests
import gradio as gr

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --paper:#171816; --panel:#20211e; --line:#3a3b36; --ink:#f4f3ee; --muted:#aaa9a1; --acid:#d9ff57; }
body, .gradio-container { background:var(--paper)!important; color:var(--ink)!important; font-family:Manrope,Arial,sans-serif!important; }
.gradio-container { max-width:1280px!important; margin:0 auto!important; padding:24px 30px!important; }
footer { display:none!important; }.prose * { color:var(--ink)!important; }
#brand { border-bottom:1px solid var(--line); padding:0 0 27px!important; margin:0 0 32px!important; }
#brand h1 { font-size:clamp(2.8rem,6vw,5.4rem)!important; line-height:.9!important; letter-spacing:-.085em!important; margin:0!important; font-weight:800!important; }
#brand p { color:var(--muted)!important; font-size:1rem!important; margin:18px 0 0!important; }.label { font:11px 'DM Mono'!important; letter-spacing:.08em!important; text-transform:uppercase!important; color:var(--muted)!important; }
#library, #chat-area { background:var(--panel)!important; border:1px solid var(--line)!important; border-radius:3px!important; padding:22px!important; }
#library h2, #chat-area h2 { letter-spacing:-.06em!important; margin:6px 0 6px!important; }.gr-button { border-radius:2px!important; border:0!important; background:var(--acid)!important; color:#171816!important; font:600 12px 'DM Mono'!important; letter-spacing:.04em!important; text-transform:uppercase!important; min-height:42px!important; }
#reset { background:transparent!important; color:var(--muted)!important; border:1px solid var(--line)!important; }.gr-button:hover { filter:brightness(.93)!important; }
input, textarea, .gr-dropdown { background:#171816!important; border-color:var(--line)!important; color:var(--ink)!important; border-radius:2px!important; box-shadow:none!important; } label span, .gr-file { color:var(--muted)!important; }
.gr-file { border:1px dashed #575951!important; background:#1a1b19!important; border-radius:2px!important; }.gr-chatbot { border:0!important; background:transparent!important; }.message { border-radius:2px!important; }.message.user { background:#30322d!important; }.message.bot { background:#242521!important; }
#sources, #diagnostics, #context { border-top:1px solid var(--line)!important; padding-top:12px!important; }.table-wrap, table { background:#171816!important; color:var(--ink)!important; }
@media(max-width:760px){ .gradio-container{padding:16px!important} #brand h1{font-size:3.4rem!important} }
"""


def ask(message, history, corpus):
    response = requests.post(f"{API_URL}/chat", json={"question": message, "corpus": corpus, "history": (history or [])[-6:]}, timeout=300).json()
    answer = response["answer"]
    rows = [[x["source"], x["module"] or x["chapter"] or x["topic"], x["page_number"], x["text"]] for x in response["citations"]]
    for i in range(16, len(answer), 16):
        yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer[:i]}], rows, response["diagnostics"]
    yield history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}], rows, response["diagnostics"]


def show_chunk(event: gr.SelectData): return event.row_value[3] if event.row_value else "Select a citation to view its complete retrieved chunk."


def upload(files):
    for file in files or []:
        with open(file, "rb") as handle: requests.post(f"{API_URL}/upload", files={"file": handle}, timeout=120).raise_for_status()
    result = requests.post(f"{API_URL}/ingest", timeout=600).json()
    return f"Indexed {result['chunks']} chunks from {result['files']} file(s)."


with gr.Blocks(theme=gr.themes.Base(), css=CSS, title="Course Assistant") as demo:
    gr.Markdown("<div class='label'>Local-first study workspace</div><h1>course / assistant</h1><p>Upload course material. Ask better questions. Get cited answers from your notes.</p>", elem_id="brand")
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Column(elem_id="chat-area"):
                gr.Markdown("<div class='label'>02 / ask your notes</div><h2>Study with evidence.</h2>")
                chatbot = gr.Chatbot(type="messages", height=470, label="Conversation")
                with gr.Row(): prompt = gr.Textbox(placeholder="What would you like to understand?", scale=5, show_label=False); corpus = gr.Dropdown(["all", "content", "questions"], value="all", label="Search")
                with gr.Row(): send = gr.Button("Ask →", variant="primary"); reset = gr.ClearButton([prompt, chatbot], value="Reset", elem_id="reset")
        with gr.Column(scale=2, elem_id="library"):
            gr.Markdown("<div class='label'>01 / build your library</div><h2>Add course material.</h2><p>Upload notes, slides, or questions. Indexing creates your searchable study library.</p>")
            files = gr.File(file_count="multiple", file_types=[".pdf", ".docx", ".pptx", ".md", ".txt"], label="Course files")
            ingest = gr.Button("Upload & index"); status = gr.Markdown()
            sources = gr.Dataframe(headers=["Source", "Module / topic", "Page", "Chunk preview"], interactive=False, label="Sources - select a row to view the chunk", elem_id="sources")
            context = gr.Textbox(label="Retrieved context", lines=6, interactive=False, value="Select a citation to view its complete retrieved chunk.", elem_id="context")
            diagnostics = gr.JSON(label="Retrieval diagnostics", elem_id="diagnostics")
    send.click(ask, [prompt, chatbot, corpus], [chatbot, sources, diagnostics]).then(lambda: "", None, prompt)
    prompt.submit(ask, [prompt, chatbot, corpus], [chatbot, sources, diagnostics]).then(lambda: "", None, prompt)
    sources.select(show_chunk, None, context)
    ingest.click(upload, files, status)

if __name__ == "__main__": demo.launch(server_name="0.0.0.0", server_port=7860)
