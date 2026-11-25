"""Headless wrapper around fbot7 + fbot7_tools: load once, then reply(text) -> text.
No terminal needed. Used by the Android app (main.py) but also works on a PC:
    python3 engine.py
"""
import io, os, re, sys, contextlib, threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class _Sink(io.TextIOBase):
    """Collects what the bot would print and forwards the reply text as it streams."""
    def __init__(self, on_text=None):
        self.buf, self.on_text, self.tool = "", on_text, None

    def writable(self):
        return True

    def write(self, s):
        self.buf += s
        if self.on_text:
            self.on_text(self._reply_so_far())
        return len(s)

    def flush(self):
        pass

    def _reply_so_far(self):
        out = []
        for line in self.buf.split("\n"):
            t = line.strip()
            if t.startswith("[tool:"):
                self.tool = t
            elif line.startswith("assistant> "):
                out.append(line[len("assistant> "):])
            elif out and not t.startswith("["):      # continuation line of a reply
                out.append(line)
        return "\n".join(out).strip()


class Bot:
    def __init__(self):
        self.lock = threading.Lock()
        with contextlib.redirect_stdout(io.StringIO()):
            import fbot7_tools
            self.fb = fbot7_tools.install()
            fb = self.fb
            fb._validate_presets()
            fb.load_settings()
            if fb.SETTINGS["size"] in fb.SIZE_PRESETS:
                fb.CONFIG.update(fb.SIZE_PRESETS[fb.SETTINGS["size"]])
            self.tok = fb.build_or_load_tokenizer()
            self.gpt = fb.ConfigurableGPT(vocab_size=self.tok.vocab_size, **fb.CONFIG)
            loaded, _meta = fb.load_model(self.gpt)
            if not loaded:
                raise RuntimeError("model.npz not found next to engine.py")
            fb.to_float32(self.gpt)
        self.tool = None

    def new_chat(self):
        self.fb.HISTORY.clear()

    def reply(self, text, on_text=None):
        """Run one chat turn. on_text(partial_reply) is called while it streams."""
        fb = self.fb
        text = " ".join(text.split())
        if not text:
            return ""
        with self.lock:
            sink = _Sink(on_text)
            with contextlib.redirect_stdout(sink):
                new_ids = fb.stream_generate(self.gpt, self.tok, text, obs=fb.CURRENT_OBS["text"])
            self.tool = sink.tool
            decoded = self.tok.decode(new_ids)
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    fb.apply_auto_profile(decoded)
                except OSError:
                    pass
            m = re.search(r"<REPLY>\s*(.*?)\s*</REPLY>", decoded, re.S)
            if m:
                fb.HISTORY.append((text, " ".join(m.group(1).split())))
            return sink._reply_so_far()


if __name__ == "__main__":
    import time
    t = time.time(); b = Bot(); print(f"loaded in {time.time()-t:.1f}s")
    while True:
        try:
            q = input("you> ")
        except EOFError:
            break
        t = time.time(); r = b.reply(q); print("bot>", r, f"({time.time()-t:.1f}s)")
