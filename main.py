import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

Window.softinput_mode = "resize"          # keep the input box above the keyboard
Window.clearcolor = (0.07, 0.07, 0.09, 1)

USER_BG, BOT_BG, SYS_BG = (0.16, 0.40, 0.85, 1), (0.20, 0.20, 0.25, 1), (0.12, 0.12, 0.15, 1)


class Bubble(BoxLayout):
    def __init__(self, text, bg, **kw):
        super().__init__(size_hint=(0.85, None), padding=dp(10), **kw)
        self.label = Label(text=text, color=(1, 1, 1, 1), font_size=sp(16), halign="left",
                           valign="top", size_hint_y=None, markup=False)
        self.label.bind(width=lambda *_: setattr(self.label, "text_size", (self.label.width, None)),
                        texture_size=self._fit)
        self.add_widget(self.label)
        with self.canvas.before:
            Color(*bg)
            self.rect = RoundedRectangle(radius=[dp(14)])
        self.bind(pos=self._draw, size=self._draw)

    def _fit(self, *_):
        self.label.height = self.label.texture_size[1]
        self.height = self.label.height + dp(20)

    def _draw(self, *_):
        self.rect.pos, self.rect.size = self.pos, self.size


class Row(FloatLayout):
    def __init__(self, text, kind, **kw):
        super().__init__(size_hint_y=None, **kw)
        bg = {"user": USER_BG, "bot": BOT_BG}.get(kind, SYS_BG)
        self.bubble = Bubble(text, bg)
        self.bubble.pos_hint = {"right": 1} if kind == "user" else {"x": 0}
        self.add_widget(self.bubble)
        self.bubble.bind(height=lambda *_: setattr(self, "height", self.bubble.height))
        self.height = self.bubble.height


class BotApp(App):
    title = "FBot"

    def build(self):
        self.bot = None
        self.busy = True
        root = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(6))

        top = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        top.add_widget(Label(text="FBot", font_size=sp(20), bold=True, halign="left"))
        new_btn = Button(text="New chat", size_hint_x=None, width=dp(100),
                         background_normal="", background_color=(0.2, 0.2, 0.25, 1))
        new_btn.bind(on_release=lambda *_: self.new_chat())
        top.add_widget(new_btn)
        root.add_widget(top)

        self.scroll = ScrollView(do_scroll_x=False)
        self.chat = BoxLayout(orientation="vertical", size_hint_y=None, spacing=dp(8), padding=(0, dp(4)))
        self.chat.bind(minimum_height=self.chat.setter("height"))
        self.scroll.add_widget(self.chat)
        root.add_widget(self.scroll)

        bar = BoxLayout(size_hint_y=None, height=dp(48), spacing=dp(6))
        self.inp = TextInput(multiline=False, hint_text="Type a message...", font_size=sp(16),
                             background_color=(0.15, 0.15, 0.19, 1), foreground_color=(1, 1, 1, 1),
                             hint_text_color=(0.6, 0.6, 0.65, 1), cursor_color=(1, 1, 1, 1),
                             padding=(dp(12), dp(14)))
        self.inp.bind(on_text_validate=lambda *_: self.send())
        self.send_btn = Button(text="Send", size_hint_x=None, width=dp(80),
                               background_normal="", background_color=USER_BG)
        self.send_btn.bind(on_release=lambda *_: self.send())
        bar.add_widget(self.inp)
        bar.add_widget(self.send_btn)
        root.add_widget(bar)

        self.add("Loading the model...", "sys")
        threading.Thread(target=self._load, daemon=True).start()
        return root

    # ---- helpers
    def add(self, text, kind):
        row = Row(text, kind)
        self.chat.add_widget(row)
        self.to_bottom()
        return row

    def to_bottom(self):
        Clock.schedule_once(lambda dt: setattr(self.scroll, "scroll_y", 0), 0.05)

    def _load(self):
        try:
            from engine import Bot
            bot = Bot()
            Clock.schedule_once(lambda dt: self._ready(bot))
        except Exception as e:
            Clock.schedule_once(lambda dt: self.add("Could not load the model: %r" % (e,), "sys"))

    def _ready(self, bot):
        self.bot, self.busy = bot, False
        self.add("Ready. Say hi!", "sys")

    def new_chat(self):
        if self.busy or not self.bot:
            return
        self.bot.new_chat()
        self.chat.clear_widgets()
        self.add("New chat started.", "sys")

    def send(self):
        text = self.inp.text.strip()
        if not text or self.busy or not self.bot:
            return
        self.inp.text = ""
        self.busy = True
        self.add(text, "user")
        row = self.add("...", "bot")
        threading.Thread(target=self._run, args=(text, row), daemon=True).start()
        Clock.schedule_once(lambda dt: setattr(self.inp, "focus", True), 0.1)

    def _run(self, text, row):
        def show(t):
            Clock.schedule_once(lambda dt: self._set(row, t or "..."))
        try:
            final = self.bot.reply(text, on_text=show)
            final = final or "(no reply)"
        except Exception as e:
            final = "Error: %r" % (e,)
        Clock.schedule_once(lambda dt: self._done(row, final))

    def _set(self, row, t):
        row.bubble.label.text = t
        self.to_bottom()

    def _done(self, row, t):
        self._set(row, t)
        self.busy = False


if __name__ == "__main__":
    BotApp().run()
