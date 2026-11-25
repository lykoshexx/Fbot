import threading

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.core.window import Window
from kivy.graphics import Color, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget

Window.softinput_mode = ""          # we move the layout ourselves (see _kb_poll)
Window.clearcolor = (0.07, 0.07, 0.09, 1)

USER_BG, BOT_BG, SYS_BG = (0.16, 0.40, 0.85, 1), (0.22, 0.22, 0.28, 1), (0.12, 0.12, 0.15, 1)
FONT = sp(16)
_measure = CoreLabel(font_size=FONT)


class Msg(Label):
    """A chat bubble that is only as wide as its text (up to 78% of the screen)."""

    def __init__(self, text, kind, **kw):
        super().__init__(text=text, font_size=FONT, color=(1, 1, 1, 1),
                         size_hint=(None, None), halign="left", valign="middle", **kw)
        with self.canvas.before:
            Color(*{"user": USER_BG, "bot": BOT_BG}.get(kind, SYS_BG))
            self._r = RoundedRectangle(radius=[dp(14)])
        self.bind(text=self._resize, texture_size=self._fit, pos=self._draw, size=self._draw)
        self._resize()

    def _resize(self, *_):
        maxw = Window.width * 0.78
        try:
            nat = max(_measure.get_extents(line or " ")[0] for line in self.text.split("\n"))
            w = min(maxw, nat + dp(40))
        except Exception:
            w = maxw
        self.width = max(w, dp(56))
        self.text_size = (self.width - dp(32), None)

    def _fit(self, *_):
        self.height = self.texture_size[1] + dp(24)

    def _draw(self, *_):
        self._r.pos, self._r.size = self.pos, self.size


class Row(BoxLayout):
    """Puts a bubble on the right (you), left (bot) or centre (system)."""

    def __init__(self, msg, kind, **kw):
        super().__init__(size_hint_y=None, height=msg.height, **kw)
        if kind == "user":
            self.add_widget(Widget())
            self.add_widget(msg)
        elif kind == "bot":
            self.add_widget(msg)
            self.add_widget(Widget())
        else:
            self.add_widget(Widget())
            self.add_widget(msg)
            self.add_widget(Widget())
        msg.bind(height=lambda *_: setattr(self, "height", msg.height))


class BotApp(App):
    title = "FBot"

    def build(self):
        self.bot = None
        self.busy = True
        self._kb = 0
        self.rootbox = BoxLayout(orientation="vertical", padding=dp(8), spacing=dp(8))
        root = self.rootbox

        top = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(6))
        top.add_widget(Label(text="FBot", font_size=sp(20), bold=True))
        new_btn = Button(text="New chat", size_hint_x=None, width=dp(100),
                         background_normal="", background_color=(0.22, 0.22, 0.28, 1))
        new_btn.bind(on_release=lambda *_: self.new_chat())
        top.add_widget(new_btn)
        root.add_widget(top)

        self.scroll = ScrollView(do_scroll_x=False)
        self.chat = GridLayout(cols=1, size_hint_y=None, spacing=dp(12), padding=(0, dp(6)))
        self.chat.bind(minimum_height=self.chat.setter("height"))
        self.scroll.add_widget(self.chat)
        root.add_widget(self.scroll)

        bar = BoxLayout(size_hint_y=None, height=dp(52), spacing=dp(6))
        self.inp = TextInput(multiline=False, hint_text="Type a message...", font_size=FONT,
                             background_color=(0.15, 0.15, 0.19, 1), foreground_color=(1, 1, 1, 1),
                             hint_text_color=(0.6, 0.6, 0.65, 1), cursor_color=(1, 1, 1, 1),
                             padding=(dp(12), dp(15)))
        self.inp.bind(on_text_validate=lambda *_: self.send())
        send_btn = Button(text="Send", size_hint_x=None, width=dp(80),
                          background_normal="", background_color=USER_BG)
        send_btn.bind(on_release=lambda *_: self.send())
        bar.add_widget(self.inp)
        bar.add_widget(send_btn)
        root.add_widget(bar)

        Clock.schedule_interval(self._kb_poll, 0.1)
        self.add("Loading the model...", "sys")
        threading.Thread(target=self._load, daemon=True).start()
        return root

    def _kb_poll(self, dt):
        """Shrink the chat area by the keyboard's height (like a normal chat app)."""
        try:
            h = int(Window.keyboard_height)
        except Exception:
            h = 0
        if h != self._kb:
            self._kb = h
            p = dp(8)
            self.rootbox.padding = [p, p, p, p + h]
            self.to_bottom()

    def add(self, text, kind):
        m = Msg(text, kind)
        self.chat.add_widget(Row(m, kind))
        self.to_bottom()
        return m

    def to_bottom(self):
        Clock.schedule_once(lambda dt: setattr(self.scroll, "scroll_y", 0), 0.1)

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
        msg = self.add("...", "bot")
        threading.Thread(target=self._run, args=(text, msg), daemon=True).start()
        Clock.schedule_once(lambda dt: setattr(self.inp, "focus", True), 0.1)

    def _run(self, text, msg):
        def show(t):
            Clock.schedule_once(lambda dt: self._set(msg, t or "..."))
        try:
            final = self.bot.reply(text, on_text=show) or "(no reply)"
        except Exception as e:
            final = "Error: %r" % (e,)
        Clock.schedule_once(lambda dt: self._done(msg, final))

    def _set(self, msg, t):
        msg.text = t
        self.to_bottom()

    def _done(self, msg, t):
        self._set(msg, t)
        self.busy = False


if __name__ == "__main__":
    BotApp().run()
