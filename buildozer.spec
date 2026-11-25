[app]
title = FBot
package.name = fbot
package.domain = org.fbot
source.dir = .
source.include_exts = py,json,npz
source.exclude_dirs = .github,.git,bin,.buildozer
version = 1.0
requirements = python3,kivy==2.3.0,numpy
orientation = portrait
fullscreen = 0
android.permissions = INTERNET
android.api = 33
android.minapi = 24
android.archs = arm64-v8a
android.accept_sdk_license = True
android.enable_androidx = True

[buildozer]
log_level = 2
warn_on_root = 1
