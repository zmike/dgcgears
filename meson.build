# Copyright © 2022 Collabora Ltd
# Copyright © 2024 Valve Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

project('dgcgears', 'c', 'cpp',
  version : '1.0.0',
  meson_version : '>= 0.60',
  default_options: ['c_std=c11', 'cpp_std=c++17'])

if host_machine.system() == 'darwin'
  add_languages('objc', native : false)
  add_global_arguments('-fobjc-arc', language : 'objc')
endif

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

null_dep = dependency('', required : false)

dep_m = cc.find_library('m', required : false)
dep_winmm = cc.find_library('winmm', required : false)
dep_epoll = dependency('epoll-shim', required : false)
dep_vulkan = dependency('vulkan', required : true)

dep_drm = dependency('libdrm',
  required : get_option('libdrm'),
  disabler : true
)
dep_x11 = dependency('x11, xext',
  required : get_option('x11'),
  disabler : true
)
dep_wayland = dependency('wayland-client, wayland-egl, xkbcommon',
  required : get_option('wayland'),
  disabler : true
)

dep_xcb = dependency('xcb, xkbcommon, xkbcommon-x11',
  required : get_option('x11'),
  disabler : true
)

dep_libdecor = null_dep
if dep_wayland.found()
  dep_wl_scanner = dependency('wayland-scanner', native: true)
  prog_wl_scanner = find_program(dep_wl_scanner.get_variable(pkgconfig : 'wayland_scanner'))
  if dep_wl_scanner.version().version_compare('>= 1.15')
    wl_scanner_arg = 'private-code'
  else
    wl_scanner_arg = 'code'
  endif
  dep_wl_protocols = dependency('wayland-protocols', version : '>= 1.12')
  wayland_xdg_shell_xml = join_paths(
    dep_wl_protocols.get_variable(pkgconfig : 'pkgdatadir'), 'stable',
    'xdg-shell', 'xdg-shell.xml'
  )
  dep_libdecor = dependency('libdecor-0', version : '>= 0.1')
endif

dep_threads = dependency('threads')

dep_dl = null_dep
if not cc.has_function('dlopen')
  dep_dl = cc.find_library('dl', required : host_machine.system() != 'windows')
endif

prog_glslang = find_program('glslangValidator')

if host_machine.system() == 'darwin'
  add_project_arguments('-DGL_SILENCE_DEPRECATION', language: 'c')
endif

if cc.has_function('sincos', prefix: '#define _GNU_SOURCE\n#include <math.h>',
                   dependencies: dep_m)
  add_project_arguments('-DHAVE_SINCOS', language: 'c')
endif

if ['linux', 'cygwin', 'gnu', 'freebsd', 'gnu/kfreebsd', 'haiku', 'android'].contains(host_machine.system())
  add_project_arguments(['-D_GNU_SOURCE', '-D_XOPEN_SOURCE=500'], language: ['c', 'cpp'])
elif host_machine.system() == 'windows'
  add_project_arguments(['-D_USE_MATH_DEFINES'], language: ['c', 'cpp'])
elif host_machine.system() == 'openbsd'
  add_project_arguments(['-D_ISOC11_SOURCE', '-D_XOPEN_SOURCE=500'], language: ['c', 'cpp'])
endif

c_args = []
cpp_args = []
if cc.get_argument_syntax() == 'msvc'
  c_args += '-D_CRT_SECURE_NO_WARNINGS'
  c_args += cc.get_supported_arguments([
    '/wd4100', # 'var': unreferenced formal parameter
    '/wd4244', # conversion from 'type1' to 'type2', possible loss of data
    '/wd4305', # truncation from 'type1' to 'type2'
    '/wd4459', # declaration of 'var' hides global declaration
    '/wd4005', # macro redefinition
  ])
  cpp_args += cc.get_supported_arguments([
    '/wd4305', # truncation from 'type1' to 'type2'
    '/wd4005', # macro redefinition
  ])
else
  c_args += cc.get_supported_arguments([
    '-Werror=implicit-function-declaration',
    '-Werror=missing-prototypes',
    '-Werror=return-type',
    '-Werror=empty-body',
    '-Wimplicit-fallthrough=3',
    '-Wmisleading-indentation',
    '-Wno-missing-field-initializers',
    '-Wno-format-truncation',
    '-fno-math-errno',
    '-fno-trapping-math',
    '-Qunused-arguments',
    # Clang
    '-Wno-microsoft-enum-value',
    '-Wno-unused-function',
  ])

  if host_machine.system() != 'darwin'
    c_args += cc.get_supported_arguments([
      '-Werror=incompatible-pointer-types',
      '-Werror=int-conversion',
    ])
  endif
endif

add_project_arguments(c_args, language: 'c')
add_project_arguments(cpp_args, language: 'cpp')

subdir('src')
