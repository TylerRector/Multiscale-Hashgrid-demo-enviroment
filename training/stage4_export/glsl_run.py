"""stage4_export/glsl_run.py — compile + run the exported GLSL on a headless EGL context.

Proves game/shaders/nrp.frag actually COMPILES and RUNS on a real GL driver and that its
per-fragment nrp() output matches the numpy reference (and hence the torch model). Uses the
SAME nrp.frag source as the game, transformed only to feed the G-buffer via uniforms.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import ctypes
import numpy as np
from OpenGL import EGL, GL


def make_context(w=1, h=1):
    dpy = EGL.eglGetDisplay(EGL.EGL_DEFAULT_DISPLAY)
    maj, minr = ctypes.c_long(), ctypes.c_long()
    assert EGL.eglInitialize(dpy, maj, minr)
    cfg_attr = (EGL.EGLint * 13)(
        EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
        EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
        EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8, EGL.EGL_BLUE_SIZE, 8,
        EGL.EGL_ALPHA_SIZE, 8, EGL.EGL_NONE)
    cfg = (EGL.EGLConfig * 1)()
    ncfg = ctypes.c_long()
    assert EGL.eglChooseConfig(dpy, cfg_attr, cfg, 1, ncfg) and ncfg.value >= 1
    pb = (EGL.EGLint * 5)(EGL.EGL_WIDTH, w, EGL.EGL_HEIGHT, h, EGL.EGL_NONE)
    surf = EGL.eglCreatePbufferSurface(dpy, cfg[0], pb)
    assert EGL.eglBindAPI(EGL.EGL_OPENGL_API)
    # EGL_CONTEXT_MAJOR_VERSION=0x3098, MINOR=0x30FB, PROFILE_MASK=0x30FD, CORE_BIT=0x1
    for attr in (
        (0x3098, 3, 0x30FB, 3, 0x30FD, 0x1, EGL.EGL_NONE),   # 3.3 core
        (0x3098, 3, 0x30FB, 3, EGL.EGL_NONE),                # 3.3 default
        (EGL.EGL_NONE,),                                     # driver default
    ):
        ctx = EGL.eglCreateContext(dpy, cfg[0], EGL.EGL_NO_CONTEXT, (EGL.EGLint * len(attr))(*attr))
        if ctx:
            break
    assert ctx, "eglCreateContext failed"
    assert EGL.eglMakeCurrent(dpy, surf, surf, ctx)
    return dpy, surf, ctx


def _compile(src, kind):
    s = GL.glCreateShader(kind)
    GL.glShaderSource(s, src)
    GL.glCompileShader(s)
    if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
        raise RuntimeError(GL.glGetShaderInfoLog(s).decode())
    return s


def build_test_fragment(frag_src):
    """Transform nrp.frag: feed G-buffer + single light via uniforms, output raw nrp() (no tonemap)."""
    src = frag_src.replace("in vec3 g_pos;", "uniform vec3 g_pos;")
    src = src.replace("in vec3 g_n;", "uniform vec3 g_n;")
    src = src.replace("in vec3 g_rho;", "uniform vec3 g_rho;")
    # replace main() with a raw single-sample evaluator
    head = src.split("void main()")[0]
    test_main = """
uniform vec3 uTestV;
void main() {
    frag = vec4(nrp(g_pos, normalize(g_n), g_rho, uTestV), 1.0);
}
"""
    return head + test_main


def run_parity(frag_path, exp, atlas, samples):
    dpy, surf, ctx = make_context(1, 1)
    with open(frag_path) as f:
        frag_src = f.read()
    vsrc = "#version 330 core\nvoid main(){ vec2 p=vec2((gl_VertexID<<1)&2, gl_VertexID&2);" \
           " gl_Position=vec4(p*2.0-1.0,0,1); }"
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, _compile(vsrc, GL.GL_VERTEX_SHADER))
    GL.glAttachShader(prog, _compile(build_test_fragment(frag_src), GL.GL_FRAGMENT_SHADER))
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        raise RuntimeError(GL.glGetProgramInfoLog(prog).decode())

    # atlas texture (R32F), width W
    W = 4096
    H = (atlas.size + W - 1) // W
    pad = np.zeros(W * H, np.float32); pad[:atlas.size] = atlas
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F, W, H, 0, GL.GL_RED, GL.GL_FLOAT, pad)

    # float FBO 1x1
    fbo = GL.glGenFramebuffers(1); GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    ctex = GL.glGenTextures(1); GL.glBindTexture(GL.GL_TEXTURE_2D, ctex)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA32F, 1, 1, 0, GL.GL_RGBA, GL.GL_FLOAT, None)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, ctex, 0)
    GL.glViewport(0, 0, 1, 1)

    GL.glUseProgram(prog)
    GL.glActiveTexture(GL.GL_TEXTURE0); GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glUniform1i(GL.glGetUniformLocation(prog, "uAtlas"), 0)
    GL.glUniform1i(GL.glGetUniformLocation(prog, "uAtlasW"), W)
    GL.glUniform3f(GL.glGetUniformLocation(prog, "uAABBmin"), *exp["aabb_min"])
    GL.glUniform3f(GL.glGetUniformLocation(prog, "uAABBmax"), *exp["aabb_max"])
    vao = GL.glGenVertexArrays(1); GL.glBindVertexArray(vao)

    outs = []
    for (x, n, rho, v) in samples:
        GL.glUniform3f(GL.glGetUniformLocation(prog, "g_pos"), *x)
        GL.glUniform3f(GL.glGetUniformLocation(prog, "g_n"), *n)
        GL.glUniform3f(GL.glGetUniformLocation(prog, "g_rho"), *rho)
        GL.glUniform3f(GL.glGetUniformLocation(prog, "uTestV"), *v)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        px = GL.glReadPixels(0, 0, 1, 1, GL.GL_RGBA, GL.GL_FLOAT)
        outs.append(np.frombuffer(px, np.float32)[:3].copy())
    return np.array(outs), (GL.glGetString(GL.GL_VERSION).decode(),
                            GL.glGetString(GL.GL_RENDERER).decode())
