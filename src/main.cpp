// Multiscale hashgrid lighting demo. SDL2 + OpenGL 3.3.
// CPU geometry/reflections produce a G-buffer; the GPU evaluates the trained
// hashgrid and MLP for diffuse lighting. WASD/mouse controls the camera.
#include <SDL2/SDL.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "hud_font.h"

// ---------------------------------------------------------------- GL function loader
// Core-profile GL (shaders/VAO/FBO/queries/multitexture) is not in libGL's link surface on Linux;
// load what we use through SDL_GL_GetProcAddress. Base GL 1.1 calls come straight from <GL/gl.h>.
#define GLFUNCS \
  X(PFNGLCREATESHADERPROC, glCreateShader) X(PFNGLSHADERSOURCEPROC, glShaderSource) \
  X(PFNGLCOMPILESHADERPROC, glCompileShader) X(PFNGLGETSHADERIVPROC, glGetShaderiv) \
  X(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog) X(PFNGLCREATEPROGRAMPROC, glCreateProgram) \
  X(PFNGLATTACHSHADERPROC, glAttachShader) X(PFNGLLINKPROGRAMPROC, glLinkProgram) \
  X(PFNGLGETPROGRAMIVPROC, glGetProgramiv) X(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog) \
  X(PFNGLUSEPROGRAMPROC, glUseProgram) X(PFNGLDELETESHADERPROC, glDeleteShader) \
  X(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation) X(PFNGLUNIFORM1IPROC, glUniform1i) \
  X(PFNGLUNIFORM1FPROC, glUniform1f) X(PFNGLUNIFORM2FPROC, glUniform2f) \
  X(PFNGLUNIFORM3FPROC, glUniform3f) X(PFNGLUNIFORM3FVPROC, glUniform3fv) \
  X(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays) \
  X(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray) X(PFNGLGENQUERIESPROC, glGenQueries) \
  X(PFNGLBEGINQUERYPROC, glBeginQuery) X(PFNGLENDQUERYPROC, glEndQuery) \
  X(PFNGLGETQUERYOBJECTUI64VPROC, glGetQueryObjectui64v) \
  X(PFNGLGETQUERYOBJECTIVPROC, glGetQueryObjectiv) \
  X(PFNGLGENBUFFERSPROC, glGenBuffers) X(PFNGLBINDBUFFERPROC, glBindBuffer) \
  X(PFNGLBUFFERDATAPROC, glBufferData) X(PFNGLBINDBUFFERBASEPROC, glBindBufferBase) \
  X(PFNGLGETUNIFORMBLOCKINDEXPROC, glGetUniformBlockIndex) \
  X(PFNGLUNIFORMBLOCKBINDINGPROC, glUniformBlockBinding) \
  X(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers) X(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer) \
  X(PFNGLFRAMEBUFFERTEXTURE2DPROC, glFramebufferTexture2D)

#define X(type, name) static type name = nullptr;
GLFUNCS
#undef X
static void load_gl() {
#define X(type, name) name = (type)SDL_GL_GetProcAddress(#name);
  GLFUNCS
#undef X
}

// ---------------------------------------------------------------- tiny math
struct V3 {
  float x, y, z;
  V3 operator+(V3 b) const { return {x + b.x, y + b.y, z + b.z}; }
  V3 operator-(V3 b) const { return {x - b.x, y - b.y, z - b.z}; }
  V3 operator*(float s) const { return {x * s, y * s, z * s}; }
};
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static V3 norm(V3 a) { float l = std::sqrt(dot(a, a)); return l > 1e-9f ? a * (1.0f / l) : a; }

// ---------------------------------------------------------------- level (mirrors level/level.toml)
static const std::vector<std::string> MAP = {
  "############", "#..........#", "#..####....#", "#..#.......#", "#..#..##...#",
  "#.....##...#", "#........#.#", "#..........#", "############"};
static const float CELL = 1.0f, EYE = 0.5f, WALLH = 1.0f;
static const float ALB[3][3] = {{0.72f, 0.68f, 0.60f}, {0.40f, 0.42f, 0.46f}, {0.55f, 0.52f, 0.50f}};
static int ROWS() { return (int)MAP.size(); }
static int COLS() { return (int)MAP[0].size(); }
static bool is_wall(int r, int c) {
  if (r < 0 || c < 0 || r >= ROWS() || c >= COLS()) return true;
  return MAP[r][c] == '#';
}

// conductor objects (level.toml [[objects]]): copper orb (sphere) + silver pillar (box)
struct Obj { int type; V3 a, b; float rad; V3 f0; };   // type 0=sphere(a=center,rad), 1=box(a=min,b=max)
static const std::vector<Obj> OBJS = {
  {0, {5.5f, 0.5f, 6.5f}, {0, 0, 0}, 0.45f, {0.95f, 0.64f, 0.54f}},          // orb, copper
  {1, {9.0f, 0.0f, 4.0f}, {9.4f, 1.0f, 4.4f}, 0.0f, {0.91f, 0.92f, 0.92f}},  // pillar, silver
};

struct Light { V3 pos; V3 E; float radius; };

// ---------------------------------------------------------------- geometry intersections
static const float TFAR = 1e30f;

// DDA the tile world (walls) + floor/ceiling planes. Returns hit t, pos, normal, mat(0/1/2).
static bool cast_world(V3 o, V3 d, float& tOut, V3& pos, V3& n, int& mat) {
  float tFloor = d.y < -1e-4f ? (0.0f - o.y) / d.y : TFAR;
  float tCeil = d.y > 1e-4f ? (WALLH - o.y) / d.y : TFAR;
  float px = o.x, pz = o.z;
  int mapX = (int)std::floor(px / CELL), mapZ = (int)std::floor(pz / CELL);
  float ddx = d.x == 0 ? TFAR : std::fabs(CELL / d.x);
  float ddz = d.z == 0 ? TFAR : std::fabs(CELL / d.z);
  int stepX = d.x < 0 ? -1 : 1, stepZ = d.z < 0 ? -1 : 1;
  float sdx = d.x < 0 ? (px / CELL - mapX) * ddx : (mapX + 1 - px / CELL) * ddx;
  float sdz = d.z < 0 ? (pz / CELL - mapZ) * ddz : (mapZ + 1 - pz / CELL) * ddz;
  float tWall = TFAR; V3 wn{0, 0, 0};
  for (int i = 0; i < 96; i++) {
    bool xside = sdx < sdz;
    if (xside) { sdx += ddx; mapX += stepX; } else { sdz += ddz; mapZ += stepZ; }
    if (is_wall(mapZ, mapX)) {
      tWall = xside ? (sdx - ddx) : (sdz - ddz);
      wn = xside ? V3{(float)-stepX, 0, 0} : V3{0, 0, (float)-stepZ};
      break;
    }
  }
  float t = std::fmin(tWall, std::fmin(tFloor, tCeil));
  if (t > 1e29f) return false;
  tOut = t; pos = o + d * t;
  if (t == tWall) { n = wn; mat = 0; }
  else if (t == tFloor) { n = {0, 1, 0}; mat = 1; }
  else { n = {0, -1, 0}; mat = 2; }
  return true;
}

static bool hit_sphere(V3 o, V3 d, V3 c, float r, float& t, V3& n) {
  V3 oc = o - c; float b = dot(oc, d); float cc = dot(oc, oc) - r * r;
  float disc = b * b - cc; if (disc < 0) return false;
  float sq = std::sqrt(disc); float t0 = -b - sq, t1 = -b + sq;
  float tt = t0 > 1e-4f ? t0 : (t1 > 1e-4f ? t1 : TFAR);
  if (tt >= TFAR) return false;
  t = tt; n = norm((o + d * tt) - c); return true;
}
static bool hit_box(V3 o, V3 d, V3 mn, V3 mx, float& t, V3& n) {
  float ix = 1.0f / (std::fabs(d.x) < 1e-9f ? 1e-9f : d.x);
  float iy = 1.0f / (std::fabs(d.y) < 1e-9f ? 1e-9f : d.y);
  float iz = 1.0f / (std::fabs(d.z) < 1e-9f ? 1e-9f : d.z);
  float t0x = (mn.x - o.x) * ix, t1x = (mx.x - o.x) * ix;
  float t0y = (mn.y - o.y) * iy, t1y = (mx.y - o.y) * iy;
  float t0z = (mn.z - o.z) * iz, t1z = (mx.z - o.z) * iz;
  float smx = std::fmin(t0x, t1x), sMx = std::fmax(t0x, t1x);
  float smy = std::fmin(t0y, t1y), sMy = std::fmax(t0y, t1y);
  float smz = std::fmin(t0z, t1z), sMz = std::fmax(t0z, t1z);
  float tnear = std::fmax(smx, std::fmax(smy, smz));
  float tfar = std::fmin(sMx, std::fmin(sMy, sMz));
  if (tfar < std::fmax(tnear, 1e-4f)) return false;
  float tt = tnear > 1e-4f ? tnear : tfar; if (tt >= TFAR) return false;
  t = tt;
  // normal = axis of the max entry slab (matches intersectors.intersect_box)
  int ax = (smx >= smy && smx >= smz) ? 0 : (smy >= smz ? 1 : 2);
  n = {0, 0, 0};
  float dc = ax == 0 ? d.x : ax == 1 ? d.y : d.z;
  float s = dc > 0 ? -1.0f : 1.0f;
  if (ax == 0) n.x = s; else if (ax == 1) n.y = s; else n.z = s;
  return true;
}
static bool nearest_obj(V3 o, V3 d, float& t, V3& n, V3& f0) {
  t = TFAR; bool got = false;
  for (const Obj& ob : OBJS) {
    float tt; V3 nn; bool h;
    if (ob.type == 0) h = hit_sphere(o, d, ob.a, ob.rad, tt, nn);
    else h = hit_box(o, d, ob.a, ob.b, tt, nn);
    if (h && tt < t) { t = tt; n = nn; f0 = ob.f0; got = true; }
  }
  return got;
}
static V3 schlick(V3 f0, float cosT) {
  float c = std::pow(std::fmax(0.0f, 1.0f - std::fmin(std::fmax(cosT, 0.0f), 1.0f)), 5.0f);
  return {f0.x + (1 - f0.x) * c, f0.y + (1 - f0.y) * c, f0.z + (1 - f0.z) * c};
}
static V3 reflect(V3 d, V3 n) { return d - n * (2.0f * dot(d, n)); }

// ---------------------------------------------------------------- .npy + export.json readers
static uint16_t f32_to_f16bits(float);  // fwd (unused)
static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits;
  if (exp == 0) {
    if (man == 0) bits = sign;
    else { int e = -1; do { e++; man <<= 1; } while (!(man & 0x400)); man &= 0x3FF;
           bits = sign | ((127 - 15 - e) << 23) | (man << 13); }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f; std::memcpy(&f, &bits, 4); return f;
}
static std::vector<float> load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return {}; }
  char magic[6]; f.read(magic, 6);
  uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);
  uint32_t hlen; if (major == 1) { uint16_t h16; f.read((char*)&h16, 2); hlen = h16; }
  else { f.read((char*)&hlen, 4); }
  std::string header(hlen, '\0'); f.read(&header[0], hlen);
  bool f16 = header.find("<f2") != std::string::npos || header.find("|f2") != std::string::npos;
  bool f32 = header.find("<f4") != std::string::npos;
  // shape
  size_t sp = header.find("'shape':"); size_t lp = header.find('(', sp), rp = header.find(')', lp);
  std::string shp = header.substr(lp + 1, rp - lp - 1);
  size_t n = 1; { std::stringstream ss(shp); std::string tok; n = 0; size_t prod = 1; bool any = false;
    while (std::getline(ss, tok, ',')) { tok.erase(std::remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
      if (tok.empty()) continue; prod *= std::stoul(tok); any = true; } n = any ? prod : 0; }
  std::vector<float> out(n);
  if (f16) { std::vector<uint16_t> raw(n); f.read((char*)raw.data(), n * 2);
             for (size_t i = 0; i < n; i++) out[i] = f16_to_f32(raw[i]); }
  else if (f32) { f.read((char*)out.data(), n * 4); }
  else { fprintf(stderr, "npy dtype not f2/f4 in %s\n", path.c_str()); }
  return out;
}
static bool json_vec3(const std::string& s, const std::string& key, V3& out) {
  size_t k = s.find("\"" + key + "\""); if (k == std::string::npos) return false;
  size_t lb = s.find('[', k); if (lb == std::string::npos) return false;
  size_t rb = s.find(']', lb);
  std::string body = s.substr(lb + 1, rb - lb - 1);
  for (char& ch : body) if (ch == ',') ch = ' ';
  std::stringstream ss(body); ss >> out.x >> out.y >> out.z; return true;
}

// ---------------------------------------------------------------- deferred shader assembly
static std::string slurp(const std::string& p) { std::ifstream f(p); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }
static std::string strip(std::string s, const std::string& what) {
  size_t p; while ((p = s.find(what)) != std::string::npos) s.erase(p, what.size()); return s;
}
static void replace_all(std::string& s, const std::string& from, const std::string& to) {
  size_t p = 0; while ((p = s.find(from, p)) != std::string::npos) { s.replace(p, from.size(), to); p += to.size(); }
}
// nrp.frag -> deferred: strip the vertex `in` decls, replace main() with a G-buffer sampler that
// composites emit + tint * Σ nrp() (single-count E) and applies the shared Reinhard+gamma tonemap.
//
// C3 perf lever: the MLP weights (indices >= hg_size) are read every pixel, so per-fragment
// texelFetch of the atlas serializes catastrophically. Move them into a std140 UBO (cached uniform
// memory) via `watlas()`; the hashgrid (sparse, 32 fetches/pixel) stays a texture. `n_mlp` floats,
// packed 4/vec4.
static std::string make_deferred(const std::string& frag_src, int hg_size, int n_mlp) {
  std::string src = frag_src;
  src = strip(src, "in vec3 g_pos;");
  src = strip(src, "in vec3 g_n;");
  src = strip(src, "in vec3 g_rho;");
  int nwvec = (n_mlp + 3) / 4;
  char ubo[512];
  snprintf(ubo, sizeof ubo,
    "const int NWVEC = %d;\nlayout(std140) uniform Weights { vec4 uW[NWVEC]; };\n"
    "float watlas(int i){ int j = i - %d; return uW[j>>2][j&3]; }\n", nwvec, hg_size);
  // insert the UBO accessor just before hash() (after the texture atlas() accessor)
  size_t hp = src.find("uint hash(");
  if (hp != std::string::npos) src.insert(hp, ubo);
  // route the two dense MLP weight fetches through the UBO
  replace_all(src, "atlas(boff+o)", "watlas(boff+o)");
  replace_all(src, "atlas(woff + o*din + i)", "watlas(woff + o*din + i)");
  size_t mp = src.find("void main()");
  std::string head = src.substr(0, mp);
  head +=
    "uniform sampler2D uGPos; uniform sampler2D uGN; uniform sampler2D uGRho;\n"
    "uniform sampler2D uGTint; uniform sampler2D uGEmit; uniform vec2 uRes;\n"
    "uniform float uExposure;\n"
    "void main() {\n"
    "  vec2 uv = gl_FragCoord.xy / uRes;\n"
    "  vec4 P = texture(uGPos, uv);\n"
    "  vec3 emit = texture(uGEmit, uv).rgb;\n"
    "  vec3 tint = texture(uGTint, uv).rgb;\n"
    "  vec3 c = emit;\n"
    "  if (P.a > 0.5) {\n"
    "    vec3 gp = P.rgb; vec3 gn = normalize(texture(uGN, uv).rgb); vec3 gr = texture(uGRho, uv).rgb;\n"
    "    vec3 s = vec3(0.0);\n"
    "    for (int i=0;i<uNLights && i<4; ++i) s += nrp(gp, gn, gr, uLightPos[i]);\n"  // single-count E (D1)
    "    c += tint * s;\n"
    "  }\n"
    "  c *= uExposure;\n"
    "  c = pow(c/(1.0+c), vec3(1.0/2.2));\n"   // shared exposure+Reinhard+gamma (C0 unified transform)
    "  frag = vec4(c, 1.0);\n"
    "}\n";
  return head;
}
static GLuint compile(const char* src, GLenum kind) {
  GLuint s = glCreateShader(kind); glShaderSource(s, 1, &src, nullptr); glCompileShader(s);
  GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) { char log[4096]; glGetShaderInfoLog(s, 4096, nullptr, log); fprintf(stderr, "shader: %s\n", log); }
  return s;
}

// ---------------------------------------------------------------- HUD overlay (8x13 burned-in font)
static void hud_char(std::vector<uint8_t>& buf, int W, int H, int x0, int y0, char ch, uint8_t r, uint8_t g, uint8_t b) {
  if (ch < 32 || ch > 126) return;
  const unsigned char* gl = HUD_FONT[ch - 32];
  for (int ry = 0; ry < HUD_FONT_H; ry++)
    for (int rx = 0; rx < HUD_FONT_W; rx++)
      if (gl[ry] & (1 << rx)) {
        int x = x0 + rx, y = y0 + ry; if (x < 0 || y < 0 || x >= W || y >= H) continue;
        int i = (y * W + x) * 4; buf[i] = r; buf[i + 1] = g; buf[i + 2] = b; buf[i + 3] = 255;
      }
}
static void hud_text(std::vector<uint8_t>& buf, int W, int H, int x, int y, const std::string& t,
                     uint8_t r = 235, uint8_t g = 240, uint8_t b = 120) {
  int cx = x;
  for (char c : t) { hud_char(buf, W, H, cx, y, c, r, g, b); cx += HUD_FONT_W; }
}

// ---------------------------------------------------------------- camera / state
struct Cam { V3 pos; float yaw, pitch; bool fly; bool torch_follow; V3 torch_pos; };
static V3 cam_fwd(const Cam& c) {
  float cp = std::cos(c.pitch);
  return norm({cp * std::cos(c.yaw), std::sin(c.pitch), cp * std::sin(c.yaw)});
}

int main(int argc, char** argv) {
  int RES = 256; bool headless = false;
  int max_frames = 0, light_cap = 1; bool start_fullscreen = false;
  std::string save_dir, model_dir = "model";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--model") model_dir = next();
    else if (a == "--res") RES = std::stoi(next());
    else if (a == "--fullscreen") start_fullscreen = true;
    else if (a == "--headless") headless = true;
    else if (a == "--frames") max_frames = std::stoi(next());
    else if (a == "--lights") light_cap = std::stoi(next());
    else if (a == "--save") save_dir = next();
  }

  // Load the trained hashgrid/MLP atlas and its matching shader.
  std::string base = model_dir + "/";
  std::string exp_json = slurp(base + "export.json");
  if (exp_json.empty()) { fprintf(stderr, "missing %sexport.json (run from the demo root or pass --model DIR)\n", base.c_str()); return 1; }
  V3 aabbmin{0, 0, 0}, aabbmax{12, 1, 9};
  json_vec3(exp_json, "aabb_min", aabbmin); json_vec3(exp_json, "aabb_max", aabbmax);
  std::vector<float> atlas = load_npy(base + "atlas.npy");
  std::string frag_src = slurp(base + "nrp.frag");
  if (atlas.empty() || frag_src.empty()) { fprintf(stderr, "missing atlas.npy or nrp.frag under %s\n", base.c_str()); return 1; }

  // ---- SDL + GL context
  // On a Wayland session SDL2's Wayland backend can grab the pointer (relative mouse works) while the
  // window never receives KEYBOARD focus, so SDL_GetKeyboardState reads all-up and WASD/F/L do nothing.
  // Force the X11/XWayland backend (DISPLAY is set), which focuses reliably. Override via SDL_VIDEODRIVER.
  SDL_setenv("SDL_VIDEODRIVER", "x11", 0);   // 0 = do not overwrite a user-set value
  if (SDL_Init(SDL_INIT_VIDEO) != 0) { fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); return 1; }
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  Uint32 wflags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                  (headless ? SDL_WINDOW_HIDDEN : SDL_WINDOW_SHOWN) |
                  (start_fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
  SDL_Window* win = SDL_CreateWindow("Multiscale Hashgrid Demo — live free camera", SDL_WINDOWPOS_CENTERED,
                                     SDL_WINDOWPOS_CENTERED, RES, RES, wflags);
  if (!win) { fprintf(stderr, "SDL_CreateWindow: %s\n", SDL_GetError()); return 1; }
  SDL_GLContext ctx = SDL_GL_CreateContext(win);
  if (!ctx) { fprintf(stderr, "SDL_GL_CreateContext: %s\n", SDL_GetError()); return 1; }
  load_gl();
  SDL_GL_SetSwapInterval(0);  // uncapped; FPS is measured, not vsync-limited
  if (!headless) { SDL_RaiseWindow(win); SDL_SetWindowInputFocus(win); }  // ensure keyboard focus
  bool is_fullscreen = start_fullscreen;

  std::string renderer = (const char*)glGetString(GL_RENDERER);
  bool software_gl = renderer.find("llvmpipe") != std::string::npos ||
                     renderer.find("softpipe") != std::string::npos;
  printf("GL_RENDERER: %s%s\n", renderer.c_str(), software_gl ? "  [SOFTWARE-GL: perf numbers not a method measurement]" : "");

  // ---- build the deferred NRP program (C3: MLP weights -> UBO; hashgrid stays a texture)
  int hg_size = 16384; { size_t k = exp_json.find("\"table_len\"");   // hashgrid float count
    if (k != std::string::npos) { size_t c = exp_json.find(':', k); hg_size = std::stoi(exp_json.substr(c + 1)); } }
  int n_mlp = (int)atlas.size() - hg_size; if (n_mlp < 0) n_mlp = 0;
  std::string dfrag = make_deferred(frag_src, hg_size, n_mlp);
  const char* vsrc = "#version 330 core\nvoid main(){ vec2 p=vec2((gl_VertexID<<1)&2, gl_VertexID&2);"
                     " gl_Position=vec4(p*2.0-1.0,0,1); }";
  GLuint prog = glCreateProgram();
  glAttachShader(prog, compile(vsrc, GL_VERTEX_SHADER));
  glAttachShader(prog, compile(dfrag.c_str(), GL_FRAGMENT_SHADER));
  glLinkProgram(prog);
  GLint linked; glGetProgramiv(prog, GL_LINK_STATUS, &linked);
  if (!linked) { char log[4096]; glGetProgramInfoLog(prog, 4096, nullptr, log); fprintf(stderr, "link: %s\n", log); return 1; }

  // ---- atlas texture (R32F), same packing as glsl_run / render_headless
  int AW = 4096; int AH = (int)((atlas.size() + AW - 1) / AW);
  std::vector<float> apad(AW * AH, 0.0f); std::memcpy(apad.data(), atlas.data(), atlas.size() * 4);
  GLuint atex; glGenTextures(1, &atex); glBindTexture(GL_TEXTURE_2D, atex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, AW, AH, 0, GL_RED, GL_FLOAT, apad.data());

  // ---- MLP-weight UBO (C3 perf lever): floats [hg_size, atlas_len), packed 4/vec4
  int nwvec = (n_mlp + 3) / 4;
  std::vector<float> wpad((size_t)nwvec * 4, 0.0f);
  if (n_mlp > 0) std::memcpy(wpad.data(), atlas.data() + hg_size, (size_t)n_mlp * 4);
  GLuint ubo; glGenBuffers(1, &ubo); glBindBuffer(GL_UNIFORM_BUFFER, ubo);
  glBufferData(GL_UNIFORM_BUFFER, (GLsizeiptr)wpad.size() * 4, wpad.data(), GL_STATIC_DRAW);
  GLuint blk = glGetUniformBlockIndex(prog, "Weights");
  if (blk != GL_INVALID_INDEX) { glUniformBlockBinding(prog, blk, 0); glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo); }
  else fprintf(stderr, "warning: Weights UBO block not found (falling back to texelFetch)\n");

  // ---- G-buffer textures (RGBA32F): pos+valid, n, rho, tint, emit
  auto mktex = [&]() { GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, RES, RES, 0, GL_RGBA, GL_FLOAT, nullptr); return t; };
  GLuint tPos = mktex(), tN = mktex(), tR = mktex(), tT = mktex(), tE = mktex();
  std::vector<float> bPos(RES * RES * 4), bN(RES * RES * 4), bR(RES * RES * 4), bT(RES * RES * 4), bE(RES * RES * 4);

  // ---- HUD texture + program (a textured fullscreen quad, alpha-blended over the scene)
  GLuint hudTex; glGenTextures(1, &hudTex); glBindTexture(GL_TEXTURE_2D, hudTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  std::vector<uint8_t> hud(RES * RES * 4, 0);
  const char* hud_fsrc = "#version 330 core\nout vec4 frag; uniform sampler2D uHud; uniform vec2 uRes;\n"
    "void main(){ vec2 uv=vec2(gl_FragCoord.x/uRes.x, 1.0-gl_FragCoord.y/uRes.y); frag=texture(uHud,uv);}";
  GLuint hprog = glCreateProgram();
  glAttachShader(hprog, compile(vsrc, GL_VERTEX_SHADER));
  glAttachShader(hprog, compile(hud_fsrc, GL_FRAGMENT_SHADER));
  glLinkProgram(hprog);

  GLuint vao; glGenVertexArrays(1, &vao); glBindVertexArray(vao);
  GLuint timer; glGenQueries(1, &timer);

  // ---- offscreen scene FBO (render at RES, then blit letterboxed to the window/fullscreen) ----
  GLuint sceneTex; glGenTextures(1, &sceneTex); glBindTexture(GL_TEXTURE_2D, sceneTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, RES, RES, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
  GLuint sceneFbo; glGenFramebuffers(1, &sceneFbo); glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sceneTex, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  const char* blit_fsrc = "#version 330 core\nout vec4 frag; uniform sampler2D uScene; uniform vec2 uRes; uniform vec2 uOff;\n"
    "void main(){ frag = texture(uScene, (gl_FragCoord.xy - uOff)/uRes); }";
  GLuint bprog = glCreateProgram();
  glAttachShader(bprog, compile(vsrc, GL_VERTEX_SHADER));
  glAttachShader(bprog, compile(blit_fsrc, GL_FRAGMENT_SHADER));
  glLinkProgram(bprog);

  // ---- lights (from level.toml): torch (dynamic) + 2 static sconces; cap 4
  V3 torchE{12.0f, 8.5f, 5.0f};
  std::vector<Light> statics = {{{1.5f, 0.6f, 1.5f}, {6.0f, 5.0f, 4.0f}, 0.10f},
                                {{10.5f, 0.6f, 6.5f}, {4.0f, 4.5f, 6.0f}, 0.10f}};
  const float TORCH_R = 0.12f;
  const int MAXB = 2;   // Two analytic specular bounces

  Cam cam{{2.5f, EYE, 2.5f}, 0.4f, 0.0f, false, true, {0, 0, 0}};

  if (!headless) SDL_SetRelativeMouseMode(SDL_TRUE);

  bool running = true; int frame = 0;
  double cpu_ms = 0, gpu_ms = 0, fps = 0; uint32_t last = SDL_GetTicks();

  while (running) {
    // -------- input / camera update --------
    float dt = (SDL_GetTicks() - last) / 1000.0f; if (dt <= 0) dt = 1.0f / 60; last = SDL_GetTicks();
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      if (e.type == SDL_QUIT) running = false;
      if (e.type == SDL_MOUSEMOTION) { cam.yaw += e.motion.xrel * 0.0025f; cam.pitch -= e.motion.yrel * 0.0025f; }
      if (e.type == SDL_KEYDOWN) {
        SDL_Keycode s = e.key.keysym.sym;
        if (s == SDLK_ESCAPE) running = false;
        if (s == SDLK_f) cam.fly = !cam.fly;
        if (s == SDLK_l) { cam.torch_follow = !cam.torch_follow; if (!cam.torch_follow) cam.torch_pos = cam.pos + cam_fwd(cam) * 0.3f; }
        if (s == SDLK_F11 || (s == SDLK_RETURN && (e.key.keysym.mod & KMOD_ALT))) {
          is_fullscreen = !is_fullscreen;
          SDL_SetWindowFullscreen(win, is_fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
        }
        if (s == SDLK_g) SDL_SetRelativeMouseMode(SDL_GetRelativeMouseMode() ? SDL_FALSE : SDL_TRUE);
      }
    }
    const Uint8* k = SDL_GetKeyboardState(0);
    cam.pitch = std::fmax(-1.4f, std::fmin(1.4f, cam.pitch));
    float sp = (k[SDL_SCANCODE_LSHIFT] ? 4.5f : 2.2f) * dt;
    V3 fwdg = norm({std::cos(cam.yaw), 0, std::sin(cam.yaw)});
    V3 rightg = norm(cross(fwdg, {0, 1, 0}));
    V3 mv{0, 0, 0};
    V3 walkFwd = cam.fly ? cam_fwd(cam) : fwdg;
    if (k[SDL_SCANCODE_W]) mv = mv + walkFwd;
    if (k[SDL_SCANCODE_S]) mv = mv - walkFwd;
    if (k[SDL_SCANCODE_D]) mv = mv + rightg;
    if (k[SDL_SCANCODE_A]) mv = mv - rightg;
    if (cam.fly) { if (k[SDL_SCANCODE_SPACE]) mv.y += 1; if (k[SDL_SCANCODE_LCTRL]) mv.y -= 1; }
    mv = norm(mv) * sp;
    V3 np = cam.pos + mv;
    if (cam.fly) { cam.pos = np; }
    else { np.y = EYE; if (!is_wall((int)(np.z / CELL), (int)(np.x / CELL))) cam.pos = np; }  // wall collide
    if (cam.torch_follow) cam.torch_pos = cam.pos + cam_fwd(cam) * 0.3f;
    // -------- pass 1: CPU G-buffer + specular chase (OpenMP) --------
    auto c0 = std::chrono::high_resolution_clock::now();
    V3 fwd = cam_fwd(cam), wup{0, 1, 0};
    V3 right = norm(cross(fwd, wup)); V3 tup = cross(right, fwd);
    float htan = std::tan(75.0f * 3.14159265f / 180.0f / 2.0f);
    std::vector<Light> lights = {{cam.torch_pos, torchE, TORCH_R}};
    for (auto& s : statics) lights.push_back(s);
    int cap = std::max(1, std::min(4, light_cap));
    if ((int)lights.size() > cap) lights.resize(cap);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int j = 0; j < RES; j++) {
      for (int i = 0; i < RES; i++) {
        float sx = (i + 0.5f) / RES * 2 - 1, sy = 1 - (j + 0.5f) / RES * 2;
        V3 d = norm(fwd + right * (sx * htan) + tup * (sy * htan));
        V3 o = cam.pos;
        V3 tint{1, 1, 1}, emit{0, 0, 0};
        float validEP = 0; V3 gp{0, 0, 0}, gn{0, 1, 0}, gr{0, 0, 0};
        for (int b = 0; b <= MAXB; b++) {
          float tw; V3 pw, nw; int mat; bool hw = cast_world(o, d, tw, pw, nw, mat);
          float to; V3 no, f0; bool ho = nearest_obj(o, d, to, no, f0);
          float surf = std::fmin(hw ? tw : TFAR, ho ? to : TFAR);
          // light seen in this segment (torch/sconce sphere) before any surface
          float tl = TFAR; V3 El{0, 0, 0};
          for (auto& L : lights) { float tt; V3 nn;
            if (hit_sphere(o, d, L.pos, std::fmax(L.radius, 1e-3f), tt, nn) && tt < tl && tt < surf) { tl = tt; El = L.E; } }
          if (tl < surf) {   // torch/sconce sphere seen directly or in the mirror -> add its radiance
            emit.x += tint.x * El.x; emit.y += tint.y * El.y; emit.z += tint.z * El.z;
            break;
          }
          if (surf >= TFAR) break;                       // miss -> background
          bool objNear = ho && (to <= (hw ? tw : TFAR));
          if (!objNear) {                                // diffuse endpoint -> NRP shades it
            gp = pw; gn = nw; const float* al = ALB[mat]; gr = {al[0], al[1], al[2]}; validEP = 1; break;
          }
          // conductor: specular chase
          if (b < MAXB) {
            V3 po = o + d * to; float cosT = std::fabs(dot(d * -1.0f, no));
            V3 F = schlick(f0, cosT); tint = {tint.x * F.x, tint.y * F.y, tint.z * F.z};
            d = norm(reflect(d, no)); o = po + no * 1e-3f;
          } else { emit = emit + tint * 0.02f; break; }  // budget exhausted on metal: dim fallback
        }
        int idx = (j * RES + i) * 4;
        bPos[idx] = gp.x; bPos[idx + 1] = gp.y; bPos[idx + 2] = gp.z; bPos[idx + 3] = validEP;
        bN[idx] = gn.x; bN[idx + 1] = gn.y; bN[idx + 2] = gn.z; bN[idx + 3] = 0;
        bR[idx] = gr.x; bR[idx + 1] = gr.y; bR[idx + 2] = gr.z; bR[idx + 3] = 0;
        bT[idx] = tint.x; bT[idx + 1] = tint.y; bT[idx + 2] = tint.z; bT[idx + 3] = 0;
        bE[idx] = emit.x; bE[idx + 1] = emit.y; bE[idx + 2] = emit.z; bE[idx + 3] = 0;
      }
    }
    auto c1 = std::chrono::high_resolution_clock::now();
    cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();

    // upload G-buffer
    auto uploadTex = [&](GLuint t, std::vector<float>& b) { glBindTexture(GL_TEXTURE_2D, t);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, RES, RES, 0, GL_RGBA, GL_FLOAT, b.data()); };
    uploadTex(tPos, bPos); uploadTex(tN, bN); uploadTex(tR, bR); uploadTex(tT, bT); uploadTex(tE, bE);

    // -------- pass 2: GPU NRP shading (timed), into the offscreen RES x RES scene FBO --------
    glBindFramebuffer(GL_FRAMEBUFFER, sceneFbo);
    glViewport(0, 0, RES, RES);
    glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    struct { const char* n; int u; GLuint t; } binds[] = {
      {"uAtlas", 0, atex}, {"uGPos", 1, tPos}, {"uGN", 2, tN}, {"uGRho", 3, tR}, {"uGTint", 4, tT}, {"uGEmit", 5, tE}};
    for (auto& bd : binds) { glActiveTexture(GL_TEXTURE0 + bd.u); glBindTexture(GL_TEXTURE_2D, bd.t);
      glUniform1i(glGetUniformLocation(prog, bd.n), bd.u); }
    glUniform1i(glGetUniformLocation(prog, "uAtlasW"), AW);
    glUniform3f(glGetUniformLocation(prog, "uAABBmin"), aabbmin.x, aabbmin.y, aabbmin.z);
    glUniform3f(glGetUniformLocation(prog, "uAABBmax"), aabbmax.x, aabbmax.y, aabbmax.z);
    glUniform2f(glGetUniformLocation(prog, "uRes"), (float)RES, (float)RES);
    glUniform1f(glGetUniformLocation(prog, "uExposure"), 1.0f);
    int nl = (int)lights.size();
    glUniform1i(glGetUniformLocation(prog, "uNLights"), nl);
    for (int i = 0; i < nl; i++) {
      char nm[32]; snprintf(nm, 32, "uLightPos[%d]", i);
      glUniform3f(glGetUniformLocation(prog, nm), lights[i].pos.x, lights[i].pos.y, lights[i].pos.z);
    }
    glBeginQuery(GL_TIME_ELAPSED, timer);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndQuery(GL_TIME_ELAPSED);
    GLuint64 ns = 0; glGetQueryObjectui64v(timer, GL_QUERY_RESULT, &ns);
    gpu_ms = ns / 1e6;

    // -------- HUD (burned into the frame) --------
    std::fill(hud.begin(), hud.end(), 0);
    double total_ms = cpu_ms + gpu_ms; double inst_fps = total_ms > 0 ? 1000.0 / total_ms : 0;
    fps = fps == 0 ? inst_fps : fps * 0.9 + inst_fps * 0.1;
    char l1[160], l2[160], l3[160], l4[160], l0[200];
    snprintf(l0, sizeof l0, "LIVE FREE CAMERA - seed 1337 - %s", renderer.c_str());
    snprintf(l1, sizeof l1, "POS %.2f %.2f %.2f  YAW %.3f  PITCH %.3f", cam.pos.x, cam.pos.y, cam.pos.z, cam.yaw, cam.pitch);
    snprintf(l2, sizeof l2, "FPS %.1f  (cpu %.1fms + gpu %.2fms)  RES %d", fps, cpu_ms, gpu_ms, RES);
    snprintf(l3, sizeof l3, "LIGHTS %d  FLY %s  TORCH-FOLLOW %s%s", nl, cam.fly ? "ON" : "off",
             cam.torch_follow ? "ON" : "off", software_gl ? "  [SOFTWARE-GL]" : "");
    snprintf(l4, sizeof l4, "frame %d  WASD move  mouse look  F fly  L torch  ESC quit", frame);
    hud_text(hud, RES, RES, 6, 6, l0, 255, 210, 90);
    hud_text(hud, RES, RES, 6, 6 + 15, l1);
    hud_text(hud, RES, RES, 6, 6 + 30, l2);
    hud_text(hud, RES, RES, 6, 6 + 45, l3);
    hud_text(hud, RES, RES, 6, RES - 18, l4, 150, 180, 220);
    glBindTexture(GL_TEXTURE_2D, hudTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, RES, RES, 0, GL_RGBA, GL_UNSIGNED_BYTE, hud.data());
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(hprog);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, hudTex);
    glUniform1i(glGetUniformLocation(hprog, "uHud"), 0);
    glUniform2f(glGetUniformLocation(hprog, "uRes"), (float)RES, (float)RES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_BLEND);

    // -------- optional screenshot frames (PPM, read the scene framebuffer) --------
    if (!save_dir.empty()) {
      std::vector<uint8_t> px(RES * RES * 3);
      glReadPixels(0, 0, RES, RES, GL_RGB, GL_UNSIGNED_BYTE, px.data());
      char fn[512]; snprintf(fn, sizeof fn, "%s/frame_%05d.ppm", save_dir.c_str(), frame);
      std::ofstream of(fn, std::ios::binary); of << "P6\n" << RES << " " << RES << "\n255\n";
      for (int y = RES - 1; y >= 0; y--) of.write((char*)&px[y * RES * 3], RES * 3);  // flip to top-down
    }

    // -------- blit the square scene to the window, centered + letterboxed (handles fullscreen/resize) --------
    if (!headless) {
      int ww, wh; SDL_GL_GetDrawableSize(win, &ww, &wh);
      int side = std::min(ww, wh);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glViewport(0, 0, ww, wh); glClearColor(0, 0, 0, 1); glClear(GL_COLOR_BUFFER_BIT);
      int offx = (ww - side) / 2, offy = (wh - side) / 2;
      glViewport(offx, offy, side, side);
      glUseProgram(bprog);
      glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, sceneTex);
      glUniform1i(glGetUniformLocation(bprog, "uScene"), 0);
      glUniform2f(glGetUniformLocation(bprog, "uRes"), (float)side, (float)side);
      glUniform2f(glGetUniformLocation(bprog, "uOff"), (float)offx, (float)offy);
      glDrawArrays(GL_TRIANGLES, 0, 3);
      SDL_GL_SwapWindow(win);
    }
    frame++;
    if (max_frames > 0 && frame >= max_frames) running = false;
  }

  if (headless)
    printf("{\"renderer\":\"%s\",\"software_gl\":%s,\"res\":%d,\"frames\":%d,\"fps\":%.2f,\"cpu_ms\":%.2f,\"gpu_ms\":%.3f}\n",
           renderer.c_str(), software_gl ? "true" : "false", RES, frame, fps, cpu_ms, gpu_ms);
  SDL_GL_DeleteContext(ctx); SDL_DestroyWindow(win); SDL_Quit();
  return 0;
}
static uint16_t f32_to_f16bits(float) { return 0; }
