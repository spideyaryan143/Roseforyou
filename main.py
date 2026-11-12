# Rose - Web Ready Version (Real-Time Speed Fix)
# JIN2 (@jin_22ee), 2026. Free to use for personal projects, see LICENSE.txt.

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pygame

TAU = np.pi * 2
LOOP = 20.0         # length of one run (s)
FLOOR = -1.22

POOLS = ["1il!;:", "rtfjcvxz=+", "aeoun0{}[]<>", "08B$#&%Q"]   # dark -> bright
CHARS = " .,:~+*^01|/\\-_#()x" + "".join(POOLS)
AT = {c: i for i, c in enumerate(CHARS)}
FILL = np.array([[AT[p[k % len(p)]] for k in range(12)] for p in POOLS])

STEM, LEAF, SEPAL, PETAL, EDGE, GLINT, FIRE_R, FIRE_B, DUST, REFL = range(10)


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


# stem and leaves

HEAD_Y = 0.22


def stem_curve(s):
    # centre line, s = 0 at the bottom, 1 under the flower
    y = -1.15 + s * (HEAD_Y + 1.15)
    x = 0.045 * np.sin(s * 3.1 + 0.4) - 0.02
    z = 0.02 * np.sin(s * 2.0)
    return np.stack([x, y, z], -1)


def stem():
    s, phi = [a.ravel() for a in np.meshgrid(np.linspace(0, 1, 260), np.linspace(0, TAU, 14, endpoint=False))]
    c = stem_curve(s)
    n = np.stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)], 1)
    r = 0.026 + 0.006 * (1 - s)
    return c + n * r[:, None], n, s


def leaf(at, side, length, lift, twist):
    # pointed oval, folded along the midrib, tip bending down
    s, w = [a.ravel() for a in np.meshgrid(np.linspace(0, 1, 70), np.linspace(-1, 1, 25))]
    base = stem_curve(np.array(at))
    d = np.array([side * np.cos(lift), np.sin(lift), 0.25 * twist])
    d /= np.linalg.norm(d)
    wide = np.cross(d, [0, 0, 1.0])
    wide /= np.linalg.norm(wide)
    up = np.cross(wide, d)
    hw = 0.15 * np.sin(np.pi * np.clip(s, 0, 1)) ** 0.75 * length / 0.42
    p = (base + d * 0.07 + d * (s * length)[:, None] + wide * (w * hw)[:, None]
         + up * (0.05 * np.abs(w) - 0.10 * s ** 2)[:, None])
    du = np.gradient(p.reshape(25, 70, 3), axis=1).reshape(-1, 3)
    dv = np.gradient(p.reshape(25, 70, 3), axis=0).reshape(-1, 3)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    n *= np.sign(n @ up + 1e-9)[:, None]
    return p, n, s


STEM_P, STEM_N, STEM_S = stem()
LEAF_AT = [0.32, 0.55]
LEAVES = [leaf(LEAF_AT[0], -1, 0.42, 0.35, 1), leaf(LEAF_AT[1], 1, 0.36, 0.45, -1)]

SEP_S, SEP_W = [x.ravel() for x in np.meshgrid(np.linspace(0, 1, 34), np.linspace(-1, 1, 9))]


def sepals(open_):
    # five sepals, spread sideways, folding further back as the flower opens
    P, N = [], []
    s, w = SEP_S, SEP_W
    beta = (1.05 + 0.35 * open_) + (0.55 + 0.75 * open_) * s
    L = 0.36 - 0.06 * open_
    for i in range(5):
        a = TAU * i / 5 + 0.3
        r = 0.045 + L * s * np.sin(beta)
        ang = a + w * 0.10 * (1 - s) ** 1.3 / np.maximum(r / 0.2, 0.25)
        y = HEAD_Y + 0.02 + L * s * np.cos(beta)
        p = np.stack([np.cos(ang) * r, y, np.sin(ang) * r], 1)
        n = np.stack([np.cos(ang) * np.cos(beta), -np.sin(beta), np.sin(ang) * np.cos(beta)], 1)
        P.append(p)
        N.append(n / np.linalg.norm(n, axis=1, keepdims=True))
    return np.concatenate(P), np.concatenate(N)


N_SEPAL = 5 * len(SEP_S)


# bloom: one spiral surface, blended from a closed bud

NT, NX = 1300, 40
TH, XX = np.meshgrid(np.linspace(2.2 * np.pi, 20 * np.pi, NT), np.linspace(0, 1, NX))
THN = (TH - TH.min()) / (TH.max() - TH.min())           # 0 outer .. 1 heart
PETAL_ID = np.floor(3.6 * TH / TAU).astype(int)
SIZE = 0.80


def petals(open_):
    A = 1.60 + 4.00 * open_
    phi = (np.pi / 2) * np.exp(-TH / (A * np.pi))
    X = 1 - 0.5 * (1.25 * (1 - np.mod(3.6 * TH, TAU) / np.pi) ** 2 - 0.25) ** 2
    yy = 1.95653 * XX ** 2 * (1.27689 * XX - 1) ** 2 * np.sin(phi)
    r = X * (XX * np.sin(phi) + yy * np.cos(phi))
    p = np.stack([r * np.sin(TH), X * (XX * np.cos(phi) - yy * np.sin(phi)), r * np.cos(TH)], -1) * SIZE
    p[..., 1] *= 0.84 + 0.16 * open_
    p[..., [0, 2]] *= 1.05 - 0.05 * open_
    p[..., 1] += HEAD_Y
    # closed bud: egg profile, inner layers slightly smaller
    h = XX
    r = 0.170 * h ** 0.70 * (1 - h) ** 0.75 / 0.367 * (1 - 0.30 * THN) + 0.003
    bud = np.stack([r * np.sin(TH), HEAD_Y + 0.03 + 0.52 * h * (1 - 0.06 * THN), r * np.cos(TH)], -1)
    w = smooth(open_ / 0.6)
    p = bud * (1 - w) + p * w
    du = np.gradient(p, axis=1)
    dv = np.gradient(p, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    out = np.stack([np.sin(TH), np.zeros_like(TH) + 0.2, np.cos(TH)], -1)
    n *= np.sign((n * out).sum(-1, keepdims=True) + 1e-9)
    edge = XX > 0.93
    return p.reshape(-1, 3), n.reshape(-1, 3), edge.reshape(-1)


PETAL_J = PETAL_ID.reshape(-1) - PETAL_ID.min()
PETAL_SJ = 1 - THN.reshape(-1)                          # 0 heart .. 1 outer
PETAL_V = XX.reshape(-1)


def bloom(t):
    # 0 = bud, 1 = full bloom
    u = np.clip((t - 5.5) / 9.0, 0, 1)
    return u * (2 - u) * 0.55 + u * u * (3 - 2 * u) * 0.45


# drawing order: stem, leaves, sepals, then the bud from base to tip

O_STEM = STEM_S
O_LEAF = [LEAF_AT[k] + 0.05 + LEAVES[k][2] * 0.25 for k in range(2)]
O_SEPAL = 1.0 + 0.10 * np.tile(SEP_S, 5)
O_PETAL = 1.10 + PETAL_V * 0.42 + PETAL_SJ * 0.10
O_MAX = 1.66
DRAW = (0.3, 5.4)


def reveal(order, t):
    head = (t - DRAW[0]) / (DRAW[1] - DRAW[0]) * O_MAX
    soft = np.where(order >= 1.0, 0.09, 0.05)
    return smooth((head - order) / soft), (np.abs(head - order) < 0.03) & (order < 1.0)


# lights, motion

def lights(t):
    key = np.array([-0.35, 0.85, 0.65])
    rim = np.array([np.cos(t * 0.8), 0.4, np.sin(t * 0.8)])
    return key / np.linalg.norm(key), rim / np.linalg.norm(rim)


def sway(t):
    return 0.22 * np.sin(max(t - 7.0, 0) * 0.35) * smooth((t - 7.0) / 4.0)


def rot_y(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([c * p[:, 0] + s * p[:, 2], p[:, 1], -s * p[:, 0] + c * p[:, 2]], 1)


# particles

SPK = np.random.default_rng(13).random((1600, 5))


def sparks(t, tip_pts):
    # thrown off the growing stem, pulled down
    if len(tip_pts) == 0 or t > DRAW[1] + 0.2:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)
    k = (SPK[:, 0] * len(tip_pts)).astype(int)
    age = (t * 1.7 + SPK[:, 1]) % 1.0
    a = SPK[:, 2] * TAU
    v = np.stack([np.cos(a) * (0.3 + 0.5 * SPK[:, 3]), 0.3 + 0.7 * SPK[:, 4], np.sin(a) * (0.3 + 0.5 * SPK[:, 3])], 1)
    p = tip_pts[k] + v * (age * 0.55)[:, None] - np.array([0, 1.0, 0]) * (0.5 * 1.4 * (age * 0.55) ** 2)[:, None]
    glow = (1 - age) ** 1.4 * (0.6 + 0.4 * np.sin(age * 25 + a))
    return p, glow, np.where(SPK[:, 3] > 0.6, GLINT, DUST)


GLITTER = np.random.default_rng(21).random((700, 5))


def glitter(t):
    # slow cloud of twinkling dust around the rose
    g = GLITTER
    a = g[:, 0] * TAU + t * (0.05 + 0.08 * g[:, 1])
    r = 0.55 + 1.6 * g[:, 2] ** 0.7
    y = -1.1 + 2.3 * g[:, 3] + 0.05 * np.sin(t * 0.6 + g[:, 4] * TAU)
    p = np.stack([np.cos(a) * r, y, np.sin(a) * r * 0.6 - 0.35], 1)
    tw = np.clip(np.sin(t * (1.2 + 2.5 * g[:, 1]) + g[:, 4] * 40), 0, 1) ** 6
    glow = (0.10 + 0.85 * tw) * smooth((t - 1.0 - g[:, 3] * 3.0) / 1.5)
    kind = np.select([g[:, 4] < 0.12, g[:, 4] < 0.3], [FIRE_B, FIRE_R], GLINT)
    return p, glow, kind


DEW = np.random.default_rng(31).random((40, 2))

# falling drops: hang on a low petal tip, fall, splash
_open = petals(1.0)[0]
_low = np.flatnonzero((PETAL_SJ > 0.7) & (PETAL_V > 0.82))
DROP_AT = _low[np.argsort(_open[_low, 1])[:400:67]]
DROP_T0 = 13.4 + np.array([0.0, 1.5, 2.8, 4.0])
SPLASH = np.random.default_rng(51).random((8, 2))
G_DROP = 1.7


def dew_drops(t, flower):
    P, S, T = [], [], []
    for idx, t0 in zip(DROP_AT, DROP_T0):
        tau = t - t0
        if tau < 0:
            continue
        hang = 1.4
        start = flower[idx] + np.array([0, -0.012, 0.01])
        if tau < hang:
            g = 0.006 + 0.012 * tau / hang
            P.append(start + np.array([[0, 0, 0], [g, 0, 0], [-g, 0, 0], [0, -g, 0], [0, -2 * g, 0]]))
            S.append([0.8 + 0.5 * tau / hang] * 5)
            T.append([FIRE_B] * 5)
            continue
        u = tau - hang
        y = start[1] - 0.5 * G_DROP * u ** 2
        if y > FLOOR + 0.02:
            v = G_DROP * u
            for j, back in enumerate((0.0, 0.02, 0.04, 0.07, 0.10)):
                P.append(np.array([[start[0], y + back * min(v, 1.2), start[2]]]))
                S.append([1.3 - 0.22 * j])
                T.append([FIRE_B])
        else:
            w = u - np.sqrt(2 * (start[1] - FLOOR) / G_DROP)
            if w > 0.7:
                continue
            a = SPLASH[:, 0] * TAU
            sp = 0.25 + 0.35 * SPLASH[:, 1]
            px = start[0] + np.cos(a) * sp * w
            pz = start[2] + np.sin(a) * sp * w * 0.5
            py = FLOOR + 0.02 + 0.9 * sp * w - 0.5 * G_DROP * w ** 2
            ok = py >= FLOOR
            P.append(np.stack([px, py, pz], 1)[ok])
            S.append(np.full(ok.sum(), 0.9 * (1 - w / 0.7)))
            T.append(np.full(ok.sum(), FIRE_B))
    if not P:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, int)
    return np.concatenate(P), np.concatenate([np.asarray(x, float) for x in S]), np.concatenate([np.asarray(x, int) for x in T])


# camera, scene

def look(P):
    target, eye = np.array([0, -0.05, 0]), np.array([0.0, 1.75, 4.0])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 1.0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    rel = P - eye
    return rel @ right, rel @ up, rel @ fwd, -fwd


def scene(t):
    yaw = sway(t)
    op = bloom(t)
    pp, pn, pedge = petals(op)
    sep_p, sep_n = sepals(op)
    lp = [lf[0] for lf in LEAVES]
    ln = [lf[1] for lf in LEAVES]
    body = np.concatenate([STEM_P] + lp + [sep_p, pp])
    normal = np.concatenate([STEM_N] + ln + [sep_n, pn])
    order = np.concatenate([O_STEM] + O_LEAF + [O_SEPAL, O_PETAL])
    kind = np.concatenate([np.full(len(STEM_P), STEM), np.full(len(lp[0]) + len(lp[1]), LEAF),
                           np.full(N_SEPAL, SEPAL), np.where(pedge, EDGE, PETAL)])
    part = np.concatenate([np.zeros(len(STEM_P), int), np.ones(len(lp[0]), int), np.full(len(lp[1]), 2),
                           np.full(N_SEPAL, 3), 10 + PETAL_J])
    k, tip = reveal(order, t)

    # outer, lower petals lag behind the sway
    vel = (sway(t + 0.05) - sway(t - 0.05)) / 0.1
    fi0 = len(body) - len(pp)
    loose = np.clip((PETAL_SJ - 0.45) / 0.55, 0, 1) * PETAL_V ** 2 * op
    ripple = 0.010 * np.sin(t * 2.4 + PETAL_J * 1.7) * smooth((t - 8.0) / 2.0)
    body[fi0:, 0] += -0.10 * vel * loose
    body[fi0:, 1] += (ripple - 0.006 * np.abs(vel)) * loose
    body = rot_y(body, yaw)
    normal = rot_y(normal, yaw)

    key, rim = lights(t)
    view = look(np.zeros((1, 3)))[3]
    diff = np.clip(normal @ key, 0, 1)
    half_k = (key + view) / np.linalg.norm(key + view)
    half_r = (rim + view) / np.linalg.norm(rim + view)
    spec = np.clip(normal @ half_k, 0, 1) ** 24 + 0.7 * np.clip(normal @ half_r, 0, 1) ** 18
    is_petal = kind >= PETAL
    depth = np.concatenate([np.ones(len(body) - len(pp)), 0.74 + 0.26 * PETAL_SJ])   # darker heart
    shade = (0.62 + 0.60 * diff + np.where(is_petal, 0.35, 0.25) * spec) * depth

    # dew on the outer petals
    if op > 0.85:
        outer = np.flatnonzero(PETAL_SJ > 0.55)
        idx = outer[(DEW[:, 0] * len(outer)).astype(int)] + len(body) - len(pp)
        tw = np.clip(np.sin(t * 3 + DEW[:, 1] * 30), 0, 1) ** 8
        kind[idx] = np.where(tw > 0.4, GLINT, kind[idx])
        shade[idx] = np.maximum(shade[idx], 0.6 + 0.6 * tw)

    part = np.where(k < 0.96, -2, part)
    shade = np.where(k < 0.96, np.where(tip, 1.2, 0.0), shade)
    kind = np.where(k < 0.96, np.where(tip, GLINT, DUST), kind)

    refl = body.copy()                                  # floor reflection
    refl[:, 1] = 2 * FLOOR - refl[:, 1]
    rs = np.where(k >= 0.96, 0.18 * np.clip(shade, 0, 1.2), 0)

    sp, ss, st = sparks(t, body[tip])
    gp, gs, gt = glitter(t)
    dp, ds, dt_ = dew_drops(t, body[len(body) - len(pp):])
    P = np.concatenate([body, refl, gp, sp, dp])
    shade = np.concatenate([shade, rs, gs, ss, ds])
    tag = np.concatenate([kind, np.full(len(refl), REFL), gt, st, dt_])
    pp_ = np.concatenate([part, np.full(len(P) - len(part), -1)])
    return P, shade, tag, pp_, {"bloom": float(op), "deg": float(np.degrees(yaw) % 360)}


def outline_letter(gx, gy):
    ax, ay = np.abs(gx), np.abs(gy)
    c = np.where(ax > 2 * ay, AT["|"], np.where(ay > 2 * ax, np.where(gy > 0, AT["_"], AT["-"]),
                 np.where(gx * gy > 0, AT["/"], AT["\\"])))
    return np.where((ax + ay) == 0, AT["+"], c)


def rays(char, lum, tag, cy, cx, reach, rows, cols, strength=1.0):
    for dist in range(1, reach + 1):
        fade = (1 - dist / (reach + 1)) * strength
        for (dy, dx, ch) in ((0, 1, "-"), (0, -1, "-"), (1, 0, "|"), (-1, 0, "|")):
            yy, xx = cy + dy * dist, cx + dx * dist * 2
            if 0 <= yy < rows and 0 <= xx < cols and lum[yy, xx] < fade:
                char[yy, xx], lum[yy, xx], tag[yy, xx] = AT[ch], fade, GLINT
        if dist <= reach // 2:
            for (dy, dx, ch) in ((1, 1, "\\"), (-1, -1, "\\"), (1, -1, "/"), (-1, 1, "/")):
                yy, xx = cy + dy * dist, cx + dx * dist
                if 0 <= yy < rows and 0 <= xx < cols and lum[yy, xx] < fade * 0.7:
                    char[yy, xx], lum[yy, xx], tag[yy, xx] = AT[ch], fade * 0.7, GLINT
    char[cy, cx], lum[cy, cx] = AT["*"], 1.0


def frame(t, cols, rows, aspect):
    P, shade, T, K, info = scene(t)
    x, y, z, _ = look(P)
    F = 2.9
    zz = np.maximum(z, 0.2)
    sx = (0.5 + x / zz * F * 0.5) * cols
    sy = (0.5 - y / zz * F * 0.5 * aspect) * rows
    order = np.argsort(-z)                              # far to near
    sx, sy, z, shade, T, K = sx[order], sy[order], z[order], shade[order], T[order], K[order]
    ix, iy = sx.astype(int), sy.astype(int)
    ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows) & (z > 0.2)

    part = np.full((rows, cols), -1)
    depth = np.full((rows, cols), 99.0)
    lum = np.zeros((rows, cols))
    tag = np.zeros((rows, cols), int)
    b = ok & (K >= 0)
    part[iy[b], ix[b]], depth[iy[b], ix[b]] = K[b], z[b]
    lum[iy[b], ix[b]], tag[iy[b], ix[b]] = shade[b], T[b]
    body = part >= 0

    # outlines: next to empty space or to a different part further back
    pp = np.pad(part, 1, constant_values=-1)
    pd = np.pad(depth, 1, constant_values=99)
    edge = np.zeros_like(body)
    gx = np.zeros((rows, cols))
    gy = np.zeros((rows, cols))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == dy == 0:
                continue
            nb = pp[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx]
            nz = pd[1 + dy:rows + 1 + dy, 1 + dx:cols + 1 + dx]
            out = body & ((nb < 0) | ((nb != part) & (nz > depth + 0.035)))
            gx += dx * out
            gy += dy * out
            if dx == 0 or dy == 0:
                edge |= out

    r, c = np.mgrid[0:rows, 0:cols]
    h = (r * 7919 + c * 104729) % 9973
    level = np.clip((lum * 3.2).astype(int), 0, 3)
    inside = FILL[level, (h + (t * 4 + h % 11).astype(int)) % 12]
    sparkle = body & (tag == GLINT)
    char = np.where(body, np.where(edge, outline_letter(gx, gy), inside), 0)
    char = np.where(sparkle, AT["*"], char)
    lum = np.where(body, np.where(edge, np.maximum(0.85, lum), np.clip(0.45 + 0.55 * lum / 0.9, 0.45, 1)), 0)

    # particles, reflection
    fx = ok & (K < 0) & (shade > 0.02)
    hidden = body[iy[fx], ix[fx]] & (depth[iy[fx], ix[fx]] < z[fx])
    fi, fj, fs, ft = (a[fx][~hidden] for a in (iy, ix, shade, T))
    keep = ~body[fi, fj]
    fi, fj, fs, ft = fi[keep], fj[keep], fs[keep], ft[keep]
    refl = FILL[np.clip((fs * 6).astype(int), 0, 1), (fi * 31 + fj) % 12]
    mote = np.where(fs > 0.7, AT["*"], np.where(fs > 0.35, AT["+"], AT["."]))
    char[fi, fj], lum[fi, fj], tag[fi, fj] = np.where(ft == REFL, refl, mote), np.clip(fs, 0, 1), ft

    # rays: brightest glints, falling drops, brightest dust
    score = np.where(sparkle, lum + (h % 7) * 0.01, 0)
    for n, idx in enumerate(np.argsort(score.ravel())[::-1][:4]):
        if score.ravel()[idx] <= 0:
            break
        cy, cx = divmod(int(idx), cols)
        rays(char, lum, tag, cy, cx, int(4 + 6 * (0.5 + 0.5 * np.sin(t * 7 + n * 1.7))), rows, cols)
    drop = (ft == FIRE_B) & (fs > 1.0)
    for cy, cx in zip(fi[drop], fj[drop]):
        rays(char, lum, tag, int(cy), int(cx), 5, rows, cols)
    bright = (fs > 0.9) & np.isin(ft, [GLINT, FIRE_B, FIRE_R])
    for cy, cx, v in list(zip(fi[bright], fj[bright], fs[bright]))[:5]:
        rays(char, lum, tag, int(cy), int(cx), int(2 + 3 * min(v, 1.1)), rows, cols, strength=0.6)
    return char, lum, tag, info, int(body.sum())


# window

TITLE = "Rose - JIN2"
SIGN = "@jin_22ee · 2026"
CW, CH, FONT_PX = 9, 14, 15
W, H = 1080, 960
COLS, ROWS = W // CW, H // CH
ASPECT = (COLS * CW) / (ROWS * CH)
TRAIL = 0.5
COLORS = np.array([[70, 214, 96], [52, 196, 84], [80, 206, 100], [255, 34, 64], [255, 120, 146], [255, 255, 255],
                   [255, 176, 204], [188, 220, 255], [255, 214, 150], [140, 60, 76]], np.float32) / 255
DIM, PINK = (110, 104, 120), (255, 122, 182)
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"


def hud(info):
    return f"bloom {info['bloom'] * 100:5.1f}%   sway {info['deg']:06.2f}°"


def glyphs():
    font = pygame.font.SysFont(MONO, FONT_PX)
    atlas = []
    for ch in CHARS:
        s = pygame.Surface((CW, CH))
        g = font.render(ch, True, (255, 255, 255))
        s.blit(g, ((CW - g.get_width()) // 2, (CH - g.get_height()) // 2))
        atlas.append(pygame.surfarray.array3d(s)[:, :, 0].T.astype(np.float32) / 255)
    return np.stack(atlas)


class Canvas:
    def __init__(self, atlas):
        self.atlas = (atlas * 255).astype(np.uint8)
        self.shape = pygame.Surface((COLS * CW, ROWS * CH))
        self.trail = pygame.Surface((W, H))
        self.fade = (int(TRAIL * 255),) * 3

    def reset(self):
        self.trail.fill((0, 0, 0))

    def draw(self, t):
        char, lum, tag, info, count = frame(t, COLS, ROWS, ASPECT)
        mask = self.atlas[char.T].transpose(0, 3, 1, 2).reshape(COLS * CW, ROWS * CH)
        pygame.surfarray.blit_array(self.shape, np.repeat(mask[..., None], 3, 2))
        col = (COLORS[np.clip(tag, 0, len(COLORS) - 1)] * np.clip(lum, 0, 1)[..., None] * 255).astype(np.uint8)
        tint = pygame.transform.scale(pygame.surfarray.make_surface(col.transpose(1, 0, 2)), self.shape.get_size())
        self.shape.blit(tint, (0, 0), special_flags=pygame.BLEND_MULT)
        self.trail.fill(self.fade, special_flags=pygame.BLEND_MULT)     # previous frame fades
        self.trail.blit(self.shape, (0, 0), special_flags=pygame.BLEND_MAX)
        return self.trail, info, count


def blur(surf, k, radius):
    small = pygame.transform.smoothscale(surf, (W // k, H // k))
    if hasattr(pygame.transform, "gaussian_blur"):
        small = pygame.transform.gaussian_blur(small, max(1, radius // k))
    return pygame.transform.smoothscale(small, (W, H))


def glow_layer(surf, amount):
    layer = surf.copy()
    layer.fill((int(255 * amount),) * 3, special_flags=pygame.BLEND_MULT)
    return layer


def window_size():
    # shrink on small / scaled displays
    info = pygame.display.Info()
    k = 1.0
    if info.current_w > 0 and info.current_h > 0:
        k = min(1.0, 0.85 * info.current_h / H, 0.95 * info.current_w / W)
    return int(W * k), int(H * k)


async def main():
    pygame.init()
    size = window_size()
    window = pygame.display.set_mode(size)
    pygame.display.set_caption(TITLE)
    screen = pygame.Surface((W, H))
    small = pygame.font.SysFont(MONO, 17)
    canvas, clock = Canvas(glyphs()), pygame.time.Clock()
    t, paused = 0.0, False
    art, info, count = canvas.draw(t)
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
         
