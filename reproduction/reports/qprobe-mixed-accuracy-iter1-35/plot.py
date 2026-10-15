#!/usr/bin/env python3
"""Reproduce the selected Top512/E0, pre-fit, iterations 1--35 figure.

Run with the existing bundled Python and Pillow; no dependency installation:
    python3 plot.py
To rebuild the sanitized scalar data from the three local source receipts:
    python3 plot.py --refresh-source --source-dir /path/to/preparation

The default run needs only this file and figure-data.json. macOS CoreText turns
all SVG lettering into glyph outlines. Pillow renders the matching 200 dpi PNG.
Only existing aggregate scalars are used; this script has no remote operations,
model execution, sampling, evaluation, QA, question identifiers, or trajectories.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path('/Users/shaoshitong/Documents/Search-G1/preparation')
SOURCE_NAMES = (
    'qprobe-prefit-accuracy-and-mixed-fractions-iter1-35.json',
    'qprobe-mixed-accuracy-correlation-analysis-iter1-35.json',
    'qprobe-prefit-class-distribution-summary-iter1-35.json',
)
W, H, DPI = 1040, 800, 200
INK = '#20313E'
MUTED = '#566570'
AXIS = '#6B7781'
GRID = '#E6EBEE'
BLUE = '#0072B2'
ORANGE = '#CE8312'
GREEN = '#008C68'
PURPLE = '#74469B'
PURPLE_LIGHT = '#CCB9DA'
TEAL = '#007D8D'
TEAL_LIGHT = '#A7CCD1'


def trailing_mean(values, count=5):
    return [None if i + 1 < count else statistics.fmean(values[i + 1 - count:i + 1])
            for i in range(len(values))]


def sanitized_data(source_dir):
    source_paths = [source_dir / name for name in SOURCE_NAMES]
    source_bytes = [path.read_bytes() for path in source_paths]
    accuracy, correlation, classes = [json.loads(blob) for blob in source_bytes]
    raw_rows = sorted(accuracy['rows'], key=lambda row: row['iteration'])
    if [row['iteration'] for row in raw_rows] != list(range(1, 36)):
        raise ValueError('The requested figure requires exactly iterations 1 through 35.')
    if any(row['questions'] != 512 for row in raw_rows):
        raise ValueError('This figure is scoped to 512 selected questions per iteration.')

    source_keys = {
        'mixed_fraction': 'mixed_fraction',
        'binary_accuracy': 'binary_accuracy_threshold05',
        'six_class_accuracy': 'six_accuracy',
        'binary_auc': 'binary_auc_selected',
        'std_pearson': 'sample_correlation_predicted_std_actual_std',
    }
    rows = []
    for raw in raw_rows:
        row = {'iteration': int(raw['iteration']), 'selected_questions': int(raw['questions'])}
        row.update({out: float(raw[src]) for out, src in source_keys.items()})
        rows.append(row)

    rolling_names = {
        'mixed_vs_binary_pearson10': 'binary_accuracy_threshold05',
        'mixed_vs_six_class_pearson10': 'six_accuracy',
    }
    for out, src in rolling_names.items():
        windows = correlation['rolling_correlations'][src]['10']
        by_end = {int(item['last_iteration']): item for item in windows}
        for row in rows:
            iteration = row['iteration']
            window = by_end.get(iteration)
            row[out] = None if window is None else float(window['pearson'])
            if iteration >= 10 and (window is None or window['first_iteration'] != iteration - 9):
                raise ValueError('A rolling Pearson window must contain 10 consecutive iterations.')

    for key in ('binary_auc', 'std_pearson'):
        mean = trailing_mean([row[key] for row in rows], 5)
        for row, value in zip(rows, mean):
            row[key + '_trailing_mean5'] = value

    block_keys = (
        'mixed_fraction', 'binary_accuracy', 'six_accuracy',
        'binary_per_round_majority_baseline', 'six_majority_baseline',
        'selected_binary_auc_mean', 'predicted_vs_actual_std_correlation_mean',
    )
    blocks = [{
        'first_iteration': int(block['iterations'][0]),
        'last_iteration': int(block['iterations'][1]),
        **{key: float(block[key]) for key in block_keys},
    } for block in classes['blocks']]

    return {
        'schema': 'qprobe_selected_top512_e0_prefit_scalar_figure_v1',
        'iteration_start': 1,
        'iteration_end': 35,
        'selected_questions_per_iteration': 512,
        'candidate_questions_per_iteration': 5120,
        'scope': 'Selected Top512/E0 only; immutable pre-generation, pre-fit predictions.',
        'binary_definition': 'sum(p1,p2,p3,p4) >= 0.5; actual mixed = correct count in 1..4 of 5.',
        'six_class_definition': 'argmax(p0,p1,p2,p3,p4,p5) predicts actual correct count among 5.',
        'panel_b': '10-iteration trailing Pearson, assigned to window end; starts at iteration 10.',
        'panels_c_d_mean': '5-iteration trailing arithmetic mean; starts at iteration 5.',
        'sequence': 'Iterations 1..5 and CP5 restore from iteration 6 form one logical sequence.',
        'limitations': [
            'The unselected 4608 candidates have no actual labels; not full-5120 accuracy.',
            'Rolling windows overlap; correlations are descriptive, not causal evidence.',
            'Existing aggregate data only; no new sampling, model forward, or evaluation.',
        ],
        'new_sampling_or_evaluation': False,
        'sources': [
            {'filename': path.name, 'sha256': hashlib.sha256(blob).hexdigest()}
            for path, blob in zip(source_paths, source_bytes)
        ],
        'rows': rows,
        'class_distribution_block_scalars': blocks,
    }


class CGPoint(ct.Structure):
    _fields_ = [('x', ct.c_double), ('y', ct.c_double)]


class CGSize(ct.Structure):
    _fields_ = [('width', ct.c_double), ('height', ct.c_double)]


class CGPathElement(ct.Structure):
    _fields_ = [('type', ct.c_int), ('points', ct.POINTER(CGPoint))]


PathCallback = ct.CFUNCTYPE(None, ct.c_void_p, ct.POINTER(CGPathElement))


class OutlineFonts:
    """Native macOS font contours and advances, paired with Pillow fonts."""

    def __init__(self):
        self.cf = ct.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        self.coretext = ct.CDLL('/System/Library/Frameworks/CoreText.framework/CoreText')
        self.cg = ct.CDLL('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics')
        self.cf.CFStringCreateWithCString.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_uint32]
        self.cf.CFStringCreateWithCString.restype = ct.c_void_p
        self.cf.CFRelease.argtypes = [ct.c_void_p]
        self.coretext.CTFontCreateWithName.argtypes = [ct.c_void_p, ct.c_double, ct.c_void_p]
        self.coretext.CTFontCreateWithName.restype = ct.c_void_p
        self.coretext.CTFontGetGlyphsForCharacters.argtypes = [
            ct.c_void_p, ct.POINTER(ct.c_uint16), ct.POINTER(ct.c_uint16), ct.c_long]
        self.coretext.CTFontGetGlyphsForCharacters.restype = ct.c_bool
        self.coretext.CTFontGetAdvancesForGlyphs.argtypes = [
            ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint16), ct.POINTER(CGSize), ct.c_long]
        self.coretext.CTFontGetAdvancesForGlyphs.restype = ct.c_double
        self.coretext.CTFontCreatePathForGlyph.argtypes = [ct.c_void_p, ct.c_uint16, ct.c_void_p]
        self.coretext.CTFontCreatePathForGlyph.restype = ct.c_void_p
        self.cg.CGPathApply.argtypes = [ct.c_void_p, ct.c_void_p, PathCallback]
        self.cache = {}
        self.font_cache = {}
        self.pil_cache = {}
        self.definitions = []

    def native_font(self, size, bold=False):
        key = (size, bold)
        if key not in self.font_cache:
            name = b'Arial-BoldMT' if bold else b'ArialMT'
            string = self.cf.CFStringCreateWithCString(None, name, 0x08000100)
            self.font_cache[key] = self.coretext.CTFontCreateWithName(string, size, None)
            self.cf.CFRelease(string)
        return self.font_cache[key]

    def pil_font(self, size, bold, scale):
        key = (size, bold, scale)
        if key not in self.pil_cache:
            name = 'Arial Bold.ttf' if bold else 'Arial.ttf'
            self.pil_cache[key] = ImageFont.truetype(
                '/System/Library/Fonts/Supplemental/' + name, round(size * scale))
        return self.pil_cache[key]

    def glyph(self, char, size, bold=False):
        key = (char, size, bold)
        if key in self.cache:
            return self.cache[key]
        font = self.native_font(size, bold)
        characters = (ct.c_uint16 * 1)(ord(char))
        glyphs = (ct.c_uint16 * 1)()
        if not self.coretext.CTFontGetGlyphsForCharacters(font, characters, glyphs, 1):
            raise ValueError('Font missing character: ' + repr(char))
        advance = (CGSize * 1)()
        self.coretext.CTFontGetAdvancesForGlyphs(font, 0, glyphs, advance, 1)
        path = self.coretext.CTFontCreatePathForGlyph(font, glyphs[0], None)
        commands = []

        def point(p):
            return f'{p.x:.4f},{-p.y:.4f}'

        @PathCallback
        def consume(_info, element_ptr):
            element = element_ptr.contents
            points = element.points
            if element.type == 0:
                commands.append('M' + point(points[0]))
            elif element.type == 1:
                commands.append('L' + point(points[0]))
            elif element.type == 2:
                commands.append('Q' + point(points[0]) + ' ' + point(points[1]))
            elif element.type == 3:
                commands.append('C' + ' '.join(point(points[i]) for i in range(3)))
            elif element.type == 4:
                commands.append('Z')

        glyph_id = None
        if path:
            self.cg.CGPathApply(path, None, consume)
            self.cf.CFRelease(path)
            glyph_id = 'glyph' + str(len(self.definitions))
            self.definitions.append(f'<path id="{glyph_id}" d="{" ".join(commands)}"/>')
        result = (glyph_id, float(advance[0].width))
        self.cache[key] = result
        return result


class Canvas:
    def __init__(self):
        self.scale = DPI / 72
        self.image = Image.new('RGB', (round(W * self.scale), round(H * self.scale)), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.fonts = OutlineFonts()
        self.elements = []

    def line(self, points, color, width=1, dash=None):
        points = [(float(x), float(y)) for x, y in points]
        coordinates = ' '.join(f'{x:.3f},{y:.3f}' for x, y in points)
        extra = '' if dash is None else f' stroke-dasharray="{dash[0]} {dash[1]}"'
        self.elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                             f'stroke-width="{width}" stroke-linejoin="round"{extra}/>')
        if dash is None:
            self.draw.line([(x * self.scale, y * self.scale) for x, y in points],
                           fill=color, width=max(1, round(width * self.scale)), joint='curve')
            return
        phase = 0.0
        period = sum(dash)
        for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
            length = math.hypot(x2 - x1, y2 - y1)
            distance = 0.0
            while distance < length - 1e-9:
                within = phase % period
                active = within < dash[0]
                remaining = (dash[0] - within) if active else (period - within)
                end = min(length, distance + remaining)
                if active:
                    a = distance / length
                    b = end / length
                    self.draw.line([
                        ((x1 + a * (x2 - x1)) * self.scale, (y1 + a * (y2 - y1)) * self.scale),
                        ((x1 + b * (x2 - x1)) * self.scale, (y1 + b * (y2 - y1)) * self.scale),
                    ], fill=color, width=max(1, round(width * self.scale)))
                phase += end - distance
                distance = end

    def circle(self, x, y, radius, color, fill=True, width=1):
        self.elements.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius}" '
                             f'fill="{color if fill else "white"}" stroke="{color}" stroke-width="{width}"/>')
        r = radius * self.scale
        cx, cy = x * self.scale, y * self.scale
        self.draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                          fill=color if fill else 'white', outline=color,
                          width=max(1, round(width * self.scale)))

    def text(self, x, y, text, size=10, color=INK, bold=False, anchor='start', rotate=0):
        glyphs = [self.fonts.glyph(char, size, bold) for char in text]
        width = sum(advance for _, advance in glyphs)
        fraction = {'start': 0, 'middle': 0.5, 'end': 1}[anchor]
        offset = -width * fraction
        transform = f'translate({x:.3f} {y:.3f})'
        if rotate:
            transform += f' rotate({rotate})'
        uses = []
        for (glyph_id, advance) in glyphs:
            if glyph_id is not None:
                uses.append(f'<use href="#{glyph_id}" transform="translate({offset:.4f} 0)"/>')
            offset += advance
        self.elements.append(f'<g fill="{color}" transform="{transform}" aria-label="{html.escape(text)}">'
                             + ''.join(uses) + '</g>')
        font = self.fonts.pil_font(size, bold, self.scale)
        if not rotate:
            px = (x - width * fraction) * self.scale
            for char, (_, advance) in zip(text, glyphs):
                self.draw.text((px, y * self.scale), char, fill=color, font=font, anchor='ls')
                px += advance * self.scale
        else:
            if rotate != -90:
                raise ValueError('The only required rotated text is a -90 degree axis label.')
            pad = 4
            layer_width = math.ceil((width + 2 * pad) * self.scale)
            layer_height = math.ceil(4 * size * self.scale)
            layer = Image.new('RGBA', (layer_width, layer_height), (255, 255, 255, 0))
            draw = ImageDraw.Draw(layer)
            px = pad * self.scale
            baseline = 2 * size * self.scale
            for char, (_, advance) in zip(text, glyphs):
                draw.text((px, baseline), char, fill=color, font=font, anchor='ls')
                px += advance * self.scale
            transformed = layer.transpose(Image.Transpose.ROTATE_90)
            anchor_x = baseline
            anchor_y = layer_width - (pad + width * fraction) * self.scale
            self.image.paste(transformed, (round(x * self.scale - anchor_x),
                                           round(y * self.scale - anchor_y)), transformed)
        return width

    def save(self, out_dir):
        title = 'QProbe: mixed rate, accuracy and discrimination, iterations 1-35'
        desc = ('Selected Top512/E0, pre-fit. Existing aggregate data only; no new sampling. '
                'A: mixed fraction and accuracies. B: trailing 10-iteration Pearson. '
                'C: selected binary ROC AUC. D: within-iteration sample-std Pearson. '
                'Trailing 5-iteration means start at iteration 5. Rolling windows overlap. '
                'Unselected 4608 candidates have no actual labels; not full-5120 accuracy. '
                'Iterations 1-5 and CP5 restore from iteration 6 form one logical sequence. '
                'All SVG lettering is embedded as vector glyph outlines.')
        svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
               f'<svg xmlns="http://www.w3.org/2000/svg" width="{W/72:.6f}in" height="{H/72:.6f}in" '
               f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">\n'
               f'<title id="title">{html.escape(title)}</title>\n'
               f'<desc id="desc">{html.escape(desc)}</desc>\n'
               '<defs>' + '\n'.join(self.fonts.definitions) + '</defs>\n'
               f'<rect width="{W}" height="{H}" fill="white"/>\n'
               + '\n'.join(self.elements) + '\n</svg>\n')
        (out_dir / 'figure.svg').write_text(svg, encoding='utf-8')
        self.image.save(out_dir / 'figure.png', dpi=(DPI, DPI), optimize=True)


class Axes:
    def __init__(self, canvas, left, top, width, height, xlim, ylim,
                 xticks, yticks, ylabel, xlabel='Iteration', percent=False, decimals=2):
        self.c = canvas
        self.left, self.top = left, top
        self.width, self.height = width, height
        self.xlim, self.ylim = xlim, ylim
        for value in yticks:
            py = self.y(value)
            canvas.line([(left, py), (left + width, py)], GRID, 0.65)
            label = str(round(value * 100)) if percent else f'{value:.{decimals}f}'
            if not percent and decimals == 1 and value == 0:
                label = '0'
            canvas.text(left - 10, py + 3.3, label, 9.5, MUTED, anchor='end')
        for value in xticks:
            px = self.x(value)
            canvas.line([(px, top + height), (px, top + height + 4)], AXIS, 0.75)
            canvas.text(px, top + height + 18, str(value), 9.5, MUTED, anchor='middle')
        canvas.line([(left, top), (left, top + height), (left + width, top + height)], AXIS, 0.8)
        canvas.text(left + width / 2, top + height + 39, xlabel, 10, MUTED, anchor='middle')
        canvas.text(left - 46, top + height / 2, ylabel, 10, MUTED, anchor='middle', rotate=-90)

    def x(self, value):
        return self.left + (value - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.width

    def y(self, value):
        return self.top + (self.ylim[1] - value) / (self.ylim[1] - self.ylim[0]) * self.height

    def series(self, xs, ys, color, width=1.6, dash=None, markers=0, fill=True):
        points = [(self.x(x), self.y(y)) for x, y in zip(xs, ys) if y is not None]
        self.c.line(points, color, width, dash)
        if markers:
            for x, y in points:
                self.c.circle(x, y, markers, color, fill, 0.8)

    def reference(self, value, color=AXIS, dash=(4, 3)):
        self.c.line([(self.left, self.y(value)), (self.left + self.width, self.y(value))], color, 0.95, dash)


def legend(canvas, x, y, items, gap=20):
    for label, color, dash, marker, fill in items:
        canvas.line([(x, y - 3.5), (x + 22, y - 3.5)], color, 1.6, dash)
        if marker:
            canvas.circle(x + 11, y - 3.5, marker, color, fill, 0.8)
        size = canvas.text(x + 29, y, label, 9.5)
        x += 29 + size + gap


def make_figure(data, out_dir):
    rows = data['rows']
    if [row['iteration'] for row in rows] != list(range(1, 36)):
        raise ValueError('Expected the sanitized 35-iteration scalar series.')
    c = Canvas()
    xs = [row['iteration'] for row in rows]
    values = lambda key: [row[key] for row in rows]

    c.text(52, 30, 'QProbe: mixed rate, accuracy and discrimination', 18, bold=True)
    c.text(52, 52, 'Iterations 1-35  |  selected Top512 / E0  |  pre-fit predictions  |  existing data only; no new sampling',
           10.5, MUTED)

    c.text(76, 84, '(A)  Mixed rate and classification accuracy', 12, bold=True)
    c.text(76, 101, 'Binary: sum(p1..p4) >= 0.5; six-class: argmax(p0..p5)', 9.5, MUTED)
    a = Axes(c, 76, 119, 414, 194, (1, 35), (0.20, 0.65),
             [1, 5, 10, 15, 20, 25, 30, 35], [0.20, 0.30, 0.40, 0.50, 0.60],
             'Fraction / accuracy (%)', percent=True)
    a.series(xs, values('six_class_accuracy'), GREEN, 1.7, markers=1.7)
    a.series(xs, values('mixed_fraction'), BLUE, 1.7, markers=1.8)
    a.series(xs, values('binary_accuracy'), ORANGE, 1.5, dash=(4.5, 2.5), markers=2.6, fill=False)
    legend(c, 76, 374, [
        ('Actual mixed rate', BLUE, None, 1.8, True),
        ('Binary accuracy', ORANGE, (4.5, 2.5), 2.6, False),
        ('Six-class accuracy', GREEN, None, 1.7, True),
    ], gap=13)

    c.text(600, 84, '(B)  Mixed rate vs. accuracy: rolling Pearson', 12, bold=True)
    c.text(600, 101, '10-iteration trailing windows; each point is the window end', 9.5, MUTED)
    b = Axes(c, 600, 119, 414, 194, (10, 35), (-1, 1),
             [10, 15, 20, 25, 30, 35], [-1, -0.5, 0, 0.5, 1],
             'Pearson r', xlabel='Window end iteration', decimals=1)
    b.reference(0)
    b.series(xs[9:], values('mixed_vs_binary_pearson10')[9:], ORANGE, 1.9, markers=1.7)
    b.series(xs[9:], values('mixed_vs_six_class_pearson10')[9:], GREEN, 1.9, markers=1.7)
    legend(c, 600, 374, [
        ('Mixed vs. binary accuracy', ORANGE, None, 1.7, True),
        ('Mixed vs. six-class accuracy', GREEN, None, 1.7, True),
    ], gap=16)

    c.text(76, 422, '(C)  Binary discrimination within selected 512', 12, bold=True)
    c.text(76, 439, 'ROC AUC of pre-fit mixed probability; chance = 0.50', 9.5, MUTED)
    panel_c = Axes(c, 76, 457, 414, 194, (1, 35), (0.46, 0.65),
                   [1, 5, 10, 15, 20, 25, 30, 35], [0.50, 0.55, 0.60, 0.65], 'ROC AUC')
    panel_c.reference(0.50)
    panel_c.series(xs, values('binary_auc'), PURPLE_LIGHT, 1.1, markers=1.35)
    panel_c.series(xs[4:], values('binary_auc_trailing_mean5')[4:], PURPLE, 2.2, markers=1.7)
    legend(c, 76, 712, [
        ('Per iteration', PURPLE_LIGHT, None, 1.35, True),
        ('5-iteration mean', PURPLE, None, 1.7, True),
        ('Chance (0.50)', AXIS, (4, 3), 0, True),
    ], gap=18)

    c.text(600, 422, '(D)  Predicted vs. actual sample std', 12, bold=True)
    c.text(600, 439, 'Within-iteration Pearson across the selected 512 questions', 9.5, MUTED)
    d = Axes(c, 600, 457, 414, 194, (1, 35), (-0.025, 0.25),
             [1, 5, 10, 15, 20, 25, 30, 35], [0, 0.05, 0.10, 0.15, 0.20, 0.25], 'Pearson r')
    d.reference(0)
    d.series(xs, values('std_pearson'), TEAL_LIGHT, 1.1, markers=1.35)
    d.series(xs[4:], values('std_pearson_trailing_mean5')[4:], TEAL, 2.2, markers=1.7)
    legend(c, 600, 712, [
        ('Per iteration', TEAL_LIGHT, None, 1.35, True),
        ('5-iteration mean', TEAL, None, 1.7, True),
        ('Zero', AXIS, (4, 3), 0, True),
    ], gap=18)

    c.line([(52, 734), (1014, 734)], GRID, 0.8)
    c.text(52, 753, 'Scope: only selected Top512/E0 are labeled; unselected 4608 have no actual labels. These are not full-5120 accuracies.',
           9, MUTED)
    c.text(52, 769, 'Rolling windows overlap; correlations are descriptive. Iterations 1-5 and CP5 restore from iteration 6 form one logical sequence.',
           9, MUTED)
    c.text(52, 785, 'Means are trailing and begin at iteration 5; rolling Pearson begins at iteration 10. No new sampling, model forward or evaluation.',
           9, MUTED)
    c.save(out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-source', action='store_true', help='Extract only allowed aggregate scalars from the local source receipts.')
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=HERE)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / 'figure-data.json'
    if args.refresh_source:
        data = sanitized_data(args.source_dir)
        data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    else:
        data = json.loads(data_path.read_text(encoding='utf-8'))
    make_figure(data, args.output_dir)
    for name in ('plot.py', 'figure-data.json', 'figure.png', 'figure.svg'):
        path = args.output_dir / name
        if path.exists():
            print(f'{path}\t{path.stat().st_size} bytes')
    first, last = data['rows'][0], data['rows'][-1]
    print('First iteration:', json.dumps(first, sort_keys=True))
    print('Last iteration:', json.dumps(last, sort_keys=True))


if __name__ == '__main__':
    main()
