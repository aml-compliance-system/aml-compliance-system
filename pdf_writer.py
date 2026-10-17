"""Dependency-free PDF writer for SAR exports.

Produces a valid multi-page PDF 1.4 document with Helvetica text, simple tables,
a DRAFT watermark and page numbers. This is intentionally small: the project
environment cannot install reportlab/weasyprint, and SAR exports must remain
available locally.
"""

from __future__ import annotations

from io import BytesIO
from typing import List, Optional, Sequence


def _pdf_text(value: object) -> str:
    text = str(value if value is not None else "")
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"', "\u2022": "-", "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.encode("latin-1", "replace").decode("latin-1")


def _escape(value: object) -> str:
    return _pdf_text(value).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _width(text: str, size: float, bold: bool = False) -> float:
    # Conservative Helvetica approximation; wrapping is intentionally safe.
    factor = 0.54 if bold else 0.50
    return len(text) * size * factor


class SimplePDF:
    page_width = 595
    page_height = 842
    margin = 48

    def __init__(self, *, status: str = "DRAFT"):
        self.status = str(status or "DRAFT").upper()
        self.pages: List[List[str]] = []
        self.y = 0.0
        self._new_page()

    def _new_page(self) -> None:
        self.pages.append([])
        self.y = self.page_height - self.margin

    def _ensure(self, amount: float) -> None:
        if self.y - amount < self.margin:
            self._new_page()

    def _draw(self, text: str, x: float, y: float, size: float, bold: bool = False,
              gray: Optional[float] = None) -> None:
        font = "F2" if bold else "F1"
        commands = []
        if gray is not None:
            commands.append(f"{gray:.3f} g")
        commands.append(
            f"BT /{font} {size:.2f} Tf {x:.2f} {y:.2f} Td ({_escape(text)}) Tj ET"
        )
        if gray is not None:
            commands.append("0 g")
        self.pages[-1].extend(commands)

    def wrap(self, text: str, size: float, max_width: float, bold: bool = False) -> List[str]:
        text = _pdf_text(text).replace("\t", "    ")
        if not text:
            return [""]
        all_lines: List[str] = []
        for logical_line in text.split("\n"):
            words = logical_line.split(" ")
            lines: List[str] = []
            current = ""
            for word in words:
                while _width(word, size, bold) > max_width and len(word) > 1:
                    cut = max(1, int(len(word) * max_width / max(_width(word, size, bold), 1)))
                    piece, word = word[:cut], word[cut:]
                    if current:
                        lines.append(current)
                        current = ""
                    lines.append(piece)
                candidate = word if not current else f"{current} {word}"
                if _width(candidate, size, bold) <= max_width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current or not lines:
                lines.append(current)
            all_lines.extend(lines)
        return all_lines or [""]

    def text(self, text: object, *, size: float = 9.5, leading: float = 13.5,
             bold: bool = False, indent: float = 0.0) -> None:
        max_width = self.page_width - (2 * self.margin) - indent
        for line in self.wrap(str(text or ""), size, max_width, bold=bold):
            self._ensure(leading)
            self._draw(line, self.margin + indent, self.y, size, bold=bold)
            self.y -= leading

    def paragraph(self, text: object, *, size: float = 9.5) -> None:
        self.text(text, size=size, leading=size * 1.45)
        self.y -= 4

    def heading(self, text: object, *, size: float = 13.0) -> None:
        self._ensure(size + 14)
        self.y -= 4
        self._draw(str(text or ""), self.margin, self.y, size, bold=True)
        self.y -= size + 7

    def subheading(self, text: object, *, size: float = 10.5) -> None:
        self._ensure(size + 10)
        self._draw(str(text or ""), self.margin, self.y, size, bold=True)
        self.y -= size + 5

    def bullets(self, items: Sequence[object], *, size: float = 9.5) -> None:
        for item in items:
            lines = self.wrap(str(item or ""), size, self.page_width - 2 * self.margin - 14)
            for index, line in enumerate(lines):
                self._ensure(size * 1.45)
                prefix = "- " if index == 0 else "  "
                self._draw(prefix + line, self.margin, self.y, size)
                self.y -= size * 1.45
        self.y -= 4

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
        if not headers:
            return
        usable = self.page_width - 2 * self.margin
        columns = len(headers)
        widths = [usable / columns for _ in range(columns)]
        row_size = 8.0
        leading = 10.0
        padding = 3.0

        def draw_row(values: Sequence[object], bold: bool = False, shade: bool = False) -> None:
            wrapped = [
                self.wrap("" if value is None else str(value), row_size, widths[index] - 8, bold)
                for index, value in enumerate(values)
            ]
            height = max(len(lines) for lines in wrapped) * leading + 2 * padding
            self._ensure(height + 2)
            top = self.y
            if shade:
                self.pages[-1].append(
                    f"0.96 g {self.margin:.2f} {top - height:.2f} {usable:.2f} {height:.2f} re f 0 g"
                )
            x = self.margin
            for index, lines in enumerate(wrapped):
                text_y = top - padding - row_size
                for line in lines:
                    self._draw(line, x + 4, text_y, row_size, bold=bold)
                    text_y -= leading
                x += widths[index]
            self.pages[-1].append(
                f"0.5 w 0.75 G {self.margin:.2f} {top - height:.2f} m "
                f"{self.margin + usable:.2f} {top - height:.2f} l S 0 G"
            )
            self.y = top - height

        draw_row(headers, bold=True, shade=True)
        for row in rows:
            draw_row(row)
        self.y -= 6

    def spacer(self, amount: float = 8.0) -> None:
        self.y -= amount

    def render(self) -> bytes:
        total = len(self.pages)
        streams: List[str] = []
        for index, commands in enumerate(self.pages, start=1):
            header = []
            if self.status:
                header.append(
                    "q 0.90 0.90 0.90 rg BT /F2 64 Tf "
                    f"0.707 0.707 -0.707 0.707 120 380 Tm ({_escape(self.status)}) Tj ET Q"
                )
            header.append(
                f"BT /F1 8 Tf {self.margin} 28 Td "
                f"(Page {index} of {total}) Tj ET"
            )
            streams.append("\n".join(header + commands))

        objects: List[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(total))
        objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {total} >>".encode("latin-1"))
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        for index, stream in enumerate(streams):
            page_obj = 5 + 2 * index
            content_obj = page_obj + 1
            objects.append(
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.page_width} {self.page_height}] "
                    f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                    f"/Contents {content_obj} 0 R >>"
                ).encode("latin-1")
            )
            payload = stream.encode("latin-1", "replace")
            objects.append(
                b"<< /Length " + str(len(payload)).encode("ascii") + b" >>\nstream\n"
                + payload + b"\nendstream"
            )

        buffer = BytesIO()
        buffer.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for number, body in enumerate(objects, start=1):
            offsets.append(buffer.tell())
            buffer.write(f"{number} 0 obj\n".encode("ascii"))
            buffer.write(body)
            buffer.write(b"\nendobj\n")
        xref_start = buffer.tell()
        buffer.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        buffer.write(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            buffer.write(f"{offset:010d} 00000 n \n".encode("ascii"))
        buffer.write(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_start}\n%%EOF\n"
            ).encode("ascii")
        )
        return buffer.getvalue()
