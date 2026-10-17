"""Dependency-free DOCX writer for SAR exports.

Generates a minimal, valid OOXML package with paragraphs, headings, bullet
lists and tables. Keeping this in-tree avoids adding a runtime dependency in an
environment where package installation is unavailable.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from typing import Any, Iterable, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

Block = Tuple[str, Any]


def _xml(text: Any) -> str:
    return escape(str(text or ""), {'"': "&quot;"})


def _run(text: str, bold: bool = False, size: Optional[int] = None) -> str:
    props = []
    if bold:
        props.append("<w:b/>")
    if size:
        props.append(f'<w:sz w:val="{int(size) * 2}"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f"<w:r>{rpr}<w:t xml:space=\"preserve\">{_xml(text)}</w:t></w:r>"


def _paragraph(text: str, style: Optional[str] = None, bold: bool = False,
               size: Optional[int] = None, after: int = 120) -> str:
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{_xml(style)}"/>')
    ppr.append(f'<w:spacing w:after="{after}" w:line="276" w:lineRule="auto"/>')
    return f"<w:p><w:pPr>{''.join(ppr)}</w:pPr>{_run(text, bold=bold, size=size)}</w:p>"


def _cell(text: str, bold: bool = False, width: int = 2000) -> str:
    return (
        "<w:tc><w:tcPr>"
        f'<w:tcW w:w="{width}" w:type="dxa"/>'
        '<w:tcBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        "</w:tcBorders>"
        "</w:tcPr>"
        f"<w:p><w:pPr><w:spacing w:after=\"40\"/></w:pPr>{_run(text, bold=bold, size=8)}</w:p>"
        "</w:tc>"
    )


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not headers:
        return ""
    widths = [int(9000 / len(headers))] * len(headers)
    header_cells = "".join(
        _cell(str(header), bold=True, width=width)
        for header, width in zip(headers, widths)
    )
    body = []
    for row in rows:
        cells = "".join(
            _cell("" if value is None else str(value), width=width)
            for value, width in zip(row, widths)
        )
        body.append(f"<w:tr>{cells}</w:tr>")
    return (
        "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/>"
        '<w:tblW w:w="0" w:type="auto"/></w:tblPr>'
        f"<w:tr>{header_cells}</w:tr>{''.join(body)}</w:tbl>"
    )


def build_docx(
    *,
    title: str,
    subtitle: str,
    status: str,
    blocks: Iterable[Block],
) -> bytes:
    body: List[str] = [
        _paragraph(title, style="Title", bold=True, size=20, after=80),
        _paragraph(subtitle, size=10, after=80),
        _paragraph(f"Status: {status.upper()} — draft for investigator review", bold=True, size=9, after=200),
    ]
    for kind, value in blocks:
        if kind == "heading":
            body.append(_paragraph(str(value), style="Heading1", bold=True, size=13, after=100))
        elif kind == "subheading":
            body.append(_paragraph(str(value), style="Heading2", bold=True, size=11, after=80))
        elif kind == "paragraph":
            text = str(value or "")
            for part in text.split("\n"):
                body.append(_paragraph(part, after=100))
        elif kind == "bullets":
            for item in value or []:
                body.append(_paragraph(f"• {item}", after=60))
        elif kind == "table":
            headers, rows = value
            body.append(_table(headers, rows))
            body.append(_paragraph("", after=80))
        elif kind == "spacer":
            body.append(_paragraph("", after=80))

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}"
        "<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
        "<w:pgMar w:top=\"1134\" w:right=\"1134\" w:bottom=\"1134\" w:left=\"1134\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/></w:sectPr>"
        "</w:body></w:document>"
    )
    styles = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/><w:sz w:val="20"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="40"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="22"/></w:rPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/><w:left w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/><w:bottom w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/><w:right w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/><w:insideV w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""
    core = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:title>SAR draft</dc:title>
  <dc:creator>FINDETECT</dc:creator>
  <cp:lastModifiedBy>FINDETECT</cp:lastModifiedBy>
</cp:coreProperties>"""
    app = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>FINDETECT</Application>
</Properties>"""

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
        archive.writestr("word/styles.xml", styles)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
        archive.writestr("docProps/core.xml", core)
        archive.writestr("docProps/app.xml", app)
    return buffer.getvalue()
