# -*- coding: utf-8 -*-
"""把 实验报告.md 转成排版好的 .docx，用于上交。

只针对本报告的 Markdown 子集做解析：标题、段落、表格、图片、代码块、列表、
行内粗体。样式按中文报告惯例：标题黑体、正文宋体小四、1.5 倍行距、首行缩进两字符。
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

BASE = Path(__file__).resolve().parent
MD = BASE.parent / "实验报告.md"
DOCX = BASE.parent / "实验报告.docx"

INK = RGBColor(0x1F, 0x29, 0x33)
GREY = RGBColor(0x6B, 0x77, 0x84)
ACCENT = RGBColor(0x0F, 0x76, 0x6E)

BODY_CN, BODY_EN = "宋体", "Times New Roman"
HEAD_CN, HEAD_EN = "黑体", "Arial"


def set_run(run, size=11, bold=False, cn=BODY_CN, en=BODY_EN, color=INK, italic=False):
    run.font.name = en
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:eastAsia"), cn)
    rFonts.set(qn("w:ascii"), en)
    rFonts.set(qn("w:hAnsi"), en)


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:color"), "auto")
    el.set(qn("w:fill"), hexcolor)
    tcPr.append(el)


def add_rich(par, text, size=11, cn=BODY_CN, en=BODY_EN, color=INK):
    """处理 **粗体** 与 `代码` 两种行内标记。"""
    for seg in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not seg:
            continue
        if seg.startswith("**") and seg.endswith("**"):
            r = par.add_run(seg[2:-2])
            set_run(r, size, bold=True, cn=cn, en=en, color=color)
        elif seg.startswith("`") and seg.endswith("`"):
            r = par.add_run(seg[1:-1])
            set_run(r, size - 1, cn="Consolas", en="Consolas", color=RGBColor(0x9C, 0x27, 0x30))
        else:
            r = par.add_run(seg)
            set_run(r, size, cn=cn, en=en, color=color)


def add_page_number(section):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    def fld(kind, text=None):
        run = p.add_run()
        if text is not None:
            instr = OxmlElement("w:instrText")
            instr.set(qn("xml:space"), "preserve")
            instr.text = text
            run._r.append(instr)
        else:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
            run._r.append(el)
        set_run(run, 9, cn=BODY_CN, en=BODY_EN, color=GREY)
        return run

    fld("begin")
    fld(None, "PAGE")
    fld("separate")
    r = p.add_run("1")
    set_run(r, 9, cn=BODY_CN, en=BODY_EN, color=GREY)
    fld("end")


def parse_table(lines, i):
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if not re.fullmatch(r"[\s\-:|]+", "".join(cells)):
            rows.append(cells)
        i += 1
    return rows, i


def add_table(doc, rows):
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    # 表宽撑满正文（16.4cm）
    tblW = t._tbl.tblPr.find(qn("w:tblW"))
    if tblW is None:
        tblW = OxmlElement("w:tblW")
        t._tbl.tblPr.append(tblW)
    tblW.set(qn("w:type"), "pct")
    tblW.set(qn("w:w"), "5000")
    # 表头跨页重复
    trPr = t.rows[0]._tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader")
    hdr.set(qn("w:val"), "true")
    trPr.append(hdr)

    for ri, row in enumerate(rows):
        for ci in range(ncol):
            cell = t.cell(ri, ci)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            if ri == 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                shade(cell, "EEF7F4")
                add_rich(p, row[ci] if ci < len(row) else "", size=9.5, cn=HEAD_CN, en=HEAD_EN)
            else:
                add_rich(p, row[ci] if ci < len(row) else "", size=9.5)
    # 表后留白
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def convert():
    lines = MD.read_text(encoding="utf-8").splitlines()
    doc = Document()

    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.4)
    sec.left_margin = sec.right_margin = Cm(2.3)

    normal = doc.styles["Normal"]
    normal.font.name = BODY_EN
    normal.font.size = Pt(11)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_CN)

    add_page_number(sec)

    i = 0
    in_code = False
    code_buf = []
    in_meta = True          # 标题与第一节之间的元信息区，居中显示

    while i < len(lines):
        ln = lines[i].rstrip()
        stripped = ln.strip()

        # 代码块
        if stripped.startswith("```"):
            if in_code:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.6)
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.line_spacing = 1.15
                r = p.add_run("\n".join(code_buf))
                set_run(r, 9, cn="Consolas", en="Consolas", color=RGBColor(0x24, 0x29, 0x2E))
                code_buf, in_code = [], False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        # HTML 注释占位
        if stripped.startswith("<!--"):
            i += 1
            continue

        # 分隔线
        if set(stripped) <= {"-", "*"} and len(stripped) >= 3:
            i += 1
            continue

        # 标题
        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level >= 2:
                in_meta = False
            if level == 1:
                doc.add_paragraph()
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(6)
                add_rich(p, text, size=19, cn=HEAD_CN, en=HEAD_EN, color=INK)
                for r in p.runs:
                    r.font.bold = True
            else:
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(14 if level == 2 else 9)
                p.paragraph_format.space_after = Pt(6 if level == 2 else 4)
                p.paragraph_format.keep_with_next = True
                add_rich(p, text, size=14 if level == 2 else 11.5,
                         cn=HEAD_CN, en=HEAD_EN, color=ACCENT if level == 2 else INK)
                for r in p.runs:
                    r.font.bold = True
            i += 1
            continue

        # 表格
        if stripped.startswith("|"):
            rows, i = parse_table(lines, i)
            add_table(doc, rows)
            continue

        # 图片
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if m:
            cap, rel = m.group(1), m.group(2)
            img = (MD.parent / rel).resolve()
            if not img.exists():
                img = (MD.parent / "count_tool" / rel).resolve()
            if img.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(2)
                p.add_run().add_picture(str(img), width=Cm(15.4))
                if cap:
                    cp = doc.add_paragraph()
                    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    cp.paragraph_format.space_after = Pt(10)
                    add_rich(cp, cap, size=9, cn=BODY_CN, en=BODY_EN, color=GREY)
            i += 1
            continue

        # 列表
        m = re.match(r"^(\s*)[-*]\s+(.*)$", ln)
        if m:
            indent = len(m.group(1)) // 2
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75 + indent * 0.6)
            p.paragraph_format.first_line_indent = Cm(-0.4)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.4
            add_rich(p, "· " + m.group(2))
            i += 1
            continue

        m = re.match(r"^\s*(\d+)[.)]\s+(.*)$", ln)
        if m:
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.85)
            p.paragraph_format.first_line_indent = Cm(-0.5)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.4
            add_rich(p, "%s. %s" % (m.group(1), m.group(2)))
            i += 1
            continue

        # 正文段落
        if in_meta and stripped.startswith("**"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.3
            add_rich(p, stripped, size=10.5, color=GREY)
            i += 1
            continue

        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Cm(0.74)
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        p.paragraph_format.space_after = Pt(6)
        add_rich(p, stripped)
        i += 1

    doc.save(DOCX)
    print("已生成", DOCX.name, "%.1f KB" % (DOCX.stat().st_size / 1024))


if __name__ == "__main__":
    convert()
